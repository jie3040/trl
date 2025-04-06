# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available

from ..core import masked_mean, masked_whiten
from ..models import create_reference_model
from ..models.utils import unwrap_model_for_generation
from .ppo_config import PPOConfig
from .utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    log_table_to_comet_experiment,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
)


# PEFT (Parameter-Efficient Fine-Tuning) 是一种高效的模型微调方法
# 它允许我们只微调模型的一小部分参数，而不是整个模型
# 这样可以大大减少计算资源和内存需求
if is_peft_available():
    # 导入 PEFT 相关的类和函数
    # PeftConfig: PEFT 的配置类，用于设置微调参数
    # PeftModel: PEFT 模型类，用于包装原始模型
    # get_peft_model: 用于创建 PEFT 模型的函数
    from peft import PeftConfig, PeftModel, get_peft_model

# Weights & Biases (wandb) 是一个用于机器学习实验跟踪和可视化的工具
# 它可以帮助我们：
# 1. 跟踪训练过程中的各种指标（如损失值、准确率等）
# 2. 可视化训练曲线
# 3. 保存和比较不同的实验配置
# 4. 记录模型参数和超参数
# 5. 保存和可视化模型预测结果
if is_wandb_available(): 
    import wandb


INVALID_LOGPROB = 1.0


# PolicyAndValueWrapper 是一个包装器类，用于将策略模型和价值模型组合在一起
# 这个类的主要目的是在 PPO 训练过程中同时处理策略和价值预测
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        # policy: 策略模型，用于生成动作（在语言模型中就是生成文本）
        self.policy = policy
        # value_model: 价值模型，用于预测状态的价值
        self.value_model = value_model
        # 获取价值模型的基础模型部分
        # 例如，如果使用 transformer 架构，这里获取的是 transformer 层
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        # 1. 首先通过价值模型的基础模型部分处理输入
        # 这会产生隐藏状态，包含了输入的特征表示
        output = self.critic_backbone(**kwargs)
        # 2. 使用价值模型的评分层预测价值
        # 使用最后一层的隐藏状态来计算价值
        logits = self.value_model.score(output.hidden_states[-1])
        # 3. 同时通过策略模型处理输入
        # 4. 返回策略模型的输出和价值预测
        return self.policy(**kwargs), logits


class PPOTrainer(Trainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        args: PPOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        model: nn.Module,
        ref_model: Optional[nn.Module],
        reward_model: nn.Module,
        train_dataset: Dataset,
        value_model: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        # PEFT 配置参数，用于设置参数高效微调的方法
        # 例如 LoRA (Low-Rank Adaptation) 等微调方法
        peft_config: Optional["PeftConfig"] = None,
    ) -> None:
        # 检查模型和参考模型是否相同
        # 如果使用 PEFT，则允许它们相同
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if args.stop_token and args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif args.stop_token:
            if args.stop_token == "eos":
                self.policy_model.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = args.stop_token_id  # None or int

        # PEFT 支持
        # 检查是否安装了 PEFT 库
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        # 如果安装了 PEFT 并且提供了配置
        elif is_peft_available() and peft_config is not None:
            # 如果模型已经是 PEFT 模型，先合并并卸载它
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()

            # 使用给定的配置创建新的 PEFT 模型
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            # 如果使用 bf16 精度且模型是 4bit 量化加载的，将 PEFT 模块转换为 bf16
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

        # 标记当前模型是否是 PEFT 模型
        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name
        # 设置参考模型（reference model）的逻辑：
        # 1. 如果显式提供了参考模型，则使用提供的模型
        if ref_model:
            self.ref_model = ref_model
        # 2. 如果当前模型是 PEFT 模型，则参考模型设为 None
        # 这是因为 PEFT 模型本身就可以通过切换适配器（adapter）来模拟参考模型的行为
        # 使用 ref_adapter_name 可以在需要时切换到参考模型的适配器
        elif self.is_peft_model:
            self.ref_model = None
        # 3. 如果既没有提供参考模型，也不是 PEFT 模型
        # 则创建一个新的参考模型作为策略模型的副本
        else:
            self.ref_model = create_reference_model(self.policy_model)
        self.reward_model = reward_model # 奖励模型，用于评估生成文本的质量
        self.train_dataset = train_dataset # 训练数据集，用于训练策略模型
        self.train_dataset_len = len(train_dataset) # 训练数据集的长度
        self.value_model = value_model # 价值模型，用于预测状态的价值
        self.data_collator = data_collator # 数据整理器，用于整理数据
        self.eval_dataset = eval_dataset # 评估数据集，用于评估模型
        self.optimizer, self.lr_scheduler = optimizers # 优化器和学习率调度器
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
            
        # Accelerator 是 Hugging Face 提供的一个工具，用于简化分布式训练
        # 它可以自动处理：
        # 1. 多 GPU 训练
        # 2. 混合精度训练（FP16/BF16）
        # 3. 梯度累积
        # 4. 分布式训练
        # 5. DeepSpeed 集成
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        # 获取分布式训练中的进程数量（GPU数量）
        args.world_size = accelerator.num_processes
        # 计算各种批次大小
        # local_batch_size: 每个进程（GPU）的批次大小
        # 计算公式：local_batch_size = per_device_batch_size * gradient_accumulation_steps * num_mini_batches
        # - per_device_train_batch_size: 每个设备（GPU）每次处理的样本数
        # - gradient_accumulation_steps: 梯度累积步数，用于模拟更大的批次大小
        # - num_mini_batches: 每个批次被分成的小批次数量
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        # micro_batch_size: 所有进程（GPU）的总批次大小
        # 计算公式：micro_batch_size = per_device_batch_size * world_size
        # micro_batch_size: 所有进程的总批次大小
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        # batch_size: 考虑梯度累积后的总批次大小
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        # whiten_rewards 是奖励白化（Reward Whitening）的开关
        # 奖励白化是一种技术，用于标准化奖励值，使其均值为0，标准差为1
        # 这样做的好处是：
        # 1. 使训练更稳定
        # 2. 减少奖励尺度的影响
        # 3. 提高模型对不同奖励范围的适应能力
        if args.whiten_rewards:
            # 为了进行有效的奖励白化，每个进程的小批次大小必须至少为8
            # 这是因为白化需要足够的样本来计算均值和标准差
            assert args.local_mini_batch_size >= 8, (
                f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
            )
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        # time_tensor 是一个张量，用于存储当前时间
        # 这个时间戳用于生成唯一的运行名称
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        # broadcast 是分布式训练中的一个重要操作
        # 它的作用是将一个进程（通常是主进程）的数据广播到所有其他进程
        # 参数说明：
        # - time_tensor: 要广播的数据
        # - 0: 源进程的 rank（通常是主进程）
        # 这里使用 broadcast 的目的是确保所有进程使用相同的时间戳
        # 这样可以避免不同进程生成不同的时间戳，导致运行名称不一致
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        # 生成一个唯一的运行名称，用于标识当前训练实验
        # 包含实验名称、随机种子和时间戳
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        # 如果需要生成样本，则设置样本生成频率
        # 样本生成频率 = 总批次数 / 样本生成次数
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
            if module is not None:
                disable_dropout_in_model(module) # 禁用模型中的 dropout 层
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        # 创建回调处理器
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        # 添加回调
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        # 创建训练控制对象，用于控制训练过程
        self.control = TrainerControl()
        # 创建在线训练状态对象
        # OnlineTrainerState 用于跟踪训练过程中的各种状态
        # 参数说明：
        # - is_local_process_zero: 是否是本地主进程
        # - is_world_process_zero: 是否是全局主进程
        # - stateful_callbacks: 需要保持状态的回调函数列表
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        # 当前训练的浮点操作数
        self.current_flos = 0
        # 超参数搜索后端
        self.hp_search_backend = None
        # DeepSpeed 是一个深度学习优化库，提供了多种优化技术：
        # 1. 混合精度训练（FP16/BF16）
        # 2. 梯度累积
        # 3. 模型并行
        # 4. 优化器状态分片
        # 5. 激活值检查点
        # 6. 通信优化
        # 这里检查是否启用了 DeepSpeed
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        # FSDP (Fully Sharded Data Parallel) 是 PyTorch 的分布式训练优化器
        # 它通过分片模型参数、梯度和优化器状态来减少内存使用
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        # hub_model_id 是 Hugging Face Hub 上的模型仓库 ID
        # 格式通常为：username/model-name
        # 例如：'huggingface/bert-base-uncased'
        # 这个 ID 用于：
        # 1. 推送模型到 Hugging Face Hub
        # 2. 从 Hub 下载模型
        # 3. 在 Hub 上管理模型版本
        self.hub_model_id = None
        # 如果设置了 push_to_hub 参数，初始化 Hugging Face Hub 仓库
        if self.args.push_to_hub:
            self.init_hf_repo()
        #如果需要保存模型，则创建输出目录
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

        # 设置随机种子
        # 在分布式训练中，每个进程都需要设置相同的随机种子
        # 这样可以确保：
        # 1. 所有进程使用相同的随机数序列
        # 2. 模型参数初始化一致
        # 3. 数据打乱方式一致
        # 4. dropout 等随机操作一致
        # 这对于保证分布式训练的一致性和可重复性非常重要
        if self.args.seed is not None:
            set_seed(self.args.seed, device_specific=True)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        # 使用 with 语句创建一个上下文管理器
        # 这个上下文管理器用于处理 PEFT 适配器的临时切换
        with (
            # 如果模型是 PEFT 模型且没有设置参考适配器名称
            # 则临时禁用当前模型的适配器
            # 否则使用空上下文（不执行任何操作）
            self.accelerator.unwrap_model(self.model.policy).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            # 如果设置了参考适配器名称
            # 则将模型切换到参考适配器
            # 这通常用于在训练过程中临时使用参考模型
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.ref_adapter_name)
            # yield 暂停上下文管理器的执行
            # 允许执行 with 语句块中的代码
            yield
            # 在 with 语句块结束后
            # 如果之前设置了参考适配器
            # 则将模型切换回原始适配器
            # 确保模型状态被正确恢复
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.model_adapter_name or "default")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        # 定义统计数据的形状
        # 形状为 (PPO训练轮数, 小批次数量, 梯度累积步数)
        # 这个形状用于跟踪训练过程中的各种指标
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        # 初始化各种统计指标的张量，用于记录训练过程中的数据
        # 所有张量都初始化为0，并放在指定设备（GPU/CPU）上
        approxkl_stats = torch.zeros(stats_shape, device=device)  # 近似KL散度统计
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)  # 策略梯度裁剪比例统计
        pg_loss_stats = torch.zeros(stats_shape, device=device)  # 策略梯度损失统计
        vf_loss_stats = torch.zeros(stats_shape, device=device)  # 价值函数损失统计
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)  # 价值函数裁剪比例统计
        entropy_stats = torch.zeros(stats_shape, device=device)  # 熵统计
        ratio_stats = torch.zeros(stats_shape, device=device)  # 比率统计
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                # 如果 logging_steps < 1，则将其视为比例
                # 例如：0.1 表示每完成 10% 的训练步数记录一次
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                # 如果 eval_steps < 1，则将其视为比例
                # 例如：0.1 表示每完成 10% 的训练步数评估一次
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                # 如果 save_steps < 1，则将其视为比例
                # 例如：0.1 表示每完成 10% 的训练步数保存一次
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        # 调用回调函数，开始训练
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            # 在 DeepSpeed 模式下，将模型赋值给 deepspeed 属性
            # 这是为了保持与 DeepSpeed 训练器的兼容性
            self.deepspeed = self.model
            # model_wrapped 是 Hugging Face Trainer 中的一个属性
            # 它用于包装原始模型，添加额外的功能（如梯度累积、混合精度等）
            # 在 DeepSpeed 模式下，我们直接使用原始模型作为包装后的模型
            # 这是因为 DeepSpeed 已经处理了这些额外的功能
            self.model_wrapped = self.model

        # 开始训练
        for update in range(1, args.num_total_batches + 1):
            # 更新当前训练轮数
            self.state.episode += 1 * args.batch_size
            # 获取下一个数据批次
            data = next(iter_dataloader)
            # 在梯度计算之前，不计算梯度
            with torch.no_grad():
                # 获取输入数据
                queries = data["input_ids"].to(device)
                # 获取上下文长度
                context_length = queries.shape[1]
                # 初始化列表，用于存储生成结果
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []
                # 使用 unwrap_model_for_generation 上下文管理器来解包模型
                # 这个上下文管理器的主要作用是：
                # 1. 在分布式训练环境中正确解包模型，使其能够进行文本生成
                # 2. 特别处理 DeepSpeed ZeRO Stage 3 的情况，确保模型参数正确可用
                # 3. 通过 gather_deepspeed3_params 参数控制是否收集 DeepSpeed 参数
                #    如果为 True，会收集所有参数，确保生成质量，但可能消耗更多内存
                #    如果为 False，跳过参数收集，节省内存但可能影响生成速度
                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    # 使用解包后的模型进行批量生成
                    # unwrapped_model.policy 是解包后的策略模型
                    # queries 是输入数据
                    # args.local_rollout_forward_batch_size 是每个批次的大小
                    # processing_class.pad_token_id 是填充标记的 ID
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                # 将整个批次的查询数据分成多个小批次进行处理
                # queries.shape[0] 是总查询数量
                # args.local_rollout_forward_batch_size 是每个小批次的大小
                # 这个循环的作用是：
                # 1. 按小批次处理数据，避免一次性处理太多数据导致内存不足
                # 2. 步长为 args.local_rollout_forward_batch_size，确保每个小批次大小一致
                # 3. 例如：如果有 100 个查询，batch_size 为 10，则循环会处理 10 个小批次
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    # 获取当前小批次的查询数据
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    # 获取对应的模型响应
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    # 提取响应部分（去掉输入部分）
                    response = query_response[:, context_length:]
                    # 获取对应的 logits
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    torch.cuda.empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_output = forward(model.policy, query_response, processing_class.pad_token_id)
                    else:
                        ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
                    )
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)
                    _, score, _ = get_reward(
                        reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_responses == processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                # 计算 KL 散度（Kullback-Leibler divergence）
                # KL 散度用于衡量当前策略与参考策略之间的差异
                kl = logprobs - ref_logprobs
                # 计算非奖励分数（non-score reward）
                # 这部分奖励来自 KL 散度，用于控制策略更新的大小
                # -args.kl_coef 是一个系数，用于调节 KL 散度的影响
                non_score_reward = -args.kl_coef * kl
                # 初始化总奖励
                # 首先复制非奖励分数作为基础
                rewards = non_score_reward.clone()
                # 获取每个序列的起始位置（0到batch_size-1）
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                # 获取每个序列的结束位置
                # 如果序列长度小于奖励长度，使用序列长度
                # 否则使用序列长度
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                # 在序列的起始和结束位置添加奖励分数
                # 这确保了奖励只在有效的序列范围内生效
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                            output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )
                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            # 如果配置了使用 wandb，将生成的样本保存到 wandb 中
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    # 将生成的样本数据保存为 wandb 表格
                    # 这样可以方便地在 wandb 界面上查看和分析生成的样本
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @article{mziegler2019fine-tuning,
            title        = {{Fine-Tuning Language Models from Human Preferences}},
            author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
            year         = 2019,
            eprint       = {arXiv:1909.08593}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="PPO",
            trainer_citation=citation,
            paper_title="Fine-Tuning Language Models from Human Preferences",
            paper_id="1909.08593",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
