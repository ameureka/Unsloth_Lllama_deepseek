# **Train the model**

Now set up GRPO Trainer and all configurations!

这段代码的主要目的是 **配置 GRPO 训练器，为后续的模型训练过程设定各种超参数和选项**。 它通过 **实例化 `trl.GRPOConfig` 类，并设置其内部的各种参数**，来完成配置。  配置的参数涵盖了：

1. **硬件加速**: 启用 vLLM 快速推理 ( `use_vllm=True` )，尝试使用 `bfloat16` 或 `fp16` 混合精度训练 ( `bf16`, `fp16` )，使用 8-bit Paged AdamW 优化器 ( `optim="paged_adamw_8bit"` )。
2. **优化器设置**: 设置学习率 ( `learning_rate` )，AdamW 优化器的 beta1 和 beta2 参数 ( `adam_beta1`, `adam_beta2` )，权重衰减 ( `weight_decay` )。
3. **学习率调度**: 使用余弦退火学习率调度器 ( `lr_scheduler_type="cosine"` )，并设置预热比例 ( `warmup_ratio` )。
4. **训练过程控制**: 设置日志记录步长 ( `logging_steps` )，每个设备的训练批大小 ( `per_device_train_batch_size` )，梯度累积步数 ( `gradient_accumulation_steps` )，每个 prompt 生成的回复数量 ( `num_generations` )，最大 prompt 长度 ( `max_prompt_length` )，最大回复长度 ( `max_completion_length` )，最大训练步数 ( `max_steps` )，模型保存步长 ( `save_steps` )，梯度裁剪最大范数 ( `max_grad_norm` )。
5. **输出和报告**: 设置输出目录 ( `output_dir` )，禁用训练报告工具 ( `report_to="none"` )。

```
from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)
```

- **`GRPOConfig`**: 是 `trl` 库提供的 **GRPO 训练配置类**。 它用于 **封装所有训练过程所需的超参数和配置信息**，例如学习率、批大小、优化器设置、硬件加速选项等等。 通过 `GRPOConfig`，可以方便地组织和管理训练参数。
- **`GRPOTrainer`**: 是 `trl` 库提供的 **GRPO 训练器类**。 它负责 **执行实际的 GRPO 训练循环**，包括数据加载、模型前向传播、损失计算、梯度更新、日志记录等步骤。 `GRPOTrainer` 使用 `GRPOConfig` 中定义的配置来指导训练过程。

【重点核心，这块是的模型训练的配置的基础】，但是你可以看到，在进行的训练之前的实际上，我们已经对部分的模型训练的参数已经做了设置，配置之后的模型，在基础上进行训练。

**`use_vllm = True`**:  **启用 vLLM (very Large Language Model) 快速推理**。  当设置为 `True` 时，`GRPOTrainer` 可能会利用 `vLLm` 库来加速模型的推理过程，特别是在生成回复或计算奖励时。  `vLLm` 是一个高性能的 LLM 推理和服务库，可以显著提高推理速度和效率。

**`learning_rate = 5e-6`**:  **设置学习率 (learning rate)**。  学习率是 **控制模型在训练过程中参数更新步幅大小的关键超参数**。  较小的学习率可能导致训练过程更稳定但更缓慢，较大的学习率可能导致训练过程更快但不稳定，甚至可能发散。  `5e-6` 是一个相对较小的学习率，通常适用于微调大型语言模型，可以避免在微调过程中过度修改预训练模型的参数。

**`adam_beta1 = 0.9,`**:  设置 `adam_beta1` 参数为 `0.9`。

- **`adam_beta1 = 0.9`**: **设置 AdamW 优化器的 beta1 参数**。 AdamW 是一种常用的优化器，特别是在 Transformer 模型训练中。 `beta1` 是 AdamW 优化器中用于计算 **一阶矩估计的指数衰减率**。 `0.9` 是 `beta1` 的一个常用默认值。

**`adam_beta2 = 0.99,`**:  设置 `adam_beta2` 参数为 `0.99`。

- **`adam_beta2 = 0.99`**: **设置 AdamW 优化器的 beta2 参数**。 `beta2` 是 AdamW 优化器中用于计算 **二阶矩估计的指数衰减率**。 `0.99` 或 `0.999` 是 `beta2` 的常用默认值。

**`weight_decay = 0.1,`**:  设置 `weight_decay` 参数为 `0.1`。

- **`weight_decay = 0.1`**: **设置权重衰减 (weight decay)**。 权重衰减是一种 **正则化技术**，通过在损失函数中添加模型权重的 L2 范数惩罚项，来 **防止模型过拟合**，提高模型的泛化能力。 `0.1` 是一个常用的权重衰减值。

**`warmup_ratio = 0.1`**:  **设置学习率预热比例 (warmup ratio)**。  学习率预热是一种 **学习率调度策略**，在训练的 **初始阶段，逐渐增加学习率**，从一个较小的值 (甚至接近 0) 线性增加到预设的学习率 (这里是 `learning_rate = 5e-6`)。  预热阶段结束后，再按照预设的学习率调度策略 (例如余弦退火) 调整学习率。  `warmup_ratio = 0.1`  表示 **训练总步数的 10% 用于学习率预热**。  学习率预热通常有助于 **提高训练的稳定性和收敛速度**，特别是在训练 Transformer 模型时。

**`lr_scheduler_type = "cosine"`**:  **设置学习率调度器类型 (learning rate scheduler type) 为 "cosine" (余弦退火)**。  学习率调度器 **控制学习率在训练过程中如何变化**。  "cosine" 调度器是一种常见的学习率退火策略，它使学习率 **从初始值开始，按照余弦函数曲线逐渐衰减到接近 0 的值**。  余弦退火通常被认为是一种 **有效且鲁棒的学习率调度策略**，能够帮助模型更好地收敛。

【模型的鲁棒性是什么。。。。。】

- **`optim = "paged_adamw_8bit",`**: 设置 `optim` 参数为 `"paged_adamw_8bit"`。
    - **`optim = "paged_adamw_8bit"`**: **设置优化器 (optimizer) 为 "paged_adamw_8bit" (Paged AdamW 8-bit)**。 这指定使用 **8-bit 版本的 Paged AdamW 优化器**。 Paged AdamW 是一种 **内存优化的 AdamW 优化器变体**，特别适用于 **训练大型模型**。 使用 8-bit 优化器可以 **减少优化器状态 (例如矩估计量) 占用的内存**，从而 **降低 GPU 显存需求**，使得在有限的 GPU 资源下可以训练更大的模型。 `trl` 库可能集成了 `bitsandbytes` 等库来实现 8-bit 优化器。
- **`logging_steps = 1,`**: 设置 `logging_steps` 参数为 `1`。
    - **`logging_steps = 1`**: **设置日志记录步长 (logging steps) 为 1**。 这表示 **每训练 1 个 step (梯度更新一次)，就记录一次训练日志**。 日志通常包含损失值、学习率等训练指标，用于 **监控训练过程和分析训练效果**。 `logging_steps = 1` 表示 **频繁地记录日志**，可以提供更细粒度的训练监控信息。
- **`bf16 = is_bfloat16_supported(),`**: 设置 `bf16` 参数为 `is_bfloat16_supported()` 函数的返回值。
    - **`bf16 = is_bfloat16_supported()`**: **启用 `bfloat16` (Brain Floating Point 16) 混合精度训练**。 `is_bfloat16_supported()` 函数 (之前导入的) **检测当前硬件是否支持 `bfloat16` 数据类型**。 如果支持 (例如在某些较新的 NVIDIA GPU 上)，则 `is_bfloat16_supported()` 返回 `True`，`bf16` 被设置为 `True`，**启用 `bfloat16` 训练**。 `bfloat16` 是一种半精度浮点格式，相比 `float32` 可以 **减少内存占用并加速计算**，尤其是在支持它的硬件上。
- **`fp16 = not is_bfloat16_supported(),`**: 设置 `fp16` 参数为 `not is_bfloat16_supported()` 函数的返回值。
    - **`fp16 = not is_bfloat16_supported()`**: **启用 `fp16` (Half Precision Floating Point) 混合精度训练**。 `not is_bfloat16_supported()` **取 `is_bfloat16_supported()` 返回值的反**。 这意味着：
        - 如果 `bfloat16` **被支持**，则 `is_bfloat16_supported()` 返回 `True`，`not is_bfloat16_supported()` 返回 `False`，`fp16` 被设置为 `False` (**不启用 `fp16` 训练**)。
        - 如果 `bfloat16` **不被支持**，则 `is_bfloat16_supported()` 返回 `False`，`not is_bfloat16_supported()` 返回 `True`，`fp16` 被设置为 `True` (**启用 `fp16` 训练**)。
    - **这段逻辑表示，优先使用 `bfloat16` 混合精度训练 (如果硬件支持)，否则退而求其次使用 `fp16` 混合精度训练 (如果硬件不支持 `bfloat16` 但可能支持 `fp16`)。 如果硬件既不支持 `bfloat16` 也不支持 `fp16`，则 `bf16` 和 `fp16` 都将为 `False`，将使用默认的 `float32` 精度训练。** 混合精度训练 (包括 `bfloat16` 和 `fp16`) 是一种 **加速训练和减少内存占用的常用技术**。
- **`per_device_train_batch_size = 1,`**: 设置 `per_device_train_batch_size` 参数为 `1`。
    - **`per_device_train_batch_size = 1`**: **设置每个设备 (例如每个 GPU) 的训练批大小 (per-device train batch size) 为 1**。 批大小 **决定了每次梯度更新时，模型处理的样本数量**。 `per_device_train_batch_size = 1` 表示 **使用较小的批大小**，每次只处理一个样本。 较小的批大小可以 **减少 GPU 内存需求**，使得在显存有限的情况下也能训练模型，但可能会 **降低训练的并行性和吞吐量**。 在 GRPO 这样的强化学习算法中，可能需要探索更多的模型输出 (generation)，因此可能需要控制单次训练的 batch size 以适应内存限制。
- **`gradient_accumulation_steps = 1, # 增加到4以获得更平滑的训练`**: 设置 `gradient_accumulation_steps` 参数为 `1`。
    - **`gradient_accumulation_steps = 1`**: **设置梯度累积步数 (gradient accumulation steps) 为 1**。 梯度累积是一种 **用小批大小模拟大批大小的技巧**。 当 `gradient_accumulation_steps > 1` 时，模型会 **累积多个小批次的梯度，然后在累积达到 `gradient_accumulation_steps` 步之后，进行一次梯度更新**。 例如，如果 `per_device_train_batch_size = 1`， `gradient_accumulation_steps = 4`，则 **实际等效的批大小为 4** (每 4 个样本更新一次梯度)。 `gradient_accumulation_steps = 1` 表示 **不使用梯度累积，每个批次都立即更新梯度**。
    - **`# 增加到4以获得更平滑的训练`**: 注释提示， **可以增加 `gradient_accumulation_steps` 的值 (例如增加到 4) 以获得更平滑的训练**。 增加梯度累积步数可以 **平滑梯度，减小梯度更新的方差，从而可能提高训练的稳定性和泛化性能**。 但是，增加 `gradient_accumulation_steps` 也会 **增加训练时间** (因为需要更多的前向传播才能完成一次梯度更新)。 `gradient_accumulation_steps = 1` 是默认值，可以根据需要和资源情况进行调整。
- **`num_generations = 6, # 如果内存不足则减少`**: 设置 `num_generations` 参数为 `6`。
    - **`num_generations = 6`**: **设置每个 prompt 生成的回复数量 (number of generations) 为 6**。 在 GRPO 训练中，通常需要 **为每个 prompt 生成多个回复 (completions)**，然后根据奖励函数对这些回复进行评估，并使用评估结果来更新模型。 `num_generations = 6` 表示 **对于每个输入 prompt，模型会生成 6 个不同的回复**。
    - **`# 如果内存不足则减少`**: 注释提示，如果 **GPU 内存不足**，可以 **减少 `num_generations` 的值**。 增加 `num_generations` 会 **增加每次训练迭代中的计算量和内存需求** (因为需要生成和评估更多的回复)。 如果显存不足，可以适当减少 `num_generations`，例如降低到 `4` 或 `2`。
- **`max_prompt_length = 256,`**: 设置 `max_prompt_length` 参数为 `256`。
    - **`max_prompt_length = 256`**: **设置输入 prompt 的最大长度 (maximum prompt length) 为 256 个 token**。 **超过这个长度的 prompt 将会被截断**。 `max_prompt_length` 限制了模型能够处理的最长输入序列长度。 设置为 `256` 可能适用于 GSM8k 数据集，因为数学题的问题描述通常不会太长。 可以根据数据集的特点和模型的能力进行调整。
- **`max_completion_length = 200,`**: 设置 `max_completion_length` 参数为 `200`。
    - **`max_completion_length = 200`**: **设置模型生成回复的最大长度 (maximum completion length) 为 200 个 token**。 **模型生成的回复超过这个长度将会被截断**。 `max_completion_length` 限制了模型输出的最大序列长度。 设置为 `200` 可能对于数学题的推理过程和最终答案来说是足够长的。 可以根据任务需求和模型能力进行调整。
- **`# num_train_epochs = 1, # 设置为1以进行完整的训练`**: 注释掉 `num_train_epochs = 1`。
    - **`# num_train_epochs = 1`**: **`num_train_epochs` (训练 epoch 数量) 参数被注释掉了**。 `num_train_epochs` 用于 **设置训练数据集要迭代训练多少个 epoch (轮次)**。 如果 **取消注释并设置为 `1`，则表示对整个训练数据集完整地训练一轮**。
    - **`# 设置为1以进行完整的训练`**: 注释说明设置为 `1` 表示进行完整的训练 (一轮 epoch)。
- **`max_steps = 250,`**: 设置 `max_steps` 参数为 `250`。
    - **`max_steps = 250`**: **设置最大训练步数 (maximum training steps) 为 250**。 `max_steps` **限制了总的训练步数**。 与 `num_train_epochs` 不同，`max_steps` **直接控制训练的迭代次数**，而不是基于数据集的完整轮次。 通常，在微调大型模型时，更常使用 `max_steps` 来控制训练时长，因为完整地训练多个 epoch 可能非常耗时。 设置为 `250` 是一个相对较小的训练步数，可能用于快速实验或演示目的。 对于实际训练，可能需要更大的 `max_steps` 值。
- **`save_steps = 250,`**: 设置 `save_steps` 参数为 `250`。
    - **`save_steps = 250`**: **设置模型保存步长 (save steps) 为 250**。 这表示 **每训练 `save_steps` 步，就保存一次模型检查点 (checkpoint)**。 检查点包含了训练过程中的模型权重和其他训练状态，可以用于 **后续的模型加载、评估或继续训练**。 `save_steps = 250` 表示 **在训练结束时 (达到 `max_steps`) 保存一次模型检查点** (因为 `max_steps` 也设置为 `250`)。 在实际训练中，可能需要更频繁地保存检查点 (例如每隔几百或几千步保存一次)，以便在训练过程中进行监控和选择最佳模型。
- **`max_grad_norm = 0.1,`**: 设置 `max_grad_norm` 参数为 `0.1`。
    - **`max_grad_norm = 0.1`**: **设置梯度裁剪的最大范数 (maximum gradient norm) 为 0.1**。 梯度裁剪是一种 **梯度正则化技术**，用于 **防止梯度爆炸 (gradient explosion)**。 在训练过程中，如果梯度的范数超过 `max_grad_norm`，则会将梯度 **缩放到范数为 `max_grad_norm`**。 梯度裁剪可以 **提高训练的稳定性**，尤其是在训练大型模型或使用循环神经网络 (RNNs) 等模型时。 `0.1` 是一个常用的 `max_grad_norm` 值。
- **`report_to = "none", # 可以使用Weights & Biases`**: 设置 `report_to` 参数为 `"none"`。
    - **`report_to = "none"`**: **设置训练报告工具 (reporting integration) 为 "none"**。 `report_to` 参数用于 **指定要将训练日志和指标报告给哪些平台**。 设置为 `"none"` 表示 **不使用任何报告工具，只在本地记录日志**。
    - **`# 可以使用Weights & Biases`**: 注释提示，**可以使用 Weights & Biases (wandb)** 等训练监控和可视化平台。 Weights & Biases 是一个流行的 MLOps 平台，可以用于 **跟踪、可视化和管理机器学习实验**。 如果想要使用 Weights & Biases，可以将 `report_to` 设置为 `"wandb"`，并需要安装 `wandb` 库和配置 Weights & Biases 账户。
- **`output_dir = "outputs",`**: 设置 `output_dir` 参数为 `"outputs"`。
    - **`output_dir = "outputs"`**: **设置输出目录 (output directory) 为 `"outputs"`**。 `output_dir` 指定了 **模型检查点、训练日志、配置文件等训练输出文件要保存的目录**。 设置为 `"outputs"` 表示将在当前目录下创建一个名为 `outputs` 的文件夹，并将所有训练输出文件保存在该文件夹中。
- **`)`**: `GRPOConfig()` 构造函数调用结束。

【这个很关键的，在未来我们的实际的模型的训练的过程之中的，是需要修改相关的参数的】

```
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()
```

这段代码的 **核心作用是初始化并启动 GRPO 训练器，开始使用 GSM8k 数据集对 Llama 3.1 8B Instruct 模型进行强化学习微调，以使其更好地解决数学问题，并生成符合 XML-CoT 格式的回复**。 它将之前代码中准备的所有组件 (模型、tokenizer、数据集、奖励函数、训练配置) 整合到 `GRPOTrainer` 中，并通过 `trainer.train()` 命令 **启动训练循环**。  训练完成后，将会得到一个 **经过 GRPO 微调的 Llama 3.1 8B Instruct 模型**，希望它在数学问题解决和输出格式规范性方面都得到提升。 

我们尝试增家代码，在进行训练之前检查模型的“分词器”以及“数据集”

tokenizer 对象包含了模型的词汇表和文本处理规则。 我们可以打印一些 tokenizer 的关键属性，例如：

```
# 打印部分 tokenizer 信息
print("----- Tokenizer 信息 -----")
print(f"词汇表大小: {tokenizer.vocab_size}") # 词汇表大小
print(f"特殊 token 字典: {tokenizer.special_tokens_map}") # 特殊 token 字典
print(f"模型最大长度: {tokenizer.model_max_length}") # 模型最大长度

# 测试 tokenizer 的文本编码和解码
test_text = "这是一个测试句子。Let's tokenize this!"
encoded_text = tokenizer.encode(test_text) # 编码文本
decoded_text = tokenizer.decode(encoded_text) # 解码文本

print("\n文本编码测试:")
print(f"原始文本: {test_text}")
print(f"编码后的 token ID: {encoded_text}")
print(f"解码后的文本: {decoded_text}")
```

- `---- Tokenizer 信息 -----
词汇表大小: 128000
特殊 token 字典: {'bos_token': '<|begin_of_text|>', 'eos_token': '<|eot_id|>', 'pad_token': '<|finetune_right_pad_id|>'}
模型最大长度: 131072`

`文本编码测试:
原始文本: 这是一个测试句子。Let's tokenize this!
编码后的 token ID: [128000, 114880, 48044, 82805, 106808, 45829, 1811, 10267, 596, 78751, 420, 0]
解码后的文本: <|begin_of_text|>这是一个测试句子。Let's tokenize this!`

```
# 打印部分 dataset 内容
print("\n----- Dataset 样本 -----")
sample_indices = [0, 1, 2] # 想要打印的样本索引
for index in sample_indices:
    sample = dataset[index] # 获取指定索引的样本
    print(f"\n样本索引: {index}")
    print("Prompt:")
    for message in sample['prompt']: # 打印 prompt 消息列表
        print(f"  Role: {message['role']}, Content: {message['content']}")
    print(f"Answer: {sample['answer']}") # 打印 answer
```

dataset 对象存储了训练数据，我们可以打印 dataset 的一些样本，来查看数据的结构和内容。

- `---- Dataset 样本 -----`

`样本索引: 0
Prompt:
Role: system, Content:
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>`

`Role: user, Content: Natalia sold clips to 48 of her friends last month. This month, she sold half as many clips as last month. How many clips did Natalia sell this month?
Answer: 24`

训练的阶段结果如下：

- `------------------- Question:
***The gauge on a water tank shows that the tank is 1/3 full of water. To fill the tank, 16 gallons of water are added. How many gallons of water does the tank hold when full?
Answer:
24***`
- `Response:（`**模型生成的回复**:  这是模型对问题的回答。  模型尝试给出了 **逐步推理 (Chain of Thought - CoT) 的过程**。`）
Let's break this problem down.`

`If the tank is 1/3 full and 16 gallons are added to fill it, then the amount of water added is equal to the amount that was already in the tank (1/3 of the total capacity) plus the amount needed to fill it (2/3 of the total capacity).`

`Let x be the capacity of the tank in gallons.`

`The equation becomes x - (1/3)x + 16 = x.`

`To simplify the equation, we can combine the x terms: (2/3)x + 16 = x.`

`Next, subtract (2/3)x from both sides of the equation: (1/3)x = 16.`

`Now, multiply both sides of the equation by 3 to get rid of the fraction: x = 16 * 3`

`x = 48`

`The tank holds 48 gallons of water when full.`

`**Extracted:（`从模型回复中提取的答案**:  这部分显示的是 `extract_xml_answer` 函数从模型回复中提取出的答案。**`）**
Let's break this problem down.`

`If the tank is 1/3 full and 16 gallons are added to fill it, then the amount of water added is equal to the amount that was already in the tank (1/3 of the total capacity) plus the amount needed to fill it (2/3 of the total capacity).`

`Let x be the capacity of the tank in gallons.`

`The equation becomes x - (1/3)x + 16 = x.`

`To simplify the equation, we can combine the x terms: (2/3)x + 16 = x.`

`Next, subtract (2/3)x from both sides of the equation: (1/3)x = 16.`

`Now, multiply both sides of the equation by 3 to get rid of the fraction: x = 16 * 3`

`x = 48`

`The tank holds 48 gallons of water when full.`

**整体分析和结论:**

- **问题类型**: 这是一个需要 **数学计算和应用题理解** 的问题。
- **模型尝试 CoT 推理**: 模型成功尝试生成了 CoT 推理过程，表明模型正在学习逐步思考。
- **推理逻辑偏差**: 模型的推理逻辑存在偏差，导致方程构建错误，最终答案错误。 这表明模型 **在数学应用题的理解和逻辑推理方面仍有提升空间**。
- **答案错误**: 模型给出的答案 `48` 与正确答案 `24` 不符。
- **`correctness_reward_func` 奖励**: 由于提取出的答案 (整个回复) 与正确答案 (`24`) **不一致**， `correctness_reward_func` 将会 **给予零奖励 `0.0`** 给这个回复。
- **XML 格式奖励**: 由于模型回复中 **没有包含 XML 标签** (例如 `<reasoning>` 和 `<answer>`), `strict_format_reward_func` 和 `soft_format_reward_func` 都会 **给予零奖励 `0.0`**。 `xmlcount_reward_func` 的得分也会 **比较低** (接近 0 或为负值，取决于具体实现和惩罚机制)。
- **`int_reward_func` 奖励**: 由于提取出的答案 (整个回复) **不是纯数字**， `int_reward_func` 也会 **给予零奖励 `0.0`**。

另外的一个正确的
<img width="709" alt="Clipboard_Screenshot_1740497293" src="https://github.com/user-attachments/assets/cd6aad5f-2d91-4a70-94a1-f8167784d5dc" />



所以我们看到第一步进行的训练的过程之中的是错误的，所以的没有奖励函数。
<img width="711" alt="Clipboard_Screenshot_1740497306" src="https://github.com/user-attachments/assets/1901f563-a991-4830-8475-9480772b2195" />



测试进行了 20 步训练



