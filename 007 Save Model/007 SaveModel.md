# **Saving to float16 for VLLM**

We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to [https://huggingface.co/settings/tokens](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fsettings%2Ftokens) for your personal tokens.

这段代码块提供了**三种不同的方法来保存和分享经过微调的模型**，每种方法都提供了**本地保存**和**推送到 Hugging Face Hub** 的选项。  所有这些代码行都被 `if False:` 条件包裹，这意味着它们**默认情况下不会执行**。  你需要将 `if False` 修改为 `if True`，才能激活相应的保存或推送操作。

```
# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")
```

**`model.save_pretrained_merged(...)`**:  调用 `model` 对象的 `save_pretrained_merged` 方法。  这个函数的作用是 **将 LoRA 适配器的权重合并到基础模型权重中，并将合并后的完整模型保存到本地磁盘**。

**`"model"`**:  第一个参数，字符串 `"model"`，指定了 **本地保存模型的目录名称**。  执行后会在当前工作目录下创建一个名为 `model` 的文件夹，用于存放模型文件。

**`tokenizer`**:  第二个参数，`tokenizer` 对象，是之前加载的分词器。  `save_pretrained_merged` 会 **同时保存模型和 tokenizer**，确保模型可以被正确加载和使用。

- **"merged"**: 采用合并方式保存，即将 LoRA 权重合并回基础模型。 这样保存的模型是一个**独立的完整模型**，不再依赖 LoRA 适配器文件。
- **"16bit"**: 模型权重将 **转换为 16-bit 精度** 进行保存。 常见的 16-bit 精度格式是 `float16` 或 `bfloat16`。 使用 16-bit 精度可以 **显著减小模型文件大小，并可能提高推理速度** (在支持 16-bit 计算的硬件上)，但可能会牺牲少量精度。

**`"hf/model"`**:  第一个参数，字符串 `"hf/model"`，是 **Hugging Face Hub 上的仓库 (repository) 名称**。  您需要将其替换为您在 Hugging Face Hub 上创建的仓库名称，格式通常是 `"用户名/仓库名"`。  例如 `"your_username/my_llama_model"`。

**`token = ""`**:  关键字参数 `token`，用于 **提供 Hugging Face Hub 的访问令牌 (access token)**。  如果需要推送到私有仓库或组织仓库，或者首次推送模型到公共仓库，通常需要提供有效的访问令牌。  `token = ""` 是一个占位符，您需要替换为您的 Hugging Face Hub 访问令牌。  对于公共仓库，如果已经登录了 Hugging Face CLI，可能可以留空。

- **功能**: 将 LoRA 权重合并到基础模型，并以 16-bit 精度保存或推送到 Hugging Face Hub。
- **优点**:
    - 生成 **独立的完整模型**，无需额外的 LoRA 适配器文件。
    - **减小模型文件大小** (相比原始精度模型)。
    - **可能提高推理速度** (在支持 16-bit 计算的硬件上)。
    - 方便本地部署和分享。
- **缺点**:
    - 可能会 **损失少量精度** (相比更高精度的模型)。
    - 模型文件仍然相对较大 (虽然比原始精度模型小)。

**仅 LoRA 适配器" 选项:**

- **功能**: 仅保存 LoRA 适配器文件，并保存到本地或推送到 Hugging Face Hub。
- **优点**:
    - **模型文件非常小巧**，只包含 LoRA 权重。
    - **节省存储空间和分享带宽**。
    - **易于分享微调成果**，尤其是当基础模型已经公开可用时，只需分享 LoRA 适配器即可。
    - **保持基础模型的纯净性**，方便在不同 LoRA 适配器之间切换。
- **缺点**:
    - **不能独立运行**，需要依赖基础模型。
    - **使用时需要先加载基础模型，再加载并应用 LoRA 适配器**，步骤相对繁琐。

好了，那么我现在的就是开始尝试的进行保存文件

当然我们修改之后可以进行的模型的上传操作
<img width="710" alt="Clipboard_Screenshot_1740497487" src="https://github.com/user-attachments/assets/6f72a3ae-1a27-4a89-b67c-25c0bc745fbb" />
<img width="712" alt="Clipboard_Screenshot_1740497510" src="https://github.com/user-attachments/assets/640bbcf9-763e-4b30-a3e1-e0cfb089a074" />

