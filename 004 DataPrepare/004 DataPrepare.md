# **Data Prep**

特别是为了处理和加载用于训练或评估的**GSM8k 数学问题数据集**。代码中定义了如何处理数据集中的问题和答案，并将其转换为适合模型输入的格式。

【模型适合输入的数据格式的内容是什么，这个你之前的有没有了解的过，格式的意义是什么】

这块比较核心关键，就是未来我们对数据集的要求的以及把握处理是比较关键，有了数据集，就可以处理很多问题的。

```
import re
from datasets import load_dataset, Dataset

```

**`datasets`**:  这是一个由 Hugging Face 开发的 Python 库，用于 **方便地访问和操作各种机器学习数据集**，特别是自然语言处理 (NLP) 领域的数据集。 它提供了统一的 API 来加载、处理和管理数据集，无论数据集是本地文件还是托管在 Hugging Face Hub 上。

**`Dataset`**:  是 `datasets` 库中表示 **数据集** 的类。 `load_dataset` 函数返回的对象通常是 `Dataset` 类的实例，或者是一个 `DatasetDict` (包含多个 `Dataset` 的字典，例如 train/validation/test split)。 `Dataset` 对象提供了多种方法来操作和访问数据集中的数据。

```
# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""
```

**`SYSTEM_PROMPT`**:  这是一个 **系统提示 (system prompt)**，用于 **指导语言模型如何进行回复**。 在一些对话模型或指令遵循模型中，系统提示会在用户输入之前被提供给模型，以设定模型的行为或输出格式。

**`Respond in the following format: ... </answer>`**:  这段文本定义了期望模型 **回复的格式**。  它要求模型在回复中 **显式地包含 `<reasoning>` 和 `</reasoning>` 标签以及 `<answer>` 和 `</answer>` 标签**。 这表明期望模型输出包含**推理过程 (reasoning)** 和 **最终答案 (answer)** 的结构化回复，这是一种典型的 **Chain-of-Thought (CoT, 思维链)** 的格式。  CoT 是一种技术，旨在让模型在给出最终答案之前，先显式地展示其推理过程，以提高模型解决复杂问题的能力和可解释性。

【如何更好地理解cot 思维链这个是很关键的】

当实际使用时，可以 **将推理过程的文本填充到 `{reasoning}` 占位符，答案文本填充到 `{answer}` 占位符**，从而生成符合 XML-CoT 格式的字符串。  但在这段代码中，`XML_COT_FORMAT` 变量本身虽然被定义了，但似乎并没有被直接使用到后续的数据处理流程中。 它更像是一个 **格式定义的参考**。

```
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()
```

**`"""从XML格式的文本中提取答案"""`**:  函数的文档字符串 (docstring)，说明了这个函数的功能是 **从 XML 格式的文本中提取答案**。

**`extract_hash_answer` 函数的作用是从一段文本中，查找 "####" 标记。 如果找到 "####"，则提取 "####" 之后的内容作为答案，并去除首尾空白字符； 如果没有找到 "####"，则返回 `None`，表示无法提取答案。**  这个函数是 **针对特定格式 (答案用 "####" 分隔) 的数据集设计的**。

【所以这里数据集的格式已经表明出来了，后续自我设计数据集也可以通过这种模式进行】

```
# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()
```

**`'openai/gsm8k'`**:  这是 Hugging Face Hub 上 **GSM8k 数据集的标识符**。  `'openai'` 是数据集的组织或作者，`'gsm8k'` 是数据集的名称。  `load_dataset` 函数会从 Hugging Face Hub 下载并加载该数据集。 GSM8k 数据集是一个用于 **数学应用题 (grade school math 8K)** 的数据集，包含 8500 个高质量的数学问题，主要用于 **评估模型的数学推理能力**。

**`lambda x: { ... }`**:  使用 **lambda 函数定义了映射操作的具体逻辑**。  `lambda x:` 表示定义一个匿名函数，接收一个参数 `x`，`x` 代表数据集中的一个样本 (一个字典)。  `{ ... }`  定义了 **lambda 函数的返回值，它是一个新的字典，代表映射后的样本**。

注释说明 `data.map()` 的目的是 **映射数据集，以创建模型的提示 (prompt) 和答案 (answer)**。  经过 `data.map()` 操作后，数据集中的每个样本都会被转换为包含 `'prompt'` 和 `'answer'` 字段的新样本。  `'prompt'` 字段是一个消息列表，适合作为对话模型或指令遵循模型的输入，`'answer'` 字段包含了问题的最终答案。

**`return data`**:  函数 **返回经过映射处理后的数据集 `data`**。  这个数据集已经准备好了，可以用于后续的模型训练或评估。

【我们要看下的，适合模型的处理之后的数据集】

结果：

<img width="716" alt="Clipboard_Screenshot_1740497089" src="https://github.com/user-attachments/assets/e18e2dfe-47ac-4814-a4da-2cc6444f21e8" />


当然我们这里可以做一个测试，打印相关的数据集

```
print("数据集的前 2 个样本：")
for i in range(30): # 为了演示，只打印前 2 个
    sample = dataset[i]
    print(f"\n--- 样本 {i+1} ---")
    print("Prompt:")
    for message in sample['prompt']:
        print(f"  {message['role']}: {message['content']}")
    print(f"Answer: {sample['answer']}")

print(f"\n数据集总共有 {len(dataset)} 个样本。")
```

`数据集的前 2 个样本：`

- `-- 样本 1 ---
Prompt:
system:
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>`

`user: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: 72`

- `-- 样本 2 ---
Prompt:
system:
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>`

`user: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: 10`

我们在看在在 hunggingface 上的数据集的格式

https://huggingface.co/datasets/openai/gsm8k

<img width="707" alt="Clipboard_Screenshot_1740497112" src="https://github.com/user-attachments/assets/b13af9f5-a81d-4fb5-9d8a-e974aa367ca4" />


这段代码定义了一系列**奖励函数 (reward functions)**，这些函数旨在**评估模型生成的回复，并根据不同的标准给予不同的奖励分数**。 这些奖励函数很可能用于**强化学习 (Reinforcement Learning)** 训练过程中，指导模型学习生成期望类型的回复。让我们逐个分析这些奖励函数：

```
# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
```

**`correctness_reward_func` 函数的功能：**  它接收输入提示、模型生成的回复、以及正确答案。  它从模型回复中提取出 XML 格式的答案，然后将提取出的答案与正确答案进行 **严格的字符串比较**。  如果 **完全一致**，则给予 **高奖励 `2.0`**； 如果 **不一致**，则给予 **零奖励 `0.0`**。  同时，函数还会打印出问题、正确答案、模型回复和提取出的答案，用于调试和观察。  这个奖励函数主要 **关注答案的绝对正确性**，是一种 **基于精确匹配的奖励机制**。  奖励值 `2.0` 和 `0.0` 是人为设定的，可以根据具体需求调整奖励幅度。  这种奖励函数 **可能适用于需要精确答案的任务**，例如数学题、代码生成等。

```jsx
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
```

**`int_reward_func` 函数的功能：**  它接收模型回复，并提取出 XML 格式的答案。  然后，它 **检查提取出的答案是否为一个整数** (使用 `isdigit()` 方法)。  如果是 **整数**，则给予 **奖励 `0.5`**； 如果 **不是整数**，则给予 **零奖励 `0.0`**。  这个奖励函数 **关注回复的格式是否为整数**，而不关心答案的正确性。  奖励值 `0.5` 和 `0.0` 是人为设定的，表示 "是整数" 这种格式的奖励程度。  这种奖励函数 **可能用于需要模型输出整数答案的任务**，例如某些数学计算题，或者需要模型按照特定格式输出的任务。

**`strict_format_reward_func` 函数的功能：**  它接收模型回复，并使用 **严格的正则表达式模式** 检查回复是否 **完全符合预定义的 XML-CoT 格式 (包括换行符和字符串的开头结尾)**。  如果 **完全符合**，则给予 **奖励 `0.5`**； 如果 **不符合**，则给予 **零奖励 `0.0`**。  这个奖励函数 **强制模型输出非常规范的 XML-CoT 格式**。  奖励值 `0.5` 和 `0.0` 表明了对 "严格格式" 的奖励程度。 这种奖励函数 **适用于需要模型输出结构化、规范化、且格式严格符合要求的数据的任务**。

**`soft_format_reward_func` 函数的功能：**  它接收模型回复，并使用 **宽松的正则表达式模式** 检查回复是否 **包含预定义的 XML-CoT 格式 (标签对存在且顺序正确，标签间允许空白，标签内部可包含任意内容)**。  如果 **包含**，则给予 **奖励 `0.5`**； 如果 **不包含**，则给予 **零奖励 `0.0`**。  这个奖励函数 **鼓励模型输出 XML-CoT 格式，但对格式的规范性要求较低**，允许一定的灵活性。  奖励值 `0.5` 和 `0.0` 表明了对 "宽松格式" 的奖励程度。  这种奖励函数 **适用于希望模型输出结构化数据，但可以容忍一定格式偏差的任务**。

【可以看出，这种奖励函数的设置规则，不同的模型的方向的奖励的规则设置是不一样的】

```
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count
```

**`count_xml` 函数的功能：**  它接收一段文本，并 **通过字符串计数的方式，统计特定 XML 标签 ( `<reasoning>`, `</reasoning>`, `<answer>`, `</answer>` ) 的出现情况，并根据出现次数和一些规则 (例如 `</answer>` 标签后的内容长度惩罚) 计算一个浮点数得分**。  每个标签正确出现一次，可以获得 `0.125` 分奖励。  在 `</answer>` 标签之后输出过多内容会受到惩罚。  这个函数 **不是严格的 XML 格式验证或元素计数**，而是一种 **基于字符串匹配的、近似的 XML 格式相似度评分**。  总分数的上限是 `4 * 0.125 = 0.5`，再加上惩罚项，最终得分可能会低于 0 或超过 0.5 (虽然可能性较低，因为惩罚系数很小)。  这个函数 **可以用于粗略地评估模型回复是否包含了期望的 XML 结构**，并可以作为一种奖励信号。

```
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
```

**`xmlcount_reward_func` 函数的功能：**  它接收模型回复，并 **对每个回复调用 `count_xml` 函数，计算其 XML 格式相似度得分**。  然后，它 **直接将 `count_xml` 函数返回的得分作为奖励值**。  因此，`xmlcount_reward_func` 的奖励机制 **完全依赖于 `count_xml` 函数的评分逻辑**。  它 **鼓励模型输出包含更多（或更完整）的 XML 结构元素**，例如 `<reasoning>`, `</reasoning>`, `<answer>`, `</answer>` 标签，并可能 **对在 `</answer>` 标签后输出过多内容进行惩罚**。  奖励值范围取决于 `count_xml` 函数的得分范围，但从 `count_xml` 的代码来看，最大可能得分约为 `0.5` (未考虑惩罚的情况)。  这种奖励函数 **可以作为一种引导模型输出具有 XML 结构化格式的回复的辅助奖励信号**，例如，可以与 `correctness_reward_func` 结合使用，在保证答案正确性的同时，也鼓励模型以 XML-CoT 格式进行输出。

【这种奖励函数可能就是通用的奖励函数的】

这些奖励函数可以 **根据具体的训练目标和任务需求进行选择和组合**。  例如，如果主要目标是答案的正确性，可以主要使用 `correctness_reward_func`； 如果也希望模型输出 XML-CoT 格式的回复，可以将 `correctness_reward_func` 与 `xmlcount_reward_func` 或 `soft_format_reward_func` 结合使用，并调整不同奖励函数的权重，以平衡各个奖励目标。
