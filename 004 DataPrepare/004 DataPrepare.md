
Data Prep
特别是为了处理和加载用于训练或评估的GSM8k 数学问题数据集。代码中定义了如何处理数据集中的问题和答案，并将其转换为适合模型输入的格式。
【模型适合输入的数据格式的内容是什么，这个你之前的有没有了解的过，格式的意义是什么】
这块比较核心关键，就是未来我们对数据集的要求的以及把握处理是比较关键，有了数据集，就可以处理很多问题的。
import re
from datasets import load_dataset, Dataset

​
datasets:  这是一个由 Hugging Face 开发的 Python 库，用于 方便地访问和操作各种机器学习数据集，特别是自然语言处理 (NLP) 领域的数据集。 它提供了统一的 API 来加载、处理和管理数据集，无论数据集是本地文件还是托管在 Hugging Face Hub 上。
Dataset:  是 datasets 库中表示 数据集 的类。 load_dataset 函数返回的对象通常是 Dataset 类的实例，或者是一个 DatasetDict (包含多个 Dataset 的字典，例如 train/validation/test split)。 Dataset 对象提供了多种方法来操作和访问数据集中的数据。
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
​
SYSTEM_PROMPT:  这是一个 系统提示 (system prompt)，用于 指导语言模型如何进行回复。 在一些对话模型或指令遵循模型中，系统提示会在用户输入之前被提供给模型，以设定模型的行为或输出格式。
Respond in the following format: ... </answer>:  这段文本定义了期望模型 回复的格式。  它要求模型在回复中 显式地包含 <reasoning> 和 </reasoning> 标签以及 <answer> 和 </answer> 标签。 这表明期望模型输出包含推理过程 (reasoning) 和 最终答案 (answer) 的结构化回复，这是一种典型的 Chain-of-Thought (CoT, 思维链) 的格式。  CoT 是一种技术，旨在让模型在给出最终答案之前，先显式地展示其推理过程，以提高模型解决复杂问题的能力和可解释性。
【如何更好地理解cot 思维链这个是很关键的】
当实际使用时，可以 将推理过程的文本填充到 {reasoning} 占位符，答案文本填充到 {answer} 占位符，从而生成符合 XML-CoT 格式的字符串。  但在这段代码中，XML_COT_FORMAT 变量本身虽然被定义了，但似乎并没有被直接使用到后续的数据处理流程中。 它更像是一个 格式定义的参考。
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()
​
"""从XML格式的文本中提取答案""":  函数的文档字符串 (docstring)，说明了这个函数的功能是 从 XML 格式的文本中提取答案。
extract_hash_answer 函数的作用是从一段文本中，查找 "####" 标记。 如果找到 "####"，则提取 "####" 之后的内容作为答案，并去除首尾空白字符； 如果没有找到 "####"，则返回 None，表示无法提取答案。  这个函数是 针对特定格式 (答案用 "####" 分隔) 的数据集设计的。
【所以这里数据集的格式已经表明出来了，后续自我设计数据集也可以通过这种模式进行】
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
​
'openai/gsm8k':  这是 Hugging Face Hub 上 GSM8k 数据集的标识符。  'openai' 是数据集的组织或作者，'gsm8k' 是数据集的名称。  load_dataset 函数会从 Hugging Face Hub 下载并加载该数据集。 GSM8k 数据集是一个用于 数学应用题 (grade school math 8K) 的数据集，包含 8500 个高质量的数学问题，主要用于 评估模型的数学推理能力。
lambda x: { ... }:  使用 lambda 函数定义了映射操作的具体逻辑。  lambda x: 表示定义一个匿名函数，接收一个参数 x，x 代表数据集中的一个样本 (一个字典)。  { ... }  定义了 lambda 函数的返回值，它是一个新的字典，代表映射后的样本。
注释说明 data.map() 的目的是 映射数据集，以创建模型的提示 (prompt) 和答案 (answer)。  经过 data.map() 操作后，数据集中的每个样本都会被转换为包含 'prompt' 和 'answer' 字段的新样本。  'prompt' 字段是一个消息列表，适合作为对话模型或指令遵循模型的输入，'answer' 字段包含了问题的最终答案。
return data:  函数 返回经过映射处理后的数据集 data。  这个数据集已经准备好了，可以用于后续的模型训练或评估。
【我们要看下的，适合模型的处理之后的数据集】
结果：

当然我们这里可以做一个测试，打印相关的数据集
print("数据集的前 2 个样本：")
for i in range(30): # 为了演示，只打印前 2 个
    sample = dataset[i]
    print(f"\n--- 样本 {i+1} ---")
    print("Prompt:")
    for message in sample['prompt']:
        print(f"  {message['role']}: {message['content']}")
    print(f"Answer: {sample['answer']}")


print(f"\n数据集总共有 {len(dataset)} 个样本。")
​
数据集的前 2 个样本：
-- 样本 1 ---
Prompt:
system:
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
user: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: 72
-- 样本 2 ---
Prompt:
system:
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
user: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: 10
我们在看在在 hunggingface 上的数据集的格式
https://huggingface.co/datasets/openai/gsm8k


