import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.1,
)

response = llm.stream("你是什么模型，你能做什么")

for chunk in response:
    print(chunk.content, end="", flush=True)

'''
| 组件         | 作用                                                |
| ------------ | --------------------------------------------------- |
| `ChatOpenAI` | LangChain 中用于调用 OpenAI-compatible 聊天模型的类 |
| `base_url`   | 指向 DeepSeek 的 API 地址                           |
| `api_key`    | DeepSeek API Key                                    |
| `model`      | DeepSeek 模型名称                                   |
| `invoke()`   | 发起一次模型调用                                    |

'''
