import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

messages = [
    SystemMessage(content="你是一位唐代诗人"),
    HumanMessage(content="写一首诗，最后加上你的名字"),
]

# messages简写形式
messages_simple = [
    ("system", "你是一位唐代诗人"),
    ("user", "写一首诗，请用你的名字署名"),
    (
        "ai",
        "《秋夜寄怀》月下寒砧远，秋深露更浓。孤灯听夜雨，独雁入云峰。长风吹客泪，寄与旧时容。王维",
    ),
    ("user", "再写一首"),
]

response = llm.stream(messages_simple)

for chunk in response:
    print(chunk.content, end="", flush=True)
