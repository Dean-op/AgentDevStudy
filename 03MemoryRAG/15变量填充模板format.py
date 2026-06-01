from langchain_core.prompts import PromptTemplate

# `format()` 的作用很直接：把变量填进模板，返回一个字符串。
prompt_template = PromptTemplate.from_template(
    "请把下面的用户问题分类：{question}"
)

prompt = prompt_template.format(
    question="我申请退款三天了，钱还没到账。"
)

print(prompt)