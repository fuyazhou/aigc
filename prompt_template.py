from langchain.prompts import PromptTemplate

# Build prompt
template1 = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

# Build prompt
template2 = """
基于以下已知信息，回答用户的问题，尽量按照已知内容的原文回答。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
已知内容:
{context}
问题:
{question}"""

template3 = """
已知内容: {context}
\n\n

根据上文请解释或者回答下面这个问题，要尽可能的详细，一步一步的回答问题:
问题:{question}

如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

"""

# openai 官方的prompt
template4 = """
SYSTEM Use the provided articles delimited by triple quotes to answer questions. If the answer cannot be found in the articles, write "I could not find an answer."

USER <insert articles, each delimited by triple quotes>

Question:
"""

template5 = '''
Use the provided articles delimited by triple quotes to answer questions,尽可能详细的回答问题. If the answer cannot be found in the articles, write "根据已知信息无法回答该问题。"，答案请使用中文。

"""{context}"""


Question:{question}

'''



template_contract = """
已知内容: 
{context}
\n\n

根据上面的内容模版，结合下面的问题查找所需要的合同模版，答案请使用中文:  
问题:{question}

"""