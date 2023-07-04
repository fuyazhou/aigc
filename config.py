# openai key
openai_api_key = "hVaNL4EHBl5PbUoc3fuZT3BlbkFJiRyIxlPqVVSdRRI96dVt"
# webpilot key
webpilot_key = 'BBearer b2eea1d2008249ad950e8254c962d10e'
# google search key ,配置地址：https://serpapi.com/
google_search_api_key = "83bd9d6ce07a161441f4cdbc234456c79ebdf677a8d706d4b5dfcb0e962c2b05"
# 数据（数据库保存的位置）
database_path = "data/data.csv"
article_path = "data/article.csv"
article_txt_path = "data/article.txt"
faiss_index_path = "data/faiss_index"

# summary prompt : 根据search文章生成summary
summary_prompt = "Generate an abstract based on the following keywords and key sentences, the logic should be clear, about 200 words, Let's work this out in a step by step way to be sure we have the right answer."

summary_prompt_2 = "You are a helpful assistant that Write an abstract of about 400 words based on the following summary, article should be fluent, storytelling, and respectful of facts."

# generate prompt : 根据summary生成文章的prompt
generate_prompt = "I hope you are a skilled writer with many years of writing experience and write a high-quality essay based on the summary below. The essay should be fluent, respectful of facts, and highly readable, with a word count between 1500 and 2500. Let's work this out in a step by step way to be sure we have the right answer."

generate_prompt_2 = "Polish the article below to make it grammatically correct and logically correct, and output the polished article directly."
