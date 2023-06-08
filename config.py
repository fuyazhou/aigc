# openai key
openai_api_key = "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"
# google search key ,配置地址：https://serpapi.com/
#google_search_api_key = "38383416a5433c9aa87aeaa9068d2c52be467b0677b86094bc3daab9905f49d9"
google_search_api_key = "83bd9d6ce07a161441f4cdbc234456c79ebdf677a8d706d4b5dfcb0e962c2b05"
# 数据（数据库保存的位置）
database_path = "data/data.csv"
article_path = "data/article.csv"
faiss_index_path = "data/faiss_index"

# summary prompt : 根据search文章生成summary
summary_prompt = "You are a helpful assistant that summarizes documents. let us step by step."
summary_prompt_2 = "You are a helpful assistant that summarizes documents. let us step by step."

# generate prompt : 根据summary生成文章的prompt
generate_prompt = "You are a helpful assistant that Write an article of about 1500 words based on the following " \
                  "summary, article should be fluent, storytelling, and respectful of facts."


generate_prompt_2 = "You are a helpful assistant that Write an article of about 1500 words based on the following " \
                  "summary, article should be fluent, storytelling, and respectful of facts."

