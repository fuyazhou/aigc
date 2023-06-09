# 项目说明 

该项目提供生成特定主题相关的文章。
```
1、首先使用了 Google 搜索的相关问题
2、summary搜索结果
3、然后使用 FAISS 对其进行相似度搜索
4、最后使用 OpenAI chatGPT 生成文章。
```

流程图：
![img.png](img.png)


## 配置
在 config.py 文件中配置相关参数，如 Google 搜索 API 密钥(https://serpapi.com/)、OpenAI API 密钥、数据库路径等


## 使用
Python的版本3.9   
使用前需要安装相应的依赖库，可以通过以下命令安装：
```
pip install -r requirements.txt
```

运行 main.py 文件。

生成数据和生成文章的示例：
```
python3 main.py --generate_data_query "amazon bussiness model"
python3 main.py --generate_article_query " Amazon was one of the first companies to offer same-day delivery, and it has since expanded its delivery options to include two-hour delivery, in-car delivery,"
```
如果查询特征网站的 在query之后加上:  site:website


demo:
```
python3 main.py --generate_data_query "Tesla bussiness model. site:twitter.com "
```


http服务
```
启动服务： python server.py



请求入参示例：

POST /generate HTTP/1.1
Content-Type: application/json
{
    "generate_data_query": "你的query",
    "generate_article_query": "你的query"
}
```



