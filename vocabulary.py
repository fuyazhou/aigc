from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import threading
import time
import pandas as pd
import mysql.connector
import datetime

os.environ["OPENAI_API_KEY"] = "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"

chat = ChatOpenAI(temperature=0.0)


vocabulary_template = """
For the following text, extract the following information:

word: 热点并且有影响力的词汇，比如['五一','航天日', '华为', '华为笔记本,超材料天线技术', '基辅', '极氪发布会', '间谍活动', '中国国家安全', '间谍', '进口', '榴莲', '抗原', '二阳', '新冠二次感染', '克里姆林宫', '裸眼3D', '马斯克', '马斯克,特斯拉', '马斯克,星舰', '买房', '美国国债', '美国太空探索技术公司,星舰', '欧冠', '苹果', '苹果公司', '切尔西,布莱顿', '人口自然增长率', '上海车展', '上海车展，直播', '社交媒体，二阳，疫情', '蔬菜', '数字中国', '顺丰速运', '台湾海峡形势', '特斯拉', '上海超级工厂', '外交部发言人,中国公民,苏丹邻国', '网络媒体', '乌克兰']等等。
domain: 热点词汇对应的细分股票板块，比如 ['新能源汽车', '5G概念', '人工智能', '区块链', '医疗健康', '云计算', '工业互联网', '智能家居', '物联网', '文化传媒', '海外并购', '稀缺资源', '长江三角洲', '粤港澳大湾区', '食品饮料', '家用电器', '服饰', '纺织制造', '家具家居用品制造', '医药制造', '中成药', '生物制药', '医疗器械服务', '银行', '保险', '证券', '房地产开发', '建筑工程', '专业工程建设', '物业服务', '石油开采', '天然气开采', '煤炭开采', '机械制造', '电力设备', '化工原料', '化工制品', '计算机设备', '通信设备', '半导体及元件', '纺织服装', '旅游酒店', '电子竞技', '网络游戏', 'c2m', '新零售', '网红', '苹果', '消费电子', '苹果产业链', '航空航天', '军工板块', '6g', '卫星通讯', '卫星定位', '影视概念', '影视', '汽车整车', '锂电池', '贸易板块', '互联网', '迪斯尼','数字货币', '聚合支付', '虚拟数字人', '智慧城市', '智慧政务', '人造肉', '石墨烯', '电子烟']等等
If this information is not found, output -1.

Format the output as JSON with the following keys:
word
domain

text: {text}

"""


def generate_vocabulary(content, vocabulary_template) -> str:
    try:
        prompt_template = ChatPromptTemplate.from_template(vocabulary_template)
        if len(content) > 1100:
            content = content[0:1100]
        messages = prompt_template.format_messages(text=content)
        response = chat(messages)
        res = response.content
        print("************")
        print(content)
        print("\n")
        print(res)
        print("\n\n\n")
        return res
    except:
        print(f"OpenAI API error")
        return "something wrong"


import threading


def run_with_timeout(func, content, department_template, timeout=5):
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(content, department_template)
            except Exception as e:
                self.result = e

    it = InterruptableThread()
    it.start()
    it.join(timeout)

    if it.is_alive():
        raise TimeoutError()

    if isinstance(it.result, Exception):
        raise it.result

    return it.result


def get_vocabulary_res(content, vocabulary_template):
    try:
        result = run_with_timeout(generate_vocabulary, content, vocabulary_template, timeout=6)
        return result
    except TimeoutError:
        print("Function timed out")
        return -1


# mm = get_vocabulary_res("她带领团队打破国产计算机无芯可用困局", vocabulary_template)


def insert_influence_vocab_data(data_id, vocabulary, stock_sector):
    try:
        mycursor = cnx.cursor()
        sql = "INSERT INTO influence_vocab (data_id, vocabulary, stock_sector) VALUES (%s, %s, %s)"
        val = (data_id, vocabulary, stock_sector)

        mycursor.execute(sql, val)
        cnx.commit()

        # 输出插入数据的行数
        print(mycursor.rowcount, "record inserted.")
    except:
        print("insert influence_vocab data wrong")


def main(interval):
    read_time = datetime.datetime.now()
    while True:
        cnx = mysql.connector.connect(user='root', password='Admin2022!', host='124.221.209.113', database='bigdata2')
        time.sleep(interval)
        df = pd.read_sql(
            f"SELECT * FROM realtimedata where timespan > '{read_time}'", con=cnx)
        if len(df) > 0:
            maintext = list(df["maintext"])
            id = list(df["id"])
            for i in range(0, len(maintext)):
                result = get_vocabulary_res(maintext[i], vocabulary_template)
                try:
                    print(f"*****result= {result}*****")
                    result = eval(result)
                    print(f"****insert_influence_vocab_data******")
                    insert_influence_vocab_data(
                        id[i], result['word'], result['domain'])
                except:
                    print("something wrong")


if __name__ == '__main__':
    main(20)
