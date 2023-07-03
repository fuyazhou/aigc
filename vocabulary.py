from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import threading
import time
import pandas as pd
import mysql.connector
import datetime

os.environ["OPENAI_API_KEY"] = "TcQ70gVKGP6Vu8B1GCUT3BlbkFJYH53nRAaH2SwBRfgmqWY"

# chat = ChatOpenAI(temperature=0.0)
chat = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-0613')
cnx = mysql.connector.connect(user='root', password='Admin2022!', host='124.221.209.113', database='bigdata2')

vocab2section = {"“五一”小长假": ["旅游酒店", "在线旅游"], "BLG": ["电子竞技"], "GG": ["电子竞技"],
                 "coa6": ["电子竞技"], "DOTA2": ["网络游戏"], "DRG": ["电子竞技"], "eStar": ["电子竞技"],
                 "狼队": ["电子竞技"], "EDG": ["电子竞技"], "GUCCI": ["c2m", "纺织服装", "新零售", "网红"],
                 "Hero": ["电子竞技"], "MTG": ["电子竞技"], "iPhone15Pro": ["苹果", "消费电子", "苹果产业链"],
                 "JDG": ["电子竞技"], "LGD": ["电子竞技"], "LPL": ["电子竞技"], "RNG": ["电子竞技"],
                 "SpaceX": ["航空航天", "军工板块", "5g", "6g", "卫星通讯", "卫星定位"],
                 "星舰": ["航空航天", "军工板块", "5g", "6g", "卫星通讯", "卫星定位"], "TTG": ["电子竞技"],
                 "AG": ["电子竞技"], "WBG": ["电子竞技"], "WTT": ["电子竞技"], "北京国际电影节": ["影视概念"],
                 "北影节": ["影视"], "比亚迪": ["汽车整车", "新能源汽车", "锂电池"], "布林肯": ["贸易板块"],
                 "访华": ["贸易板块"], "出境游": ["旅游酒店", "在线旅游"], "旅行社": ["旅游酒店", "在线旅游"],
                 "大厂": ["互联网"], "迪士尼": ["迪斯尼"], "短视频平台": ["网红直播"],
                 "俄罗斯": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "基辅": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "乌克兰": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "俄罗斯外交部长": ["贸易板块"],
                 "俄乌局势": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "俄总统": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "普京": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "二阳": ["新冠药物", "新冠板块"], "二次感染": ["新冠药物", "新冠板块"],
                 "新冠": ["新冠药物", "新冠板块"], "反导拦截": ["航空航天", "军工"], "房价房租": ["房地产", "租售同权"],
                 "分裂中国": ["国防军工", ""], "台湾": ["国防军工", ""], "风云三号": ["卫星通讯", "卫星导航"],
                 "国产": ["国产芯片", "ai算力芯片", "汽车芯片"], "芯": ["国产芯片", "ai算力芯片", "汽车芯片"],
                 "国际劳动节": ["旅游酒店", "在线旅游"], "五一": ["旅游酒店", "在线旅游"],
                 "国家航天局": ["航空航天", "军工"], "火星探测": ["航空航天", "军工"], "中国航天": ["航空航天", "军工"],
                 "中国航天日": ["航空航天"], "国家互联网信息办公室": ["数字经济", "互联网"],
                 "互联网普及率": ["数字经济", "互联网"], "数字经济规模": ["数字经济", "互联网"],
                 "国内旅游": ["旅游酒店", "在线旅游"], "航天": ["航空航天"], "航天日": ["航空航天", "军工"],
                 "华为": ["华为", "消费电子"], "华为笔记本": ["华为", "消费电子"],
                 "超材料天线技术": ["华为", "消费电子"], "极氪发布会": ["汽车整车", "新能源汽车", "锂电池"],
                 "间谍活动": ["网络安全", ""], "中国国家安全": ["网络安全", ""], "间谍": ["网络安全", ""],
                 "进口": ["农业种植", "社区团购"], "榴莲": ["农业种植", "社区团购"], "抗原": ["新冠药物", "新冠板块"],
                 "新冠二次感染": ["新冠药物", "新冠板块"],
                 "克里姆林宫": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "裸眼3D": ["裸眼3D", "ar", "vr", "mr", "数字孪生"],
                 "马斯克": ["航空航天", "军工板块", "5g", "6g", "卫星通讯", "卫星定位"],
                 "特斯拉": ["汽车整车", "新能源汽车", "锂电池"], "买房": ["房地产", "租售同权"],
                 "美国国债": ["贸易", "证券", "自由贸易港", "自贸区"],
                 "美国太空探索技术公司": ["航空航天", "军工板块", "5g", "6g", "卫星通讯", "卫星定位"],
                 "欧冠": ["世界杯", "啤酒"], "苹果": ["苹果", "消费电子"],
                 "苹果公司": ["苹果", "消费电子", "苹果产业链"], "切尔西": ["世界杯", "啤酒"],
                 "布莱顿": ["世界杯", "啤酒"], "上海车展": ["汽车整车", "新能源汽车", "锂电池"],
                 "直播": ["汽车整车", "新能源汽车", "锂电池", "网红直播"], "社交媒体": ["新冠药物", "新冠板块", "传媒"],
                 "疫情": ["新冠药物", "新冠板块", "传媒"], "蔬菜": ["农业种植", "农耕", "大农业"],
                 "数字中国": ["数字经济", "互联网"], "顺丰速运": ["智慧物流", "冷链", "社区团购"],
                 "台湾海峡形势": ["国防军工", ""], "上海超级工厂": ["新能源汽车", "锂电池"], "外交部发言人": ["军工"],
                 "中国公民": ["军工"], "苏丹邻国": ["军工"], "网络媒体": ["网红直播"],
                 "乌克兰国家通讯社": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "乌克兰总统": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "票房": ["旅游酒店", "在线旅游"], "五一档": ["旅游酒店", "在线旅游"],
                 "五一黄金周": ["旅游酒店", "在线旅游"], "五一假期": ["旅游酒店", "在线旅游"],
                 "卫星": ["旅游酒店", "在线旅游"], "五一小长假": ["旅游酒店", "在线旅游"],
                 "消博会": ["贸易板块", "自贸区"], "小米": ["小米概念", "消费电子"],
                 "小米13Ultra": ["小米概念", "消费电子"], "小米发布会": ["小米概念", "消费电子"],
                 "新型冠状病": ["新冠药物", "新冠板块"], "学区房": ["房地产", "租售同权"],
                 "云逛": ["贸易板块", "新零售"], "载人航天": ["航空航天", "军工"], "整形手术": ["医学美容"],
                 "直播带货": ["网红直播"], "直播销售": ["网红直播"], "中国航天大会": ["航空航天", "军工"],
                 "中国空间站": ["航空航天", "军工"], "航天服": ["航空航天", "军工"], "中国时间银行": ["养老产业"],
                 "时间银行": ["养老产业"], "主播": ["网红直播"], "住房保障署": ["房地产", "租售同权"],
                 "公租房": ["房地产", "租售同权"], "淄博": ["旅游酒店", "在线旅游"], "淄博市": ["旅游酒店", "在线旅游"],
                 "俄空天军总司令": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "泽连斯基": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "乌": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"], "美国国务卿": ["贸易板块"],
                 "奢侈品": ["c2m", "纺织服装", "新零售"], "路易威登": ["c2m", "纺织服装", "新零售"],
                 "阿斯巴甜": ["食品饮料"], "元气森林": ["食品饮料"], "无糖可乐": ["食品饮料"],
                 "游戏": ["电子竞技", "手游"], "消失的她": ["影视", "传媒"], "中美民间友好": ["贸易板块"],
                 "电影": ["影视", "传媒"], "旅行博主": ["网红经济"],
                 "俄大将": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "乌总统": ["石油", "天然气", "稀有气体", "黄金", "军工", "钾肥", "农产品"],
                 "lv": ["c2m", "纺织服装", "新零售"], "减碳": ["低碳科技", "低碳冶金", "内地低碳"],
                 "低碳": ["低碳科技", "低碳冶金", "内地低碳"], "氢能": ["氢能源", "燃料电池"]}

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
        res1 = {}
        for i in vocab2section.keys():
            if str(i) in str(content):
                print("**** 提取关键词 *****")
                res1['word'] = i
                res1['domain'] = vocab2section[i]
                print(str(res1))
                return res1

        prompt_template = ChatPromptTemplate.from_template(vocabulary_template)
        if len(content) > 1100:
            content = content[0:1100]
        content = content.replace("\n", "")
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
        result = run_with_timeout(generate_vocabulary, content, vocabulary_template, timeout=10)
        return result
    except TimeoutError:
        print("Function timed out")
        return -1


# mm = get_vocabulary_res("她带领团队打破国产计算机无芯可用困局", vocabulary_template)


def insert_influence_vocab_data(data_id, vocabulary, stock_sector):
    try:
        cnx = mysql.connector.connect(user='root', password='Admin2022!', host='124.221.209.113', database='bigdata2')
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
    while True:
        read_time = datetime.datetime.now()
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
                    if type(result) == str:
                        result = eval(result)
                    print(f"****insert_influence_vocab_data******")
                    insert_influence_vocab_data(
                        str(id[i]), str(result['word']), str(result['domain']))
                except:
                    print("something wrong")
                    continue


if __name__ == '__main__':
    main(20)
