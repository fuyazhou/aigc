from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import threading
import time
import pandas as pd
import mysql.connector
import datetime
import fire

os.environ["OPENAI_API_KEY"] = "TcQ70gVKGP6Vu8B1GCUT3BlbkFJYH53nRAaH2SwBRfgmqWY"
# chat = ChatOpenAI(temperature=0.0)
chat = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-0613')
cnx = mysql.connector.connect(user='root', password='Admin2022!', host='124.221.209.113', database='bigdata2')

department_list = ['习近平', '李克强', '人民日报', '工业和信息化部', '工信部装备工业司', '陕西省科技厅', '浙江证监局',
                   '上海证券交易所',
                   '国家发改委', '雄安新区党工委', '国家标准化管理委员会', '科技部', '国家广播电视总局',
                   '宁波市科创板企业上市工作推进会', '中关村管委会', '中国证券监督管理委员会', '上海经信委',
                   '黑龙江省国资委', '文化和旅游部', '深圳市国资委', '国防部', '民航局', '北京证监局', '商务部',
                   '江西省发改委', '泰国数字经济和社会部', '天津市卫生健康委', '上海证监局', '教育部', '欧盟',
                   '国家市场监督管理总局', '中国银行保险监督管理委员会', '河南证监局', '国家知识产权局',
                   '人力资源和社会保障部', '上海市科技创业中心', '民航局工信部', '中国工信部', '安徽国资委',
                   '自然资源部', '中金公司', '河南省金融局', '中国广播电视网络有限公司', '德国政府',
                   '工信部中国电子技术标准化研究院', '山东省能源局', '工信厅', '新疆发改委', '交通运输部',
                   '美国环保署', '新一代人工智能发展规划推进办公室', '证监会', '国家体育总局经济司', '发改委',
                   '省科技厅', '国家统计局', '浙江省发改委', '财政厅',
                   '上海市生态环境局', '山东证监局', '财政部', '工信部', '上海自贸区', '青岛证监局',
                   '生态环境部', '北京市委', '科技部高技术研究发展中心', '国家发展和改革委员会', '国务院',
                   '国家卫健委', '北京市科委', '国家药品监督管理局', '海南省市场监督管理局', '吉林农科院',
                   '国家医疗保障局', '辽宁省交通运输厅', '国家体育总局', '商务部研究院专家组']

department_template = """\
你是一个信息提取专家，下面的新闻中是否提及
['习近平','李克强','人民日报','工业和信息化部', '工信部装备工业司', '陕西省科技厅', '浙江证监局', '上海证券交易所',
 '国家发改委', '雄安新区党工委', '国家标准化管理委员会', '科技部', '国家广播电视总局',
 '宁波市科创板企业上市工作推进会', '中关村管委会', '中国证券监督管理委员会', '上海经信委',
 '黑龙江省国资委', '文化和旅游部', '深圳市国资委', '国防部', '民航局', '北京证监局', '商务部',
 '江西省发改委', '泰国数字经济和社会部', '天津市卫生健康委', '上海证监局', '教育部', '欧盟',
 '国家市场监督管理总局', '中国银行保险监督管理委员会', '河南证监局', '国家知识产权局',
 '人力资源和社会保障部', '上海市科技创业中心', '民航局工信部', '中国工信部', '安徽国资委',
 '自然资源部', '中金公司', '河南省金融局', '中国广播电视网络有限公司', '德国政府',
 '工信部中国电子技术标准化研究院', '山东省能源局', '工信厅', '新疆发改委', '交通运输部',
 '美国环保署', '新一代人工智能发展规划推进办公室', '证监会', '国家体育总局经济司', '发改委',
 '省科技厅', '国家统计局', '浙江省发改委', '财政厅',
 '上海市生态环境局', '山东证监局', '财政部', '工信部', '上海自贸区', '青岛证监局',
 '生态环境部', '北京市委', '科技部高技术研究发展中心', '国家发展和改革委员会', '国务院',
 '国家卫健委', '北京市科委', '国家药品监督管理局', '海南省市场监督管理局', '吉林农科院',
 '国家医疗保障局', '辽宁省交通运输厅', '国家体育总局', '商务部研究院专家组']等部门。如果提及的话，输出该部门。
If this information is not found, output -1.

text: {text}
output：
"""


def generate_department(content, department_template) -> str:
    try:
        for i in department_list:
            if str(i) in str(content):
                print("*******检索*********")
                res = str(i)
                location_prompt = f"""下面的组织或者机构属于哪个国家？如果提及省市的也输出省市，直接输出国家或者省市
                                  text: {res}
                                  output:
                                  """
                location = chat.predict(location_prompt)
                return res, location

        prompt_template = ChatPromptTemplate.from_template(department_template)
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

        if str(res) not in str(content):
            return -1

        if res != -1:
            location_prompt = f"""下面的组织或者机构属于哪个国家？如果提及省市的也输出省市，直接输出国家或者省市
                  text: {res}
                  output:
                  """
            location = chat.predict(location_prompt)

        return res, location
    except:
        print(f"OpenAI API error")
        return "something wrong"


import threading


def run_with_timeout(func, content, department_template, timeout=10):
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


def get_department_res(content, department_template):
    try:
        result = run_with_timeout(generate_department, content, department_template, timeout=10)
        return result
    except TimeoutError:
        print("Function timed out")
        return -1, -1


def insert_department_data(realtimedata_id, department, locaton):
    try:
        cnx = mysql.connector.connect(user='root', password='Admin2022!', host='124.221.209.113', database='bigdata2')
        mycursor = cnx.cursor()
        sql = "INSERT INTO department (realtimedata_id, department, location) VALUES (%s, %s, %s)"
        val = (str(realtimedata_id), str(department), str(locaton))

        mycursor.execute(sql, val)
        cnx.commit()

        # 输出插入数据的行数
        print(mycursor.rowcount, "record inserted.")
    except:
        print("insert department data wrong")


def main(data_path: str):
    df = pd.read_excel(data_path)

    if len(df) > 0:
        maintext = list(df["text"])
        id = list(df["id"])
        for i in range(0, len(maintext)):
            try:
                res, location = get_department_res(maintext[i], department_template)
                print(f"*****department result= {res} ,  location= {location}*****")

                print(f"****insert_department_data******")
                insert_department_data(
                    str(id[i]), str(res), str(location))
            except:
                print("something wrong")
                continue


if __name__ == "__main__":
    fire.Fire(main)
