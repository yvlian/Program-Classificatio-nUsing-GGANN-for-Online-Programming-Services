"""
  1. Dump data from mysql.
  2. Create json of problem_types include problem title and solution_id belong to this problem.
  3. Save source file into directory named src.
"""
import pymysql.cursors
import json
import pandas as pd
from bs4 import BeautifulSoup
"""
    -- 连接数据库配置
     1. 建立mysql数据库，导入acm.sql文件
     2. 使用pymysql连接数据库
     3. 一定要确保数据库各项配置信息和代码保持一致
"""
connect = pymysql.Connect(
    host='localhost',    # mysql库存访问主机名  http://localhost:3306
    port=3306,           # mysql数据库端口
    user='root',         # mysql数据库登录账号
    passwd='2191218696',        # mysql数据库登录密码
    db='acm',            # mysql数据库名
    charset='utf8')

cursor = connect.cursor()

"""
   1. 删选数据库代码,筛选条件: c++ && AC && solution_id.cpp exists
          1.1 筛选sql语句  -- where language = 1(c++) and result = 4(accepted)
          1.2 额外条件    --  exists solution_id.cpp（c++  file)
"""
sql = "select problem_id, description,`input`,`output` from problem"


cursor.execute(sql)   # 执行sql语句
problem_types = {}
with open('./data/problem_id_title') as f:
    problem_id_title = pd.read_csv(f,header=None)
    problem_id_title.rename(columns={0: 'id', 1: 'title'}, inplace=True)
selected_problem_id = problem_id_title.values
selected_problem = pd.DataFrame(columns=['id','description','input','output'])
for item in cursor.fetchall():

    problem_id = int(item[0])
    description = BeautifulSoup(item[1],'html.parser').get_text().replace('\n',' ')
    input = BeautifulSoup(item[2],'html.parser').get_text().replace('\n',' ')
    output = BeautifulSoup(item[3],'html.parser').get_text().replace('\n',' ')

    if problem_id in selected_problem_id:
        temp = pd.Series({'id':problem_id,"description": description,
                          "input": input,"output": output})
        selected_problem = selected_problem.append(temp, ignore_index=True)

    # save cpp file
selected_problem.to_csv('./data/selected_problem_30',index=False)
# sql = "select p.problem_id, p.title, sc.solution_id, sc.source from problem as p, solution as s, source_code as sc" \
#       " where s.language = 1 and s.result = 4 and s.problem_id = p.problem_id and s.solution_id = sc.solution_id"
#
#
# cursor.execute(sql)   # 执行sql语句
# problem_types = {}
#
# print("start create json of problem_types and save source file:")
# for item in cursor.fetchall():
#
#     problem_id = str(item[0])
#     title = item[1]
#     solution_id = item[2]  # source code index
#     source = item[3]   # source code
#
#     """
#          ---  if not problem then add
#           problem_types: { problem_id:{"title":title,"acc":[]} }
#     """
#     if problem_id not in problem_types.keys():
#         problem_types[problem_id] = {"title": title, "acc": []}
#
#     # add solution id
#     problem_types[problem_id]["acc"].append(solution_id)
#
#     # save cpp file
#     with open('./src/%d.cpp' % solution_id, 'w', encoding='utf-8') as cpp:
#         cpp.write(source)
#
# print("save all source files into ./src/ OK")
#
# # Record problems and their titles and AC source code index
# with open('./data/problem_types.json', 'w', encoding='utf-8') as file:
#     file.write(json.dumps({"problem_types": problem_types}, ensure_ascii=False))
# print("create problem_types.json OK")

