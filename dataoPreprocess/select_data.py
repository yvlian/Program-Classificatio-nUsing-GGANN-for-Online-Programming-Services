import json
import os
import shutil

"""
    1. 创建题目-代码的文件夹:./data/problem_id/
    2. 将代码文件./ast/solution_idt.ast复制进入题目-代码文件夹
"""
with open('./data/problem_types.json', 'r', encoding='utf-8') as f:
    problem_types = json.load(f)['problem_types']
m = []
for item in problem_types:
    if len(problem_types[item]['acc']) >= 800:
        m.append(item)
        if not os.path.exists('./data/%s/' % item):
            os.mkdir('./data/%s/' % item)
        for solution in problem_types[item]['acc']:
            if os.path.exists('./ast/%dt.ast' % solution):
                shutil.copyfile(src='./ast/%dt.ast' % solution, dst='./data/%s/%d.ast' % (item, solution))

print(len(m))
