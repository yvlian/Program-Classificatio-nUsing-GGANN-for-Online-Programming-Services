"""
  Transfer cpp file to ast file without include file
"""
import json
import multiprocessing
import threading
import time

from preprocess.progress import process_cpp_file


def process_problem(problem_id, problem):
    print('Start transfer problem %s-->%s acc cpp file to ast file without include file' % (
        problem_id, problem['title']))

    # travel acc
    count = 0
    for solution in problem['acc']:
        t = threading.Thread(target=process_cpp_file, args=(solution,))
        t.setDaemon(True)
        t.start()
        count += 1
        if count == 50:
            count = 0 
            time.sleep(3)
    time.sleep(30)
    print('transfer problem %s-->%s acc cpp file to ast file without include file OK' % (problem_id, problem['title']))


# Init solution list
with open('./data/problem_types.json', 'r', encoding='utf-8') as f:
    problem_types = json.load(f)['problem_types']

# travel problem
# 线程池开启线程
for problem_id, problem in problem_types.items():
    pool = multiprocessing.Pool(processes=16)
    pool.apply_async(func=process_problem, args=(problem_id, problem))
    pool.close()
    pool.join()

print('transfer all source code to ast tree OK')
