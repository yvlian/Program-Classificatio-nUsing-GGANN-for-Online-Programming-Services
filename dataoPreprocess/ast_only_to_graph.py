from preprocess.ASTOnly2Graph import ASTOnly2Graph
from glob import glob
import json


# 仅仅包含ast的graph生成代码
graph_path = './onlyGraph/'
ast_path = './data/'

# 创建 ./graph/*.graph文件
for ast_file in glob(ast_path + '*/*'):
    print(ast_file)
    graph_file = graph_path + ast_file.split('/')[3].split('.')[0] + '.graph'
    graph = ASTOnly2Graph(ast_file)
    with open(graph_file, 'w', encoding='utf-8') as f:
        json.dump(graph.get_source_graph(), f)

