from preprocess.AST_tbcnn_2Graph import AST_tbcnn_2Graph
from glob import glob
import pickle


# 仅仅包含ast的graph生成代码
nodes_pkl_path = './tbcnn_sampler_data/algorithm_nodes.pkl'
ast_path = './data/'

# 创建  algorithm_nodes.pkl
nodes = []
types_count = []
for ast_file in glob(ast_path + '*/*'):
    print(ast_file)
    ast = AST_tbcnn_2Graph(ast_file)
    graph = ast.get_source_graph()
    nodes.append(graph)

    for item in graph:
        node_type = item['node']
        if node_type not in types_count:
            types_count.append(node_type)

    print(len(types_count))
    print(types_count)

# with open(nodes_pkl_path, 'wb') as f:
#         pickle.dump(nodes, f)

