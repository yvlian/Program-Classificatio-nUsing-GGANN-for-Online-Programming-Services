import json
import glob
import os

with open('./data/valid_idx.json', 'r', encoding='utf-8') as vif:
    valid_idx = json.load(vif)['valid_idxs']
train_graphs = []
valid_graphs = []
problem_ids = task_ids = [path.split('\\')[1] for path in glob.glob('./data/*/')]
for graph_path in glob.glob('./graph/*.graph'):

    problem_id = 0
    for path in glob.glob('./data/*/'):
        if os.path.exists(path + graph_path.split('\\')[1].split('.')[0] + '.ast'):
            problem_id = path.split('\\')[1]
            break
    if problem_id == 0:
        continue
    target = [[0. for _ in range(len(problem_ids))]]
    target[0][problem_ids.index(problem_id)] = 1.
    with open(graph_path, 'r', encoding='utf-8') as gf:
        graph = json.load(gf)
        edges = graph['graph_edges']
        nodes = graph['nodes_feature']
    if graph_path.split('\\')[1].split('.')[0] in valid_idx:
        valid_graphs.append({"targets": target,
                             "graph": edges,
                             "node_features": nodes,
                             'task_id_index':problem_ids.index(problem_id)})
    else:
        train_graphs.append({"targets": target,
                             "graph": edges,
                             "node_features": nodes,
                             'task_id_index':problem_ids.index(problem_id)})

with open('./data/train_graphs.json', 'w', encoding='utf-8') as tf:
    json.dump(train_graphs, tf)

with open('./data/valid_graphs.json', 'w', encoding='utf-8') as vf:
    json.dump(valid_graphs, vf)
