import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import networkx as nx
import time


"""
    batch1: graphs: 0-1567   node_offsets: 99957
    batch2: graphs: 1568-1694  node_offsets: 7718
 """


def count_start_graphs(json_graphs):
    sum_offsets = 0
    for i in range(len(json_graphs)):
        num_nodes = len(json_graphs[i]['node_features'])
        sum_offsets += num_nodes
        if sum_offsets == 99957:
            print(i+2)
            break


def select_nodes_from_npy(nodes_start_index, nodes_end_index):
    nodes_vec = np.load('nodes25.npy')
    [rows, cols] = nodes_vec.shape
    nodes = []
    for i in range(rows):
        node = []
        for j in range(cols):
            node.append(nodes_vec[i][j])
        nodes.append(node)
    return nodes[nodes_start_index:nodes_end_index]


def get_graphs(graphs_idx):
    file = open("valid_graphs.json", 'r')
    json_graphs = json.load(file)
    graphs = json_graphs[graphs_idx]['graph']
    nodes = []
    type_edge = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    for i in range(len(graphs)):
        ele = graphs[i]
        s = ele[0]
        e = ele[1]
        t = ele[2]
        if not nodes.__contains__(s):
            nodes.append(s)
        if not nodes.__contains__(t):
            nodes.append(t)

        if e not in type_edge.keys():
            raise Exception("No such edge")
        else:
            type_edge[e].append([s, t])

    return nodes, type_edge[1]


def get_symbols(index):
    symbols = ["==", "&", "&&", "!", "+", ">", "!=",  "-", "=",  "<", "++", ",",
               ">=", "--", "||", "<=", "%", "/", "*", "%=", "+=", "~", "/=", "^",
               "-=", "^=", "*=", ">>", "|", "<<", ">>=", "|=", "&=", "<<=", "->*"]
    return symbols[index-2]

def draw():
    node_offset = [0, 105, 142, 201, 251, 290, 322, 381, 432, 469,
                   537, 575, 621, 679, 757, 789, 821, 856, 911, 944,
                   981, 1017, 1056, 1109, 1142, 1187, 1224, 1280, 1326, 1359,
                   1397, 1434, 1492, 1529, 1567, 1604, 2093, 2140, 2191, 2221,
                   2272, 2332, 2372, 2406, 2436, 2467, 2506, 2537, 2568, 2986,
                   3023, 3061, 3114, 3157, 3204, 3265,3337, 3372, 3421, 3726,
                   3764, 3817, 3855, 3892, 3929, 4014, 4097, 4136, 4186, 4224,
                   4261, 4305, 4349, 4428, 4467, 4500, 4537, 4597, 4634, 4667,
                   4713, 4778, 4816, 4869, 4921, 4972, 5020, 5066, 5102, 5135,
                   5173, 5222, 5260, 5777, 5809, 5840, 5875, 5905, 5939, 5989,
                   6024, 6075, 6126, 6177, 6215, 6302, 6352, 6390, 6428, 6464,
                   6508, 6545, 6598, 6652, 6711, 6765, 6833, 6892, 6928, 6969,
                   7026, 7065, 7098, 7135, 7187, 7617, 7676, 7718]

    nodes = select_nodes_from_npy(0, 104)   # 1604 2092
    clf = KMeans(n_clusters=3)
    y_pred = clf.fit_predict(nodes)

    data_pca_tsne = TSNE(n_components=2).fit_transform(nodes)

    x = [n[0] for n in data_pca_tsne]
    y = [n[1] for n in data_pca_tsne]
    plt.scatter(x, y, c=y_pred*10, marker='o')

    nodes_index, edges = get_graphs(1567)  # 1603
    for i in range(len(nodes_index)):
        node = nodes_index[i]
        print(node)
        if 2 <= node <= 36:
            symbol = get_symbols(node)
            plt.text(x[i]-0.1, y[i]-0.1, symbol, fontsize=8)
        if node == 1:
            plt.text(x[i]-0.1, y[i]-0.1, "FunctionDecl", color='r', fontsize=8)
        if node == 37:
            plt.text(x[i]-0.1, y[i]-0.1, "CompoundStmt", color='r', fontsize=8)
        if node == 38:
            plt.text(x[i]-0.1, y[i]-0.1, "DeclStmt", color='r', fontsize=8)
        if node == 39:
            plt.text(x[i]-0.1, y[i]-0.1, "VarStmt", color='r', fontsize=8)
        if node == 40:
            plt.text(x[i]-0.1, y[i]-0.1, "WhileStmt", color='r', fontsize=8)
        if node == 41:
            plt.text(x[i]-0.1, y[i]-0.1, "BinaryOperator", color='r', fontsize=8)
        if node == 45:
            plt.text(x[i]-0.1, y[i]-0.1, "StringLiteral", color='r', fontsize=8)
        if node == 47:
            plt.text(x[i]-0.1, y[i]-0.1, "IntegerLiteral", color='r', fontsize=8)
        if node == 46:
            plt.text(x[i]-0.1, y[i]-0.1, "UnaryOperator", color='r', fontsize=8)
        if node == 57:
            plt.text(x[i]-0.1, y[i]-0.1, "ForStmt", color='r', fontsize=8)
        if node == 48:
            plt.text(x[i]-0.1, y[i]-0.1, "IfStmt", color='r', fontsize=8)
        if node == 49:
            plt.text(x[i]+0.1, y[i]-0.1, "BreakStmt", color='r', fontsize=8)
        if node == 50:
            plt.text(x[i]-0.1, y[i]-0.1, "ReturnStmt", color='r', fontsize=10)
        if node == 60:
            plt.text(x[i]-0.3, y[i]-0.3, "ContinueStmt", color='r', fontsize=8)
        if node == 71:
            plt.text(x[i]-0.1, y[i]-0.1, "ConditionalOperator", color='r', fontsize=8)
        if node == 75:
            plt.text(x[i]-0.2, y[i]-0.2, "CompoundAssignOperator", color='r', fontsize=8)
        if node == 79:
            plt.text(x[i]-0.1, y[i]-0.1, "DoStmt", color='r', fontsize=8)
        if node == 91:
            plt.text(x[i]-0.1, y[i]-0.1, "array", color='r', fontsize=8)
        if node == 95:
            plt.text(x[i]-0.1, y[i]-0.2, "GotoStmt", color='r', fontsize=8)
        if node == 134:
            plt.text(x[i]-0.3, y[i]-0.3, "SwitchStmt", color='r', fontsize=8)
        if node == 135:
            plt.text(x[i]-0.3, y[i]-0.3, "CaseStmt", color='r', fontsize=8)
        if node == 156:
            plt.text(x[i]-0.3, y[i]-0.3, "Public", color='r', fontsize=8)
        if node == 160:
            plt.text(x[i]-0.3, y[i]-0.3, "protected", color='r', fontsize=8)


    #plt.xlim(-13, 3)
    #plt.ylim(-35, 35)
    plt.show()


def draw_graph():
    G = nx.DiGraph()
    nodes, edges = get_graphs(1567)
    G.add_nodes_from(nodes)    #加点集合
    G.add_edges_from(edges)   #加边集合
    nx.draw(G)
    plt.show()
    time.localtime()
    plt.save()

if __name__ == '__main__':
    draw()