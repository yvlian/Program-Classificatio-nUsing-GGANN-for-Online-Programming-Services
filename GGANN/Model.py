#!/usr/bin/env/python

from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict, namedtuple
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import time

from BaseModel import BaseModel
from inits import SMALL_NUMBER,glorot_init

GGNNWeights = namedtuple('GGNNWeights', ['edge_weights',
                                         'edge_biases',
                                         'edge_type_attention_weights',
                                         'rnn_cells', ])
"""
                     V : 节点数目
                     D/h_dim: 隐藏层状态维度
                     E: 当前类型边的数目
                     M: 信息数目(汇总所有的E)
"""


class Model(BaseModel):
    def __init__(self, args):
        # 从BaseModel继承的模型超参数
        super().__init__(args)

    @classmethod
    def default_params(cls):
        # 定义模型超参数
        params = dict(super().default_params())
        params.update({
            'batch_size': 100000,
            'use_edge_bias': False,
            'use_propagation_attention': True,
            'use_edge_msg_avg_aggregation': False,
            'residual_connections': {  # For layer i, specify list of layers whose output is added as an input
                "2": [0],
                "4": [0, 2]
            },

            'layer_timesteps': [2, 2, 1, 2, 1],  # number of layers & propagation steps per layer

            'graph_rnn_cell': 'GRU',  # GRU or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': 1.,
        })
        return params

    def prepare_specific_graph_model(self) -> None:
        """
         定义GNN传播层神经网络参数，并且使用字典GGNNWeights保存
                1.定义GGANN中的权重，偏置，注意力系数，激活函数
                2.增加5个网络输入： 节点初始化参数， 邻接矩阵， 邻接矩阵权重， 节点集合，  图状态保持概率
                :return:
        """
        # 隐藏层维度
        h_dim = self.params['hidden_size']
        # 初始节点向量
        # [[1 0 0 0 ... 0 0]]   一个batch里面节点数目*节点one-hot编码维度
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, h_dim],
                                                              name='node_features')
        # 节点连接关系
        # [[ [0 1] [1 0]...]...]     某种类型边（src,dest）总数*边类型数目
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.num_edge_types)]
        # [[3 1 0 0 0 0 0]]   batch节点数目 * 总边类型数目(7)
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.num_edge_types],
                                                                          name='num_incoming_edges_per_type')
        # 图编号和节点编号对应关系
        # [[0 1]...[0 44] [1 45] ...[1 88] [2 89]....] [batch图编号，图节点在batch中编号]
        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int64, [None, 2], name='graph_nodes_list')
        # 输出层dropout舍弃概率
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        # 激活函数名称
        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)
        # 定义GNN传播层神经网络权重，偏置和激活函数单元
        self.weights = {}
        # edge weights, biases, attention weights,  gated units
        self.gnn_weights = GGNNWeights([], [], [], [])

        # gnn传播层
        # gnn_layer_idx:0-4
        for layer_idx in range(len(self.params['layer_timesteps'])):
            # 每个gnn传播层变量命名空间
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                # 使用glorot方法初始化[7*h_dim, h_dim]的权重参数
                edge_weights = tf.Variable(glorot_init([self.num_edge_types * h_dim, h_dim]),
                                           name='gnn_edge_weights_%i' % layer_idx)

                """
                  shape：[N,D,D]
                  权重维度[h_dim, h_dim]
                  每层网络，每种边，一组权重  
                """
                # weights： 维度[7*h_dim, h_dim] 转化为 维度[7,h_dim,h_dim]
                edge_weights = tf.reshape(edge_weights, [self.num_edge_types, h_dim, h_dim])
                # 添加权重参数进gnn_weights.edge_weights,方便之后取出使用
                self.gnn_weights.edge_weights.append(edge_weights)

                # 注意力权重系数
                if self.params['use_propagation_attention']:
                    # 定义注意力层权重参数
                    self.gnn_weights.edge_type_attention_weights.append(
                        tf.Variable(np.ones([self.num_edge_types], dtype=np.float32),
                                    name='edge_type_attention_weights_%i' % layer_idx))
                # gnn传播层添加偏置参数
                if self.params['use_edge_bias']:
                    self.gnn_weights.edge_biases.append(
                        tf.Variable(np.zeros([self.num_edge_types, h_dim], dtype=np.float32),
                                    name='gnn_edge_biases_%i' % layer_idx))
                # 激活函数名称
                cell_type = self.params['graph_rnn_cell'].lower()
                if cell_type == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
                elif cell_type == 'rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
                else:
                    raise Exception("Unknown RNN cell type '%s'." % cell_type)
                # 带有激活函数的rnn神经网络单元
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     state_keep_prob=self.placeholders['graph_state_keep_prob'])
                self.gnn_weights.rnn_cells.append(cell)

    def compute_final_node_representations(self) -> tf.Tensor:
        """
            GNN网络计算节点状态向量框架，计算节点最后的状态向量
            :return: 返回最后一层节点状态向量
        """
        # 装载节点在不同时刻的状态向量
        node_states_per_layer = []
        # 时刻t=0时，顶点状态使用顶点初始向量表达
        node_states_per_layer.append(self.placeholders['initial_node_representation'])
        # 统计一个batch里面的节点数目
        num_nodes = tf.shape(self.placeholders['initial_node_representation'], out_type=tf.int32)[0]

        # 信息传递目标
        message_targets = []
        # 顶点间边类型的信息
        message_edge_types = []

        # adjacency_lists:[[adjacency_list_for_edge_type1],[adjacency_list_for_edge_type2]........]
        # {边类型：[src,dest]} 字典
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(
                self.placeholders['adjacency_lists']):  # shape[E, 2]
            # shape[E, 1] dest
            # 每种边上的dest节点集合
            edge_targets = adjacency_list_for_edge_type[:, 1]  # shape[E, 1] dest
            # 将dest节点作为节点间信息传播的目标
            message_targets.append(edge_targets)

            # [1,1,1,1...]  or  [2,2,2,2,2....]
            # 复制边类型标签，目的是为了让边类型标签数目和目标节点数目相等并且对齐
            message_edge_types.append(tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)
        """
              t1 = [[1, 2, 3], [4, 5, 6]]
              t2 = [[7, 8, 9], [10, 11, 12]]
              tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
              tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
             
              # tensor t3 with shape [2, 3]
              # tensor t4 with shape [2, 3]
              tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
              tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
        """
        # 将目标节点集合拼接成一个列表
        message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
        # 将边类型拼接成一个列表
        message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]
        # gnn传播层数：每个传播层的迭代次数
        for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
            # gnn传播层数编号（tensorflow对于传播层变量的命名范围）
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                """
                    抽取残差信息
                    {"2":[0],"4":[0, 2]}
                """
                # 获取第layer_idx的残差连接层[XX,XX.....]
                layer_residual_connections = self.params['residual_connections'].get(str(layer_idx))

                # 残差层节点状态
                if layer_residual_connections is None:
                    layer_residual_states = []
                else:
                    # 获取残差层，节点的状态向量集合
                    layer_residual_states = [node_states_per_layer[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]
                """
                    tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
                    tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引
                """
                # 传播Attention
                if self.params['use_propagation_attention']:
                    # 每种边的权重系数
                    message_edge_type_factors = tf.nn.embedding_lookup(
                        params=self.gnn_weights.edge_type_attention_weights[layer_idx],
                        ids=message_edge_types)  # Shape [M]

                # 更新每层节点新状态
                # 直接复制t-1时刻的顶点状态向量，目的是更快的对齐顶点数目。
                node_states_per_layer.append(node_states_per_layer[-1])
                # 每一个gnn层需要迭代的次数
                for step in range(num_timesteps):
                    with tf.variable_scope('timestep_%i' % step):
                        # 来自邻居节点，不同边的信息汇总
                        messages = []
                        # 邻居节点状态
                        message_source_states = []

                        # 收集incoming信息
                        for edge_type_idx, adjacency_list_for_edge_type in enumerate(
                                self.placeholders['adjacency_lists']):

                            # 邻居节点编号
                            edge_sources = adjacency_list_for_edge_type[:, 0]
                            # 邻居节点状态
                            edge_source_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],#选取一个张量里面索引对应的元素,ids就是對應的索引
                                                                        ids=edge_sources)
                            # 分边类型，使用邻居节点状态和边权重乘
                            all_messages_for_edge_type = tf.matmul(edge_source_states,
                                                                   self.gnn_weights.edge_weights[layer_idx][
                                                                       edge_type_idx])
                            # 源节点与神经网络权重相乘后的结果(单一边类型)
                            messages.append(all_messages_for_edge_type)
                            # (所有边类型)源节点向量与神经网络权重相乘结果，并且汇集
                            message_source_states.append(edge_source_states)
                        # 连接成一维的向量
                        messages = tf.concat(messages, axis=0)

                        # 使用节点注意力机制
                        if self.params['use_propagation_attention']:
                            # 查找中心节点和邻居节点隐藏状态
                            message_source_states = tf.concat(message_source_states, axis=0)
                            message_target_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                           ids=message_targets)
                            """
                                tf.einsum(‘ij,jk->ik’, ts1,ts2) #矩阵乘法 
                                tf.einsum(‘ij->ji’,ts1) #矩阵转置
                            """
                            # 矩阵乘法计算邻居节点和当前节点的注意力值 dot
                            message_attention_scores = tf.einsum('mi,mi->m', message_source_states,
                                                                 message_target_states)
                            message_attention_scores = message_attention_scores * message_edge_type_factors

                            # print("layer%i : timesteps: %i" % (layer_idx, num_timesteps))

                            """
                                1.对邻居节点做softmax计算概率
                                2.因为邻居节点数目变化，不能直接使用tf.softmax
                                3.手动实现softmax
                            """
                            # 步骤（1）：获取邻居节点attention的最大值
                            """
                                tf.unsorted_segment_max:沿着segments_ids计算data最大值
                            """
                            message_attention_score_max_per_target = tf.unsorted_segment_max(
                                data=message_attention_scores,
                                segment_ids=message_targets,
                                num_segments=num_nodes)

                            # 步骤（2）：再次将max-out分布到相应的消息中，并转换分数
                            """
                                tf.gather: 根据索引从参数轴上收集切片. 
                                
                                按照message_targets排列最大的attention_scores
                            """
                            message_attention_score_max_per_message = tf.gather(
                                params=message_attention_score_max_per_target,
                                indices=message_targets)  # Shape [M]
                            message_attention_scores -= message_attention_score_max_per_message

                            # 步骤(3): 每个目标执行Exp, sum up , compute exp(score) / exp(sum) 作为attention概率
                            message_attention_scores_exped = tf.exp(message_attention_scores)

                            """
                                按照message_targets汇总message_attention_scores_expand
                            """
                            message_attention_score_sum_per_target = tf.unsorted_segment_sum(
                                data=message_attention_scores_exped,
                                segment_ids=message_targets,
                                num_segments=num_nodes)

                            """
                               切片重组
                            """
                            message_attention_normalisation_sum_per_message = tf.gather(
                                params=message_attention_score_sum_per_target,
                                indices=message_targets)

                            """
                                attention概率归一化
                            """
                            message_attention = message_attention_scores_exped / (
                                    message_attention_normalisation_sum_per_message + SMALL_NUMBER)

                            # 步骤（4）：对节点使用attention概率
                            # tf.expand_dims: 在第axis位置增加一个维度
                            messages = messages * tf.expand_dims(message_attention, -1) #給函數增加維度

                        """
                            targets消息汇总
                        """
                        incoming_messages = tf.unsorted_segment_sum(data=messages,   #計算張量片段的和
                                                                    segment_ids=message_targets,
                                                                    num_segments=num_nodes)

                        if self.params['use_edge_bias']:
                            incoming_messages += tf.matmul(self.placeholders['num_incoming_edges_per_type'],
                                                           self.gnn_weights.edge_biases[layer_idx])

                        if self.params['use_edge_msg_avg_aggregation']:
                            num_incoming_edges = tf.reduce_sum(self.placeholders['num_incoming_edges_per_type'],
                                                               keep_dims=True, axis=-1)
                            # 均值化
                            incoming_messages /= num_incoming_edges + SMALL_NUMBER

                        # 每个节点拼接残差层节点信息
                        incoming_information = tf.concat(layer_residual_states + [incoming_messages],
                                                         axis=-1)

                        # 将顶点特征和当前时刻messages输入RNN
                        node_states_per_layer[-1] = self.gnn_weights.rnn_cells[layer_idx](incoming_information,
                                                                                          node_states_per_layer[-1])[
                            1]

        return node_states_per_layer[-1]

    def gated_regression(self, last_h, regression_gate, regression_transform):
        """
        得到最终的图输出结果
        :param last_h:
        :param regression_gate:
        :param regression_transform:
        :return:
        """
        # 将cur_node_state 与 初始化node连接组成gate_input （node_num*200）
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)

        """
             1.将gate_input输入到regression_gate_task_i这个MLP
             2.输入sigmoid函数，变换到0-1
             3. 输入到regression_transform_task_i这个MLP
             4. 得到结果gated_outputs(node_num*1)
        """
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)

        # 累计所有节点成一个图表达
        num_nodes = tf.shape(gate_input, out_type=tf.int64)[0]
        graph_nodes = tf.SparseTensor(indices=self.placeholders['graph_nodes_list'],
                                      values=tf.ones_like(self.placeholders['graph_nodes_list'][:, 0],dtype=tf.float32),
                                      dense_shape=[self.placeholders['num_graphs'], num_nodes])
        # 返回图分类的类别[0,1]
        return tf.sparse_tensor_dense_matmul(graph_nodes, gated_outputs)

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        # 数据预处理:  将json文件数据转化成GNN模型输入数据processed_graphs
        processed_graphs = []
        """
           processed_graphs: {"adjacency_lists","num_incoming_edge_per_type"，"init","labels"}
           adjacency_lists: 顶点邻接列表  dict{int:array(list(list))}             {边类型id:array([[src,dest]])}
           num_incoming_edge_per_type： dict(int:dict{int:int})        {"edge_type":{"dest":num}}
           init: 顶点初始向量，one-hot向量  list(list) [图节点数目*节点ont-hot向量]
           labels: 图分类标签
        """
        for d in raw_data:
            """
                adjacency_lists    list(list(list))  [[ [0 1] [1 0]...]...]     某种类型边（src,dest）总数*边类型数目
                num_incoming_edges_per_type     list(list)     [[3 1 0 0 0 0 0]]   batch节点数目 * 总边类型数目(7)
                batch_node_features             list(list)      [[1 0 0 0 ... 0 0]]   一个batch里面节点数目*节点one-hot编码维度
                target_values                   list(list)      [[1 0...] [0 0...] [0 1....]]  label数目*batch图数目
            """
            (adjacency_lists, num_incoming_edge_per_type) = self.__graph_to_adjacency_lists(d['graph'])
            processed_graphs.append({"adjacency_lists": adjacency_lists,
                                     "num_incoming_edge_per_type": num_incoming_edge_per_type,
                                     "init": d["node_features"],
                                     "labels": d["targets"][0],
                                     'task_id_index':d['task_id_index']})

        if is_training_data:
            # 训练数据随机打乱，使得训练效果更好
            np.random.shuffle(processed_graphs)
            # 控制训练数据大小
            for task_id in self.params['task_ids']:
                task_sample_ratio = self.params['task_sample_ratios']
                if task_sample_ratio is not None:
                    ex_to_sample = int(len(processed_graphs) * task_sample_ratio)
                    for ex_id in range(ex_to_sample, len(processed_graphs)):
                        processed_graphs[ex_id]['labels'][task_id] = None
        # 返回处理过的模型可直接输入的图数据
        return processed_graphs

    def __graph_to_adjacency_lists(self, graph) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
        """
        数据预处理，目的是为了解析出 final_adj_lists, adjacency_lists
              final_adj_lists：排序后的adjacency_lists  dict{int:array(list(list))}   {边类型id:array([[src,dest]])}
              num_incoming_edges_dicts_per_type： dict(int:dict{int:int})      {"edge_type":{"dest":num}}
        :param graph:
        :return:
        """
        # 定义adj_lists，num_incoming_edges_dicts_per_type两个数据容器
        adj_lists = defaultdict(list)
        num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
        """
            1. 从原始数据解析出(src,edge,dest)
            2. 解析(src,edge_type,dest)成{"edge_type":{"dest":num}}
            3. 解析(src,edge_type,dest)成{边类型id：[[src,dest]]}
        """
        for src, e, dest in graph:
            # edge_type 的编号从0开始
            fwd_edge_type = e - 1
            # 添加一条记录进adj_lists容器
            adj_lists[fwd_edge_type].append((src, dest))
            # 添加一条记录进num_incoming_edges_dicts_per_type容器
            num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1

            # 如果前向fwd和反向bkwd看成一种边
            if self.params['tie_fwd_bkwd']:
                # 添加一条记录(dest,fwd_edge_type,src)进adj_lists
                adj_lists[fwd_edge_type].append((dest, src))
                # 添加一条记录{fwd_edge_type:[[dest,src]]}进num_incoming_edges_dicts_per_type
                num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1
        # 按照edge_type编号大小顺序排序adj_lists，组成final_adj_lists
        final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                           for e, lm in adj_lists.items()}

        # 增加额外的反向边类型，反向边节点对，反向信息输入边
        if not (self.params['tie_fwd_bkwd']):
            for (edge_type, edges) in adj_lists.items():
                # 反向边编号=对应的前向边编号+前向边边种类数目
                bwd_edge_type = self.num_edge_types + edge_type
                # 复制邻接边
                final_adj_lists[bwd_edge_type] = np.array(sorted((y, x) for (x, y) in edges), dtype=np.int32)
                # 复制信息输入边
                for (x, y) in edges:
                    num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

        return final_adj_lists, num_incoming_edges_dicts_per_type

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        """
        构建能够直接喂如模型的minibatch_data
        :param data:
        :param is_training:
        :return:
        """
        # 随机打乱数据顺序
        if is_training:
            np.random.shuffle(data)
        # dropout舍弃概率
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        # 当前第num_graphs个数据图计数
        num_graphs = 0

        # 输出数据中图的总数
        print("The number of graphs in data %i\n" % (len(data)))
        # 当数据中的图数据还没有完全被使用
        while num_graphs < len(data):
            # batch中的图数据数目
            num_graphs_in_batch = 0
            # batch中的node初始向量集合
            batch_node_features = []
            batch_target_task_values = []
            batch_target_task_mask = []
            """
                1. 得到当前图中节点的batch编号
                2. 构建adj_lists容器  ->  [[]*7]
                3. batch_adjacency_lists -> list(list)  [[当前图['adjacency_lists']+node_offset]*边类型数目]
            """
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            """
               # 图节点数据*边数目 = N*7
                    [ [1 0 0 0 0 0 0]   # 节点1有一条类型为1的边
                      [0 0 0 0 0 1 1]   # 节点2各有一条类型为6和7的边
                      .....
                    ]
            """
            batch_num_incoming_edges_per_type = []
            # list(tuple)   [(batch中图编号,batch中节点编号)]
            batch_graph_nodes_list = []
            task_id_index_list = []
            # 记录batch中节点数目
            node_offset = 0

            node_offsets = [0]
            """
               当前图编号小于图数据总数，并且累计的节点总数不大于一个batch_size设置的大小
               目的：因为每个batch中节点数目不一样，这样做
                      1.可以保证一个图的节点在同一个batch中
                      2.可以保证每个batch中的图数目能够随着batch_size和图节点数目动态调整
                      3.使得喂入模型的数据能够动态适应变化的图节点数目
            """
            while num_graphs < len(data) and node_offset + len(data[num_graphs]['init']) < self.params['batch_size']:
                # 取出data中当前图数据元素cur_graph
                cur_graph = data[num_graphs]
                # 获得cur_graph中图节点数目
                num_nodes_in_graph = len(cur_graph['init'])
                """
                   通过hidden_size超参数和cur_graph['init]设置图节点初始向量
                      1. 节点notation向量 = cur_graph['init]
                      2. 节点初始化向量 = [cur_graph['init'], [0]],即notation向量补0所得。
                      3. 节点初始化向量维度为hidden_size
                """
                padded_features = np.pad(cur_graph['init'],
                                         ((0, 0), (0, self.params['hidden_size'] - self.annotation_size)),
                                         'constant')
                # 将cur_graph节点初始化向量元素添加进batch_node_features数据容器
                batch_node_features.extend(padded_features)
                task_id_index_list.append(cur_graph['task_id_index'])
                """
                   将(cur_graph在batch中图数据编号，cur_graph节点在batch中的节点编号）记录添加进batch_graph_nodes_list
                       1. cur_graph在batch中图数据编号：通过累加计数得到num_graphs_in_batch
                       2. cur_graph节点在batch中的节点编号： 通过cur_graph节点在cur_graph中的编号+cur_graph节点在batch中的节点偏移node_offset得到
                       3. node_offset:  记录当前cur_graph之前出现过的所有节点总数
                """
                batch_graph_nodes_list.extend((num_graphs_in_batch, node_offset + i) for i in range(num_nodes_in_graph))
                # 更新batch_adjacency_lists里的节点编号：cur_graph中节点编号+node_offset
                for i in range(self.num_edge_types):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                # Turn counters for incoming edges into np array:
                # 将数据转化成array数组
                num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.num_edge_types))  # shape [V, E]

                # 迭代cur_graph['num_incoming_edge_per_type'] ->  {"edge_type_id":{"dest":num}}
                for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                    # dict(int:int)-> {"dest":num}
                    for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                        # 解析数据得到   (dest,edge_type,num) -> (node_id,edge_type,edge_count)
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                # 添加(node_id,edge_type,edge_count)数据对到  batch_num_incoming_edges_per_type
                batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

                target_task_values = []
                target_task_mask = []
                for target_val in cur_graph['labels']:
                    if target_val is None:  # This is one of the examples we didn't sample...
                        target_task_values.append(0.)
                        target_task_mask.append(0.)
                    else:
                        target_task_values.append(target_val)
                        target_task_mask.append(1.)
                batch_target_task_values.append(target_task_values)  # shape [g, c]
                batch_target_task_mask.append(target_task_mask)
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

                node_offsets.append(node_offset)

            #if not is_training:
                #print(node_offsets)


            """"
            t = str(time.time())
            filename1 = t+"batchs_graphs"
            with open(filename1, 'w') as batchs_graphs_file:
                batchs_graphs_file.write(str(batchs_graphs_index))
            filename2 = t + "batchs_graphs_nodes_number"
            with open(filename2, 'w') as nodes_number_file:
                nodes_number_file.write(str(batchs_graphs_nodes_number))
            """

            # 定义placeholder
            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),  # shape [g(V), h]
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type,
                                                                                 axis=0),  # shape [g(V), E]
                self.placeholders['graph_nodes_list']:
                    np.array(batch_graph_nodes_list, dtype=np.int32),  # shape [g(V), [num_graph, node_i]]
                self.placeholders['target_values']: np.transpose(batch_target_task_values, axes=[1, 0]),
                self.placeholders['target_mask']: np.transpose(batch_target_task_mask, axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['task_id_index']: task_id_index_list,
                self.placeholders['task_target_values']: np.transpose(np.array(self.params['task_target_values'])),
            }

            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict


def main():
    #args = docopt(__doc__)
    args= {}
    try:
        model = Model(args)
        model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
