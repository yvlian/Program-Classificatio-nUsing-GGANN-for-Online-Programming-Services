"""
   Transfer ast to graph
"""
import json
import re
import pickle
# 仅仅包含ast的graph代码


class AST_tbcnn_2Graph:
    __type_file_path = './data/ast_types.json'
    __ast_file_path = ''

    __index = None
    __ast_lines = None
    __ast_types = None
    __ast_nodes = None
    __node_direct_childes = None
    __node_direct_parents = None
    __functions = None
    __operators = None
    __common_memories = {}

    __current_layer = ''  # Current node layer
    __current_type = ''   # Current node ast type
    __current_memory = ''  # Current node memory
    __current_line = ''    # Current AST line
    __current_one_hot = []   # Current node ont hot of AST types
    __current_childes = []
    __current_key = 0

    __edge_type = {
        "AST": 1,  # AST本身边
        "operand": 2,  # 连接操作数和操作符，从操作数出发
        "LastUse": 3,  # 连接上一个被改变的变量，从当前变量出发
        "ComputedFrom ": 4,  # 连接等式左边和右边变量，从左边出发
        "ReturnsTo": 5,  # 连接return和函数声明
        "FormalArgName": 6,  # 连接形参和实参
        "CallFunction": 7,  # 连接MemberExpr->FieldDecl MemberExpr->CXXMethodDecl DeclRefExpr->FunctionDecl
    }

    __operator = {'CompoundAssignOperator': ['%=', '+=', '/=', '-=', '^=', '*=', '>>=', '|=', '&=', '<<='],
                  'BinaryOperator': ['==', '&&', '+', '>', '!=', '=', '<', '-', ',', '>=', '||', '<=', '%', '/', '*',
                                     '^', '&', '>>', '|', '<<', '->*'],
                  'UnaryOperator': ['&', '!', '-', '++', '--', '~', '*', '+']}

    __graph = None

    # 初始化设置参数
    def __init__(self, path):
        self.__ast_file_path = path
        # 归为None
        self.__index = None
        self.__ast_lines = None
        self.__ast_types = None
        self.__ast_nodes = None
        self.__node_direct_childes = None
        self.__functions = None
        self.__operators = None
        self.__common_memories = {}

        # 加载types.json
        with open(self.__type_file_path, 'r', encoding='utf-8') as type_file:
            self.__ast_types = json.load(type_file)['ast_types']

        # 加载file.ast ->  self.__ast_lines
        self.__load_ast_file()
        # 解析文件-> self.__ast_nodes
        self.__parse_all_line()
        #  构造self.__node_direct_childes =  {layer:[key1,key2,key3....]}
        self.__get_node_direct_childes()

    """
         Load ast file.
         1. Delete Using subtree.   删除using子树
         2. Add node layer information.  增加层次信息
         3. Delete extra space and other symbols in lines.  删除无用字符
    """
    def __load_ast_file(self):
        with open(self.__ast_file_path, 'r', encoding='utf-8') as ast_file:
            ast_lines = ast_file.readlines()
            self.__ast_lines = []
            # delete无用行
            using_flag = 0
            for key, line in enumerate(ast_lines):
                # ast文件当前行的内容
                self.__current_line = line
                # 设置当前行在抽象语法树中的层次数 --- 利用当前行内容前面的空格计算
                self.__get_node_layer()
                # 去掉无用行 -<<<NULL>>>
                if '-<<<NULL>>>' in line:
                    continue
                # 去掉无用行 -...
                if '-...' in line:
                    continue
                # 去掉Using子树
                if using_flag != 0 and self.__current_layer > using_flag:
                    continue
                else:  # 判断Using子树结束
                    using_flag = 0
                # 判断Using子树开始
                if 'Using' in line:
                    using_flag = self.__current_layer
                    continue
                self.__ast_lines.append(line)
        self.__ast_lines[0] = '-' + self.__ast_lines[0]

    """
        1. Get node layer
        2. Throughout space number to determine the layer.
    """
    def __get_node_layer(self):
        self.__current_layer = int((len(self.__current_line.split('-')[0]) + 1) / 2)

    """
        Get node memory.   获取当前节点的内存编号 ['0x38a4778', '0x38a34d8']
        self.__common_memories = {"memory":[appeared_in_line1,appeared_in_line2....]}
    """
    def __get_node_memory(self):
        memories = re.findall(r'0x[a-f0-9]+', self.__current_line)
        if not memories:
            self.__current_memory = ''
            return
        self.__current_memory = memories[0]   # 当前node的内存编号
        if memories[0] not in self.__common_memories.keys():
            self.__common_memories[memories[0]] = []
            self.__common_memories[memories[0]].append(self.__current_key)
        # 处理memories[1]
        if len(memories) == 2:
            if memories[1] not in self.__common_memories.keys():
                self.__common_memories[memories[1]] = []
            self.__common_memories[memories[1]].append(self.__current_key)

    """
          Get node one hot by ast type.
    """
    def __get_node_one_hot(self):
        for key, ast_type in enumerate(self.__ast_types):
            if self.__current_type == ast_type:
                self.__current_one_hot = key

    """
           Get node type
           1. Replace Operator ast node type by [+ - * / ]
    """
    def __get_node_type(self):
        ast_type = self.__current_line.split('-')[1].split(' ', 1)[0]
        # 使用具体的操作运算符号代替Operator操作运算符
        if 'Operator' in ast_type:
            m = re.findall('\'([=&!+>\-<,|%/*~^]{1,3})\'', self.__current_line)
            if len(m) != 0:
                ast_type = m[0]
        self.__current_type = ast_type

    # Get node direct childes via nodes array
    def __get_node_direct_childes(self):
        if not self.__ast_nodes:
            self.__parse_all_line()
        if not self.__node_direct_childes:
            self.__node_direct_childes = []

        if not self.__node_direct_parents:
            self.__node_direct_parents = []

        auxiliary_space = {0: 0}  # 辅助空间 {layer : key }

        for key in range(1, len(self.__ast_nodes)):
            node = self.__ast_nodes[key]
            auxiliary_space[node['layer']] = key
            self.__node_direct_childes.append([])
            self.__node_direct_childes[auxiliary_space[node['layer'] - 1]].append(key)

    # Parse all line via travel lines
    def __parse_all_line(self):
        # 初始化self.ast_nodes
        if not self.__ast_nodes:
            self.__ast_nodes = []
        for key, line in enumerate(self.__ast_lines):
            self.__current_key = key
            self.__current_line = line
            # 设置当前行（node）的node类型 -> self.__current_type
            # 使用具体的运算符号(+ - * % / ^ ! ~等)替换运算操作符号Operator
            self.__get_node_type()
            # 获取node类型ont-hot中1的编号  （ast_types.json中的编号）
            self.__get_node_one_hot()
            #  node的层编号
            self.__get_node_layer()
            # 设置 self.__common_memories
            self.__get_node_memory()
            self.__ast_nodes.append({
                "key": self.__current_key,
                "type": self.__current_type,
                "memory": self.__current_memory,
                "one_hot": self.__current_one_hot,
                "layer": self.__current_layer
            })

    # TODO create ast tree via ast nodes array and childes array
    def __create_ast_tree(self):
        pass

    # Create graph via nodes array and childes array and common memories array
    def get_source_graph(self):
        if not self.__graph:
            self.__graph = []

            # 1. get children && parent
            ast_children = {}
            for key in range(len(self.__node_direct_childes)):
                children = []
                for node_key in self.__node_direct_childes[key]:
                    children.append(node_key)
                if len(children) > 0:
                    ast_children[key] = children

            ast_parent = {}
            for key in range(len(self.__node_direct_childes)):
                for node_key in self.__node_direct_childes[key]:
                    ast_parent[node_key] = []
                    ast_parent[node_key].append(key)

            #  transfer  to  symbols
            for key in range(len(self.__node_direct_childes)):
                node = {}
                if key in ast_children.keys():
                    node['node'] = self.__ast_nodes[key-1]['type']
                    node['children'] = []
                    for node_key in ast_children[key]:
                        node['children'].append(self.__ast_nodes[node_key-1]['type'])
                else:
                    node['children'] = None

                if key not in ast_parent.keys():
                    node['parent'] = None
                else:
                    node['parent'] = self.__ast_nodes[ast_parent[key][0]-1]['type']

                self.__graph.append(node)
        return self.__graph

# ast = AST_tbcnn_2Graph("./ast/100070.ast")
# graph = ast.get_source_graph()
# print(graph)
# with open("./graph/100070.pkl", 'wb') as f:
#     pickle.dump(graph, f, -1)