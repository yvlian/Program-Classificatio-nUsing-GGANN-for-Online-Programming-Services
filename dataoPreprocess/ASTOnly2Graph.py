"""
   Transfer ast to graph
"""
import json
import re

# 仅仅包含ast的graph代码


class ASTOnly2Graph:
    __type_file_path = './data/ast_types.json'
    __ast_file_path = ''

    __index = None
    __ast_lines = None
    __ast_types = None
    __ast_nodes = None
    __node_direct_childes = None
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
        #  构造self.__functions:函数调用图边
        # self.__get_functions()
        # self.__get_operators()

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
        # return
        # for ast_type in self.__ast_types:
        #     if self.__current_type == ast_type:
        #         self.__current_one_hot.append(1)
        #     else:
        #         self.__current_one_hot.append(0)

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
        auxiliary_space = {0: 0}  # 辅助空间 {layer : key }
        for key in range(1, len(self.__ast_nodes)):
            node = self.__ast_nodes[key]
            auxiliary_space[node['layer']] = key
            self.__node_direct_childes.append([])
            self.__node_direct_childes[auxiliary_space[node['layer'] - 1]].append(key)

    # Get functions information array include CXXMethodDecl type and FunctionDecl type
    def __get_functions(self):
        if not self.__ast_nodes:
            self.__parse_all_line()
        if not self.__functions:
            self.__functions = {}
        for key in range(len(self.__ast_nodes)):
            node = self.__ast_nodes[key]
            if node['type'] == 'FunctionDecl' or node['type'] == 'CXXMethodDecl':
                function_start = node['key']  # function start
                function_end = 0
                function_memory = node['memory']
                function_params = []
                function_return = []
                for i in range(function_start + 1, len(self.__ast_nodes)):
                    # break until the end of function
                    if self.__ast_nodes[i]['layer'] <= node['layer']:
                        function_end = i - 1
                        break
                    if self.__ast_nodes[i]['type'] == 'ParmVarDecl':
                        function_params.append(self.__ast_nodes[i]['key'])
                    if self.__ast_nodes[i]['type'] == 'ReturnStmt':
                        function_return.append(self.__ast_nodes[i]['key'])
                self.__functions[function_start] = {
                    "memory": function_memory,
                    "start": function_start,
                    "end": function_end,
                    "params": function_params,
                    "return": function_return
                }

    # TODO get operators information array
    def __get_operators(self):
        pass

    # TODO get node subtree
    def get_node_subtree(self, node=0):
        pass

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
            # get nodes feature one hot
            nodes_feature = []
            for node in self.__ast_nodes:
                nodes_feature.append(node['one_hot'])

            # get edge
            edges = []

            # 1. get ast edge
            ast_edges = []
            print(self.__node_direct_childes)
            for key in range(len(self.__node_direct_childes)):
                for node_key in self.__node_direct_childes[key]:
                    ast_edges.append([key, 1, node_key])
            edges.extend(ast_edges)
            print(edges)
            self.__graph = {"nodes_feature": nodes_feature, "graph_edges": edges}
        return self.__graph

ast = ASTOnly2Graph("./ast/100070.ast")
with open("./graph/100070.graph", 'w', encoding='utf-8') as f:
        json.dump(ast.get_source_graph(), f)