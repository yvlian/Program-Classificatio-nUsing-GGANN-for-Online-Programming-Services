<h5>论文：Program Classification Using Gated Graph Attention Neural Network for Online Programming Services
Ming</h5>
<h5>创新点:</h5>
这项工作属于源代码挖掘领域。我们做的就是基于代码进行分类，因为数据集的原因，做的是基于acm代码进行分类，但是我们这项工作也可以进行拓展，例如基于在线编程服务分析程序作者的编程能力，进而提出一些合适的学习建议等等。<br>
1）	模型既可以学到节点的向量表示，也可以学习到图的向量表示。同时，包含2层的注意力机制，节点与邻居节点、节点与图。<br>
2）	有代码构图的方式创新，AST为抽象语法树，并在AST的节点上加入数据流图(DFG)和函数调用图(FCG)的边，图中共有7种类不同类型的边。<br>
3）	设代码构图后图中节点有n个，图的邻接矩阵的不是简单的n*n维，而是n*n*d维度，d维描述了边的特征。<br>
<h5>实验：源代码分类</h5>
<h5> acm数据集：包含3个表，Problem、Solution、Source_code</h5>
<h5>处理步骤：</h5>
1)使用clang获取代码对应的AST文件<br>
2)将ast文件处理成graph（包括7种类型的边）<br>
3)得到数据集：<br>
{'target':graph对应的题目编号, #one_hot<br>
'graph':(源节点编号，边类型编号，目标节点编号),<br>
'node_features':节点编号与节点类型的映射关系 #one_hot<br>
4)运行run_all.py<br>


