import matplotlib.pyplot as plt

# 30个算法的分类精度画图

def avg(arr):
    res = 0
    for i in arr:
        res += i
    return res/(len(arr))


def show_avg(arr1, title1, arr2, title2, arr3, title3, arr4, title4):
    print(title1+": "+str(avg(arr1)))
    print(title2+": "+str(avg(arr2)))
    print(title3+": "+str(avg(arr3)))
    print(title4+ ": "+str(avg(arr4)))


attention_valid_acc =\
    [0.98584, 0.97831, 0.98646, 0.97353, 0.96725, 0.98451, 0.98416, 0.98070, 0.98247, 0.98283,
     0.98477, 0.98362, 0.98593, 0.97415, 0.98177, 0.98610, 0.96495, 0.95211, 0.98495, 0.98238,
     0.96742, 0.95990, 0.97973, 0.97707, 0.98734, 0.98238, 0.97318, 0.98637, 0.98097, 0.95751]
no_attention_valid_acc = \
    [0.97584, 0.96831, 0.97646, 0.96353, 0.95725, 0.97451, 0.96929, 0.93910, 0.97247, 0.97283,
     0.97477, 0.97362, 0.97593, 0.96415, 0.97177, 0.97610, 0.95495, 0.95211, 0.97495, 0.97238,
     0.96742, 0.95990, 0.96973, 0.96707, 0.97734, 0.97238, 0.96318, 0.97637, 0.95733, 0.95751]
no_attention_ast_valid_acc = \
    [0.96584, 0.96831, 0.97646, 0.96353, 0.95725, 0.97451, 0.96929, 0.93910, 0.97247, 0.97283,
     0.96477, 0.97362, 0.95593, 0.96415, 0.97177, 0.97610, 0.95495, 0.95211, 0.97495, 0.97238,
     0.96742, 0.95990, 0.96973, 0.96707, 0.97734, 0.97238, 0.96318, 0.97637, 0.95733, 0.95751]
ast_tbcnn_valid_acc = \
    [0.93697, 0.94389, 0.94172, 0.93725, 0.94144, 0.93267, 0.93860, 0.93325, 0.94012, 0.95144,
     0.94267, 0.94354, 0.95212, 0.95012, 0.95144, 0.94267, 0.93543, 0.94212, 0.9452,  0.94264,
     0.93567, 0.94244, 0.94568, 0.95021, 0.94876, 0.94587, 0.93980, 0.94876, 0.94321, 0.93125]

attention_test_acc = \
    [0.98013, 0.97455, 0.97956, 0.97329, 0.96322, 0.98233, 0.97941, 0.97031, 0.97399, 0.97627,
     0.97596, 0.98020, 0.97654, 0.96633, 0.97152, 0.97601, 0.96481, 0.95099, 0.98232, 0.98188,
     0.96841, 0.96058, 0.98020, 0.97624, 0.98287, 0.98152, 0.97457, 0.98233, 0.98245, 0.95750]
no_attention_test_acc = \
    [0.97583, 0.96695, 0.97196, 0.96519, 0.96322, 0.97733, 0.97541, 0.94343, 0.97399, 0.95627,
     0.97796, 0.97820, 0.97654, 0.97633, 0.97852, 0.97601, 0.96481, 0.95099, 0.98032, 0.97888,
     0.96841, 0.96058, 0.97620, 0.97624, 0.97887, 0.97752, 0.97457, 0.98733, 0.95654, 0.95742]
no_attention_ast_test_acc = \
    [0.97383, 0.96695, 0.97096, 0.96479, 0.96322, 0.97833, 0.97541, 0.93543, 0.97099, 0.95627,
     0.97496, 0.97020, 0.97654, 0.96633, 0.97052, 0.96951, 0.96481, 0.95099, 0.97732, 0.97188,
     0.96841, 0.96058, 0.97020, 0.97624, 0.97487, 0.97552, 0.97357, 0.97633, 0.95654, 0.95742]
ast_tbcnn_test_acc = \
    [0.93697, 0.94089, 0.94772, 0.93525, 0.94144, 0.92667, 0.93560, 0.93225, 0.94012, 0.94644,
     0.94067, 0.94014, 0.95212, 0.95012, 0.95144, 0.94167, 0.93343, 0.94212, 0.94524,  0.94164,
     0.93567, 0.94074, 0.94518, 0.95021, 0.94776, 0.94687, 0.93680, 0.94476, 0.94221, 0.93125]

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
     27, 28, 29, 30]
     #31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
figure = plt.figure(figsize=(10, 10), dpi=80)

plt.subplot(221)
plt.plot(x, attention_valid_acc, "b:", label='FDA_GGANN', linewidth=2, marker='o')
plt.plot(x, no_attention_valid_acc, "r:", label='FDA_GGNN', linewidth=2, marker='^')
plt.plot(x, no_attention_ast_valid_acc, "c:", label='AST_GGNN', linewidth=2, marker="v")
plt.plot(x, ast_tbcnn_valid_acc, "m:", label='AST_TBCNN', linewidth=2, marker="*")


plt.xlabel("Problem Tasks")
plt.ylabel("Valid/Accuracies")
plt.ylim(0.9, 1)
plt.xlim(0, 32)
plt.legend()
plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")

plt.subplot(222)
plt.plot(x, attention_test_acc, "b:", label="FDA_GGANN", linewidth=2, marker='o')
plt.plot(x, no_attention_test_acc, "r:", label="FDA_GGNN", linewidth=2, marker='^')
plt.plot(x, no_attention_ast_test_acc, "c:", label='AST_GGNN', linewidth=2, marker="v")
plt.plot(x, ast_tbcnn_test_acc, "m:", label='AST_TBCNN', linewidth=2, marker="*")

plt.xlabel("Problem Tasks")
plt.ylabel("Test/Accuracies")
plt.ylim(0.9, 1)
plt.xlim(0, 32)
plt.legend()
plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")

show_avg(attention_valid_acc, "attention_valid_acc", no_attention_ast_valid_acc, "no_attention_ast_valid_acc",
         no_attention_valid_acc, "no_attention_valid_acc", ast_tbcnn_valid_acc, "ast_tbcnn_valid_acc")

show_avg(attention_test_acc, "attention_test_acc", no_attention_ast_test_acc, "no_attention_ast_test_acc",
         no_attention_test_acc, "no_attention_test_acc", ast_tbcnn_test_acc, "ast_tbcnn_test_acc")


plt.show()
plt.savefig("acc")


