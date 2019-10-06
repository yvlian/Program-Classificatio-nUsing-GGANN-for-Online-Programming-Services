import matplotlib.pyplot as plt


#  删除某一种FDA边后的分类精度画图

attention_valid_acc =\
    [0.98584, 0.97831, 0.98646, 0.97353, 0.96725, 0.98451, 0.98416, 0.98070, 0.98247,
     0.98283, 0.98477, 0.98362, 0.98593, 0.97415, 0.98177, 0.98610,
     0.96495, 0.95211,
     0.98495, 0.98238, 0.96742, 0.95990, 0.97973,
     0.97707, 0.98734, 0.98238, 0.97318, 0.98637, 0.98097, 0.95751]

attention_test_acc = \
    [0.98013, 0.97455, 0.97956, 0.97329, 0.96322, 0.98233, 0.97941, 0.97031, 0.97399, 0.97627,
     0.97596, 0.98020, 0.97654, 0.96633, 0.97152, 0.97601, 0.96481, 0.95099, 0.98232, 0.98188,
     0.96841, 0.96058, 0.98020, 0.97624, 0.98287, 0.98152, 0.97457, 0.98233, 0.98245, 0.95750]


FDA_valid_acc = [0.98362, 0.98407, 0.98354, 0.97637, 0.97105, 0.98566, 0.98894, 0.98823, 0.98584, 0.97831, 0.98646,
                0.98451,
                0.98416, 0.98070, 0.98247,
                0.98283, 0.96495, 0.95211,
                0.98495, 0.98238, 0.96742, 0.95990, 0.97973,
                0.97707, 0.98734, 0.98238, 0.97318, 0.98637, 0.98097, 0.95751]


no_ast_valid = [0.98362, 0.98407, 0.98354,  0.97637, 0.97105, 0.98566, 0.98894, 0.98823, 0.98584, 0.97831, 0.98646,
                0.98451,
                0.99000, 0.97725, 0.98247,
                0.98283, 0.96495, 0.95211,
                0.98495, 0.98238, 0.96742, 0.95990, 0.97973,
                0.97707, 0.98734, 0.98238, 0.97318, 0.98637, 0.95733, 0.95751]

no_operand_valid = [0.98362, 0.98407, 0.98354, 0.97637, 0.97105, 0.98566, 0.98894, 0.98823, 0.98584, 0.97831, 0.98646,
                    0.98451,
                    0.97929, 0.93910, 0.98247,
                    0.98283, 0.96495, 0.95211,
                    0.98495, 0.98238, 0.96742, 0.95990, 0.97973,
                    0.97707, 0.98734, 0.98238, 0.97318, 0.98637, 0.95733, 0.95751]

no_last_use_valid = [0.98362, 0.98407, 0.98354, 0.97637, 0.97105, 0.98566, 0.98894, 0.98823, 0.98584, 0.97831, 0.98646,
                     0.98451,
                     0.97929, 0.06090,
                     0.98247, 0.98283, 0.96495, 0.95211,
                     0.98495, 0.98238, 0.96742, 0.95990, 0.97973,
                     0.97707, 0.98734, 0.98238, 0.97318, 0.98637, 0.95733, 0.95751]

no_computed_from_valid = [0.98362, 0.98407, 0.98354, 0.97637, 0.97105, 0.98566, 0.98894, 0.97229, 0.98823, 0.97831, 0.96725,
                          0.98451,
                          0.97929, 0.05657, 0.98247,
                          0.98283, 0.98610, 0.96495,
                          0.95211, 0.98495, 0.98238, 0.96742, 0.95990,
                          0.97707, 0.98734, 0.98238, 0.97318, 0.98637, 0.95733, 0.95751]

no_return_to_valid = [0.98362, 0.98407, 0.98354, 0.97637, 0.97105, 0.98566, 0.98894, 0.98823, 0.98584, 0.97831, 0.98646,
                      0.98451,
                      0.97929, 0.93910, 0.98247,
                      0.98283, 0.98610, 0.96495,
                      0.95211, 0.98495, 0.98238, 0.96742, 0.95990,
                      0.97707, 0.98734, 0.98238, 0.97318, 0.98637, 0.95733, 0.95751]

no_formal_ArgName_valid = [0.98618, 0.98469, 0.98530, 0.97712, 0.96727, 0.98636, 0.98504, 0.98935, 0.98583, 0.97694, 0.98196,
                           0.98733,
                           0.97941, 0.94342, 0.98398,
                           0.98627, 0.96480, 0.95099,
                           0.98231, 0.98187, 0.96841, 0.96058, 0.98020,
                           0.97624, 0.98486, 0.98451, 0.97457, 0.98733, 0.95653, 0.95741]

no_call_function_valid = [0.98362, 0.98407, 0.98354, 0.97637, 0.97105, 0.98566, 0.98894, 0.98823, 0.98584, 0.97831, 0.98646,
                          0.98451,
                          0.97929, 0.97991, 0.98247,
                          0.98283, 0.96495, 0.98008,
                          0.98495, 0.98238, 0.96742, 0.95990, 0.97973,
                          0.97707, 0.98734, 0.98238, 0.97318, 0.98637, 0.95733, 0.95751]

FDA_test_acc = [0.98619, 0.98469, 0.98531, 0.97712, 0.96727, 0.98636, 0.98504, 0.98935, 0.98583,
               0.97695, 0.98196, 0.98733, 0.97941, 0.97818, 0.98399, 0.98627,
               0.98601, 0.96481,
               0.98099, 0.98232, 0.98188, 0.98841, 0.96058,
               0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.98645, 0.95750]

no_ast_test = [0.98619, 0.98469, 0.98531, 0.97712, 0.96727, 0.98636, 0.98504, 0.98935, 0.98583,
               0.97695, 0.98196, 0.98733, 0.98883, 0.98284, 0.98399, 0.98627,
               0.96481, 0.95099,
               0.98232, 0.98188, 0.96841, 0.96058, 0.98020,
               0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.95654, 0.95742]


no_operand_test = [0.98619, 0.98469, 0.98531, 0.97712, 0.96727, 0.98636, 0.98504, 0.98935, 0.98583,
                  0.97695, 0.98196, 0.98733, 0.97941, 0.94343, 0.98399, 0.98627,
                  0.96481, 0.95099,
                  0.98232, 0.98188, 0.96841, 0.96058, 0.98020,
                  0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.95654, 0.95742]


no_last_use_test = [0.98619, 0.98469, 0.98531, 0.97712, 0.96727, 0.98636, 0.98504, 0.98935, 0.98583,
                   0.97695, 0.98196, 0.98733, 0.97941, 0.05657, 0.98399, 0.98627,
                   0.96481, 0.95099,
                   0.98232, 0.98188, 0.96841, 0.96058, 0.98020,
                   0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.95654, 0.95742]

no_computed_from_test = [0.98619, 0.98469, 0.98531, 0.97712, 0.96727, 0.98636, 0.98504, 0.98935, 0.98583,
                       0.97695, 0.98196, 0.98733, 0.97941, 0.05657, 0.98399, 0.98627,
                       0.96481, 0.95099,
                       0.98232, 0.98188, 0.96841, 0.96058, 0.98020,
                       0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.95654, 0.95742]


no_return_to_test = [0.98619, 0.98469, 0.98531, 0.97712, 0.96727, 0.98636, 0.98504, 0.98935, 0.98583,
                   0.97695, 0.98196, 0.98733, 0.97941, 0.94343, 0.98399, 0.98627,
                   0.96481, 0.95099,
                   0.98232, 0.98188, 0.96841, 0.96058, 0.98020,
                   0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.95654, 0.95742]


no_formal_ArgName_test= [0.98618, 0.98469, 0.98530, 0.97712, 0.96727, 0.98636, 0.98504, 0.98935, 0.98583,
                         0.97694, 0.98196, 0.98733, 0.97941, 0.94342, 0.98398, 0.98627,
                         0.96480, 0.95099,
                         0.98231, 0.98187,  0.96841, 0.96058, 0.98020,
                         0.97624, 0.98486, 0.98451, 0.97457, 0.98733, 0.95653, 0.95741]

no_call_function_test= [0.98619, 0.98469, 0.98531, 0.98575, 0.97712,  0.98636, 0.98504, 0.98935, 0.98583,
                       0.97695, 0.98196, 0.98733, 0.97941, 0.97730, 0.98399, 0.98627,
                       0.98601, 0.96481,
                       0.98232, 0.98188, 0.96841, 0.96058, 0.97994,
                       0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.95654, 0.95742]


def count_average(scores):
    sum = 0
    for score in scores:
        sum += score
    size = len(scores)
    return sum/size

def print_average():
    print("no_ast_valid", count_average(no_ast_valid))
    print("no_operand_valid", count_average(no_operand_valid))
    print("no_last_use_valid", count_average(no_last_use_valid))
    print("no_computed_from_valid", count_average(no_computed_from_valid))
    print("no_return_to_valid", count_average(no_return_to_valid))
    print("no_formal_ArgName_valid ", count_average(no_formal_ArgName_valid))
    print("no_call_function_valid", count_average(no_call_function_valid))


    print("no_ast_test", count_average(no_ast_test))
    print("no_operand_test", count_average(no_operand_test))
    print("no_last_use_test", count_average(no_last_use_test))
    print("no_computed_from_test", count_average(no_computed_from_test))
    print("no_return_to_test", count_average(no_return_to_test))
    print("no_formal_ArgName_test ", count_average(no_formal_ArgName_test))
    print("no_call_function_test", count_average(no_call_function_test))

def get_x(max_x):
    x = []
    for i in range(1, max_x+1):
        x.append(i)
    return x


def draw():
    x = get_x(30)

    #plt.subplot(221)
    plt.plot(x, FDA_valid_acc, "r:", label='FDA', linewidth=2, marker='o')
    plt.plot(x, no_ast_valid, ":", color='#00FFFF', label='Ast', linewidth=1, marker='x')
    plt.plot(x, no_operand_valid, ":", color='#556B2F', label='Operand', linewidth=1, marker='^')
    plt.plot(x, no_call_function_valid, "--", color ='#DA70D6', label='Call', linewidth=1, marker="<")
    plt.plot(x, no_last_use_valid, ":", color='#800080', label='LastUse', linewidth=1, marker="*")
    plt.plot(x, no_formal_ArgName_valid, "k:", label='Formal', linewidth=1, marker="p")
    plt.plot(x, no_computed_from_valid, "b:", label='Computed', linewidth=1, marker="v")
    plt.plot(x, no_return_to_valid, "m:", label='Return', linewidth=1, marker="s")

    plt.xlabel("Problem Tasks")
    plt.ylabel("Accuracy")
    plt.ylim(0.9, 1)
    plt.xlim(0, 31)
    plt.legend()
    plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")

    # plt.subplot(222)
    # plt.plot(x, FDA_test_acc, "r:", label='FDA', linewidth=2, marker='o')
    # plt.plot(x, no_ast_test, "b:", label='no_Ast', linewidth=1, marker='o')
    # plt.plot(x, no_operand_test, "r:", label='no_Operand', linewidth=1, marker='^')
    # plt.plot(x, no_call_function_test, "p:", label='no_CallFunction', linewidth=1, marker="v")
    # plt.plot(x, no_last_use_test, "m:", label='no_LastUse', linewidth=1, marker="*")
    # plt.plot(x, no_formal_ArgName_test, "k:", label="no_FormalArgName", linewidth=1, marker="^")
    # plt.plot(x, no_computed_from_test, "y:", label='no_ComputedFrom', linewidth=1, marker="v")
    # plt.plot(x, no_return_to_test, "g:", label='no_ReturnsTo', linewidth=1, marker="*")
    # plt.xlabel("Embeddings Dimension")
    # plt.ylabel("Loss")
    # plt.xlabel("Algorithm")
    # plt.ylabel("Accuracy")
    # plt.ylim(0.9, 1)
    # plt.xlim(0, 31)
    # plt.legend()
    # plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")

    plt.show()
    plt.savefig("deleteEdge")


if __name__ == '__main__':
    draw()

