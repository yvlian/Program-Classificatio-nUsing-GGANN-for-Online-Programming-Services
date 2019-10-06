import matplotlib.pyplot as plt

# 18个算法的分类精度画图

def avg(arr):
    res = 0
    for i in arr:
        res += i
    return res/(len(arr))


def show_avg(arr1, title1, arr2, title2, arr3, title3):
    print(title1+": "+str(avg(arr1)))
    print(title2+": "+str(avg(arr2)))
    print(title3+": "+str(avg(arr3)))

attention_valid_acc =\
    [0.95402, 0.94419, 0.95442, 0.95446, 0.95481, 0.94780, 0.95043, 0.95292, 0.96505, 0.95833,
     0.96862, 0.96285, 0.95442, 0.95434, 0.95581, 0.95780, 0.94043, 0.95292]
no_attention_valid_acc = \
    [0.95383, 0.93862, 0.95315, 0.95442, 0.94434, 0.93081, 0.92780, 0.94043, 0.95292, 0.95496, 0.96431,
     0.95895, 0.95315, 0.94001, 0.95501, 0.95976, 0.93000, 0.93987]
no_attention_ast_valid_acc = \
    [0.94984, 0.95081, 0.95292, 0.94315, 0.95502, 0.94434, 0.96480, 0.96495, 0.95608, 0.94043,
     0.91862, 0.94465, 0.93186, 0.95104, 0.95442, 0.97578, 0.92780, 0.95021]

attention_test_acc = \
    [0.96041, 0.94391, 0.94991, 0.95666, 0.94731, 0.93442, 0.95741, 0.95986, 0.95991, 0.95771,
     0.96476, 0.95951, 0.96071, 0.95591, 0.95966, 0.96101, 0.93281, 0.94631]
no_attention_test_acc = \
    [0.95886, 0.91647, 0.94026, 0.95016, 0.93731, 0.94421, 0.92741, 0.93986, 0.94991, 0.95611,
     0.96476, 0.95941, 0.96071, 0.95387, 0.95868, 0.96012, 0.93157, 0.94531]
no_attention_ast_test_acc = \
    [0.96086, 0.94407, 0.94977, 0.96056, 0.95756, 0.93717, 0.96476, 0.96266, 0.95951, 0.93987,
     0.91633, 0.94602, 0.93267, 0.95936, 0.95652, 0.97316, 0.92728, 0.94577]

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
figure = plt.figure(figsize=(10, 10), dpi=80)

plt.subplot(221)
plt.plot(x, attention_valid_acc, "b--", label='valid_attention', linewidth=2, marker='o')
#plt.plot(x, no_attention_valid_acc, "r:", label='valid', linewidth=2, marker='^')
plt.plot(x, no_attention_ast_valid_acc, "m-.", label='valid_ast_only', linewidth=2, marker="v")

plt.xlabel("Task")
plt.ylabel("Accuracies")
plt.ylim(0.8, 1)
plt.xlim(0, 23)
plt.legend()
plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")

plt.subplot(222)
plt.plot(x, attention_test_acc, "b--", label="test_attention", linewidth=2, marker='o')
#plt.plot(x, no_attention_test_acc, "r:", label="test", linewidth=2, marker='^')
plt.plot(x, no_attention_ast_test_acc, "m-.", label='test_ast_only', linewidth=2, marker="v")

plt.xlabel("Task")
plt.ylabel("Accuracies")
plt.ylim(0.8, 1)
plt.xlim(0, 23)
plt.legend()
plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")

show_avg(attention_valid_acc, "attention_valid_acc", no_attention_ast_valid_acc, "no_attention_ast_valid_acc" ,
         no_attention_valid_acc, "no_attention_valid_acc")

show_avg(attention_test_acc, "attention_test_acc", no_attention_ast_test_acc, "no_attention_ast_test_acc" ,
         no_attention_test_acc, "no_attention_test_acc")

plt.show()
plt.savefig("acc")


