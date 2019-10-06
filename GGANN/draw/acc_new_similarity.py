import matplotlib.pyplot as plt


# 10个相似算法分类精度画图

def avg(arr):
    res = 0
    for i in arr:
        res += i
    return res/(len(arr))


def show_avg(arr1, title1, arr2, title2, arr3, title3):
    print(title1+": "+str(avg(arr1)))
    print(title2+": "+str(avg(arr2)))
    print(title3+":"+str(avg(arr3)))


attention_valid_acc = [0.97407, 0.95890, 0.96725, 0.93910, 0.97093, 0.95211, 0.95738, 0.95733, 0.88765, 0.97386]
no_attention_valid_acc = [0.92534, 0.92285, 0.86437, 0.94815, 0.84654, 0.92327, 0.91456, 0.92949, 0.83091, 0.92451]
tbcnn_valid_acc = [0.85145, 0.83165, 0.86346, 0.85415, 0.81235, 0.85768, 0.88646, 0.85367, 0.81781, 0.86476]


attention_test_acc = [0.95469, 0.91504, 0.92322, 0.97343, 0.97564, 0.92099, 0.91457, 0.93654, 0.8881, 0.97457]
no_attention_test_acc = [0.93264, 0.90535, 0.86598, 0.94419, 0.85817, 0.91289, 0.91870, 0.93218, 0.84262, 0.92728]
tbcnn_test_acc = [0.86145, 0.84165, 0.87346, 0.86415, 0.83235, 0.85868, 0.87946, 0.85667, 0.83781, 0.86376]

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
figure = plt.figure(figsize=(10, 10), dpi=80)

# plt.subplot(221)
# plt.plot(x, attention_valid_acc, "b:", label='Similarity(GGANN)', linewidth=2, marker='o')
# plt.plot(x, no_attention_valid_acc, "r:", label='Similarity(GGNN)', linewidth=2, marker='^')
# plt.plot(x, tbcnn_valid_acc, "c:", label='Similarity(TBCNN)', linewidth=2, marker="*")
#
# plt.xlabel("Problem Tasks")
# plt.ylabel("Valid/Accuracies")
# plt.ylim(0.75, 1)
# plt.xlim(0, 12)
# plt.legend()
# plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")

plt.subplot(221)
plt.plot(x, attention_test_acc, "b:", label="Similarity(GGANN)", linewidth=2, marker='o')
plt.plot(x, no_attention_test_acc, "r:", label="Similarity(GGNN)", linewidth=2, marker='^')
plt.plot(x, tbcnn_test_acc, "c:", label='Similarity(TBCNN)', linewidth=2, marker="*")
#
plt.xlabel("Problem Tasks")
plt.ylabel("Test/Accuracies")
plt.ylim(0.75, 1)
plt.xlim(0, 12)
plt.legend()
plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")


show_avg(attention_valid_acc, "Atten_valid_Acc", no_attention_valid_acc, "no_Atten_valid_Acv",tbcnn_valid_acc,"tbcnn_test")
show_avg(attention_test_acc, "Atten_test_Acc", no_attention_test_acc, "no_Atten_test_Acv",tbcnn_test_acc,"tbcnn_test")
plt.show()
plt.savefig("acc_similarity")


#670800128