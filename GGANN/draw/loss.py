import matplotlib.pyplot as plt
# loss变化曲线

attention_valid_loss =\
    [0.90566, 0.49735, 0.48792, 0.48560, 0.48496, 0.48404, 0.48422, 0.48265, 0.48153,
     0.47921, 0.47591, 0.47313, 0.47209, 0.47942, 0.47845, 0.47794, 0.47749, 0.47709,
     0.47518, 0.47790, 0.47534, 0.46456, 0.46164, 0.46061, 0.46086]
attention_test_loss = \
    [0.49771, 0.49437, 0.48637, 0.48584, 0.48487, 0.48409, 0.48448, 0.48515, 0.48062,
     0.47904, 0.47597, 0.47161, 0.48203, 0.48217, 0.48177, 0.47902, 0.47909, 0.47793,
     0.47950, 0.47874, 0.47435, 0.46323, 0.46287, 0.46488, 0.46122]

no_attention_valid_loss = \
    [0.80113, 0.49967, 0.49946, 0.49935, 0.50001, 0.50000, 0.50000, 0.50000, 0.50000,
     0.50050, 0.50000, 0.50000, 0.50000, 0.50000, 0.50000, 0.50050, 0.50000, 0.50000,
     0.50000, 0.50000, 0.50000, 0.50050, 0.50000, 0.50000, 0.50000]
no_attention_test_loss = \
    [0.50008, 0.49964, 0.49957, 0.50008, 0.50003, 0.50003, 0.50004, 0.50003, 0.50004,
     0.50003, 0.50004, 0.50003, 0.50004,  0.50003, 0.50004, 0.50003, 0.50004, 0.50002,
     0.50002, 0.50002, 0.50002, 0.50002, 0.50002, 0.50002, 0.50002]

no_attention_ast_valid_loss = \
    [0.85250, 0.49992, 0.49948, 0.49859, 0.49871, 0.49869, 0.49858, 0.49864, 0.49847,
     0.49850, 0.49838, 0.49906, 0.49863, 0.49853, 0.49846, 0.49892, 0.49871, 0.49845,
     0.49833, 0.49803, 0.49755, 0.50042, 0.50000, 0.50000, 0.50000]
no_attention_ast_test_loss=\
    [0.50008, 0.49971, 0.49864, 0.49877, 0.49859, 0.49858, 0.49889, 0.49845, 0.49859,
     0.49878, 0.49835, 0.49918, 0.49857, 0.49834, 0.49876, 0.49881, 0.49826, 0.49821,
     0.49809, 0.49786, 0.49834, 0.50002, 0.50002, 0.50002, 0.50002]


ast_tbcnn_valid_loss = \
    [0.98675, 0.966142, 0.946389, 0.91889, 0.89855, 0.83356, 0.79016, 0.759120, 0.73546,
     0.71735, 0.65515, 0.63528, 0.60797, 0.60484, 0.59021, 0.58145, 0.56312,  0.55819,
     0.54231, 0.53568, 0.54156, 0.53145, 0.53654, 0.54156, 0.53675]

ast_tbcnn_test_loss = \
    [0.544675, 0.556142, 0.5389, 0.548889, 0.547855, 0.5475, 0.56142, 0.53589, 0.54689,
     0.57635, 0.53515, 0.561528, 0.56797, 0.55484, 0.56021, 0.56145, 0.55312, 0.54819,
     0.54331, 0.53768, 0.54056, 0.53345, 0.53454, 0.54056, 0.53645]

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

figure = plt.figure(figsize=(10, 10), dpi=80)

plt.subplot(221)
plt.plot(x, attention_valid_loss, "b:", label='FDA_GGANN', linewidth=2, marker='o')
plt.plot(x, no_attention_valid_loss, "r:", label='FDA_GGNN', linewidth=2, marker='^')
plt.plot(x, no_attention_ast_valid_loss, "c:", label='AST_GGNN', linewidth=2, marker="v")
plt.plot(x, ast_tbcnn_valid_loss, "m:", label='AST_TBCNN', linewidth=2, marker="*")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0.4, 1)
plt.xlim(0, 26)
plt.legend()
plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")

# plt.subplot(222)
# plt.plot(x, attention_test_loss, "b:", label="FDA_GGANN", linewidth=2, marker='o')
# plt.plot(x, no_attention_test_loss, "r:", label="FDA_GGNN", linewidth=2, marker='^')
# plt.plot(x, no_attention_ast_test_loss, "c:", label='AST_GGNN', linewidth=2, marker="v")
# plt.plot(x, ast_tbcnn_test_loss, "m:", label='AST_TBCNN', linewidth=2, marker="*")
#
# plt.xlabel("Epoch")
# plt.ylabel("Test/Loss")
# plt.ylim(0.4, 1)
# plt.xlim(0, 26)
# plt.legend()
# plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")

plt.show()
plt.savefig("acc")

