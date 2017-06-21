1.hw4_p2_00是我刚刚写的处理数据的代码
a.包括函数processOriginDataForP2用于处理数据，其中cell_length是一个自定义格点边长，默认0.0005约为50米。每次运行这个函数输出的格点编号都是现生的，所以不同输出文件之间的格点编号是独立的，没有对应关系。
参数method默认为"ChangeBases"，指每当基站号更换则视为新路径，我输出了my_final_2g_tr.csv，my_final_4g_tr.csv
如果该method设置为"Distance"，则当两个相邻点距离超过path_distance时视为新路径，我输出了my_final_2g_tr_2.csv，my_final_4g_tr_2.csv
注：用前者，就是我们一开始商量的方法，效果很差，见图input_2g_path_ChangeBases_1.png，几乎都在同一个点上，你可能没办法用；而后者可以看到路径，如input_2g_path_1.png，但一条路径上各个点对应的基站是变化的，你也许可以把基站编号也当做训练时的输入值？
你可以自己改method，cell_length，path_distance参数重新输出，跑一次2g大约30-50秒，4g大约10秒

b.drawPathForCsv用于检查输出路径，可以输出指定文件的路径图像。
drawRange为要画图的pathId数组，为None时输出所有路径，但不建议，因为可能太多。

2.hw4_p1_00和hw4_p2_01是我第一题的代码，后者用来画图。