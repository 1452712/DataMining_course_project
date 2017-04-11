1. 可直接import numpy包和pandas包

```
import numpy as np
import pandas as pd

from lshash import LSHash
```

2. 元数据包括四列：

	a. tid轨迹编号

	b. 轨迹时间

	c. 轨迹坐标x(GPS->utm码, 建议20m*20m切格子)

	d. 轨迹坐标y

	**记得备份**

3. 操作

	- 先运行示例代码;

	- 使用`data = pandas.read_csv("FILENAME")`读入,
`data_new = pandas.DataFrame(data)`转换数据格式
`data_array = list(data_new)`转数组

	- 注意指定列名(否则会少一行)
`data = pandas.read_csv("FILENAME".header=None)`

	- 调用lshash
`lshash=LSHash(6,8)` //具体参数含义查api
`lshash.index([1,2,3])` //核心难点
`lshash.query()`

	- 调用knn,参见[http://scikit-learn.org]
cluster, nearest neighbors, example, API.(fit在有label的情况下;先尝试不预设聚成几类;建议不要改n_jobs)
输出label.

	- 作业:
读入数据`data = pandas.read_csv("FILENAME")`,将时间转换为unix时间戳`import time`(具体方法自己查询),将每一条轨迹分开(目前以坐标存储,可考虑变成01向量矩阵);
调用lshash(安装可以参考[http://pypi.python.org/pypi/lshash]);
knn聚类(建议多运行几次,选择最好结果);
提交源代码和结果.(特殊的处理方法请注明,写名字&学号,**ddl 03-25**)

	- 尝试将utm投射到地图上(python有地图编码),GPS数据问助教

	- linux请远程ssh访问数据


p.s. 可尝试不同参数的组合，得出最佳结果。

p.p.s. 推荐使用Python2.7

p.p.p.s. 建议预习SVM, 朴素贝叶斯, k-means
