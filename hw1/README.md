# Specification

### Content

- print_map.py: Q1, print the track on the map
- knn.py: Implement Q2 with brief annotation.
- result.txt: The result of knn.py
- README.md: Specification

### Used Package

- sklearn: for knn
- matplotlib: for printing the map

### Thoughts

- When I try to print the track on map, I find a useful package called *matplotlib*. I find that we can apply it to diverse ways, including meteorology and geology. This is the reason that I used it instead of Baidu Map.
- According to the requirement, I didn't do Min-Hash, though I thought it would be better to calculate the min-hash before the lsh. In addition, when the code is running, I notice that there is only one cpu running 100 percentage at the same time. So if I can make a multi-threading code, I'm sure it will be faster. Also, I guess this might be why when we do data-mining, we prefer to use GPU.

Wang Jiahui, 1452712
24 Mar, 2017
