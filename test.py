import math
from itertools import islice


dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7:7, 8:8, 9:9 , 10:"sho"}
dict_size = len(dict)
thread_count = 4
i = 0
while i < dict_size:
    chunk_size = (dict_size - i) // thread_count
    thread_count -= 1
    print(list(islice(dict, i, i + chunk_size)))
    i += chunk_size
