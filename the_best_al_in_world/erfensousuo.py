import numpy as np
import time

start_time = time.time()
x = np.array([[1,2,3],[5,7,8],[1,0,2],[4,7,8],[4,5,6],[7,8,9]])
flag =[-1]
def erfen(arr, l, r, target):

    if l == (r-1):
        if list(arr[l]) == list(target):
            flag[0] = l

    else:
        mid = (l + r) // 2
        if flag[0] == -1:

            erfen(arr, l, mid, target)
        if flag[0] == -1:

            erfen(arr, mid, r, target)

erfen(x, 0, len(x), [4, 5, 6])

end_time = time.time()
print((end_time - start_time)*1000)
print(flag)