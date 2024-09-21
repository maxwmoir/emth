import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time

sizes = range(1, 1000) # For assign should be 5k but takes ages to run
times = []

for n in sizes:
    A = np.random.rand(n, n)
    print(n)
    t0 = time.time()
    B = la.inv(A)
    t1 = time.time()
    times.append(t1 - t0)

plt.plot(sizes, times)
plt.show()
