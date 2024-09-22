
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time


sizes = list(range(2, 5000, 100)) # For assign should be 5k but takes ages to run
sizes.append(5000)
times = []
theo = []


for n in sizes:
    A = np.random.rand(n, n)
    print(n)
    t0 = time.time()
    B = la.inv(A)
    t1 = time.time()
    times.append(t1 - t0)
    theo.append((59.7 * 10**-12) * (2/3 * n**3 + 2*n**2))


plt.plot(sizes, times, 'b')
plt.plot(sizes, theo, 'go')
plt.show()
