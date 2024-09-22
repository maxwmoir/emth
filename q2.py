
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time

sizes = list(range(2, 500, 100)) 
times = []
theoretical_times = []

for n in sizes:
    A = np.random.rand(n, n)
    trial_times = []

    for _ in range(5):
        t0 = time.time()
        B = la.inv(A)
        t1 = time.time()
        trial_times.append(t1 - t0)

    times.append(sum(trial_times) / len(trial_times))
    theoretical_times.append((33.8 * 10**-12) * (2/3 * n**3 + 2*n**2))

plt.plot(sizes, times, 'b')
plt.plot(sizes, theoretical_times, 'go')
plt.show()
