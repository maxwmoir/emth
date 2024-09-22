import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import random
def print_mat(A):
    for r in range(len(A)):
        for c in range(len(A[0])):
            print("{:.2f}".format(A[r][c]), end='\t')        
        print()

def gershgorin(A):
    ls = []
    for r in range(len(A)):
        d  = 0 + 0j
        ri = 0 + 0j

        for c in range(len(A[0])):
            if r == c:
                d += (A[r][c] )
            else:
                # ri += abs(A[r][c].real) + (abs(A[r][c].imag))*1j
                ri += abs(A[r][c])
        ls.append((d, ri))
    return ls

def plot_discs(A, lsr, lsc, eigens = None, ests = None):
    fig, ax = plt.subplots()
    ax.grid()

    for i in range(A.shape[0]):
        ax.add_patch(plt.Circle((lsr[i][0].real, lsr[i][0].imag), lsr[i][1].real, color = 'blue', alpha = .2))
        ax.add_patch(plt.Circle((lsc[i][0].real, lsc[i][0].imag), lsc[i][1].real, color = 'red', alpha = .2))
    
    if eigens is not None:
        # print(eigens)
        for eigen in eigens:
            ax.plot(eigen.real, eigen.imag, 'ro')
    
    if ests is not None:
        # print(ests)
        for est in ests:
            ax.plot(est.real, est.imag, 'kx')

    ax.axis('equal')
    plt.show()

def R(y, x):
    return np.dot(y, x) / np.dot(x, x)

def IPM(A, x, k):
    (PT, L, U) = la.lu(A)
    
    for n in range(k):

        lastx = x
        z = np.linalg.solve(L, PT @ x)
        y = np.linalg.solve(U, z)
        x = y / np.linalg.norm(y, float('inf'))
        m = R(y, lastx)
        # print(n, y, x, m)
        
    return lastx, y, m

def shifted_IPM(A, guesses, x):
    ests = []
    for q in guesses:
        B = A - np.identity(len(A)) * q


        (nx, y , m) = IPM(B, x, 50)


        l = q + 1/ m
        ests.append(l)
    return ests

def a(A):
    row_discs = gershgorin(A)
    col_discs = gershgorin(np.transpose(A))

    guesses = [c[0] for c in col_discs]
    for x in range(len(row_discs)):
        guesses[x] = row_discs[x][0] + random.randint(-1, 1) + random.randint(-1, 1)*1j


    x = np.array([1, 0, 1, 1])
    ests = shifted_IPM(A, guesses, x)
    print()
    print(ests)
    eigs = [-3.609220290413334 - 0.984991312241698j, 1.981433787001914+0.48455676707267j, 4.785892579025826-0.184371722542448j, 5.341893924385594+0.184806267711476j]
    # eigs = np.linalg.eig(A)
    plot_discs(A, row_discs, col_discs, ests, eigs)




if __name__ == "__main__":
    A = np.array([
        [5 + 0j, 0.1j, -1 + 0.5j, 0.1j],
        [0.1, 2 + 0.5j, 0.1 + 0.1j, 0],
        [-.1j, 0.1, 5, 2],
        [0, 1, .5, - 3.5 - 1j]
    ])


    a(A)

