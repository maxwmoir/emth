import numpy

def print_mat(A):

    for r in range(len(A)):
        for c in range(len(A[0])):
            print("{:.2f}".format(A[r][c]), end='\t')        
        print()

def markov_chain(A, p):
    # Calculates markov chain assuming visitors are p times more likely to swap than stay.
    B = numpy.zeros((len(A), len(A[0])))

    for r in range(len(B)):
        for c in range(len(B[0])):
            k = 0
            [k := k + A[x][c] for x in range(len(A))]
            x = 1 / (1 + k * p)

            if r == c:
                B[r][c] = x
            elif A[r][c] == 1:
                B[r][c] = x * p
        
    return B

def final_state(A, x_0):

    d, P = numpy.linalg.eig(A)

    # D_inf can be found by evaluating np.diag(d)**k as k -> inf. 
    D_inf = numpy.diag(d ** 1000) # This is close enough

    z = numpy.linalg.solve(P, x_0)
    x_inf = P @ D_inf @ z
    return x_inf


def print_percs(x_inf):
    rides = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    print(f"Final states (total est: {sum(x_inf):.0f}%):")
    for x in range(len(rides)):
        print("\t", rides[x], ': ', f"{x_inf[x]:.4}%")
    print()


def a(A, x_0):
    print("2.5x more likely to swap.")
    x_inf25 = final_state(markov_chain(A, 2.5), x_0)
    print_percs(x_inf25)

def b(A, x_0):
    x_inf5 = final_state(markov_chain(A, 5), x_0)
    x_inf1 = final_state(markov_chain(A, 1), x_0)
    print()
    print("5x more likely to swap.")
    print_percs(x_inf5)
    print("Same chance to swap or stay.")
    print_percs(x_inf1)


if __name__ == "__main__":

    A = numpy.array([
        #A  B  C  D  E  F  G  H  I  J
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # A
        [1, 0, 1, 1, 0, 0, 0, 0, 1, 1], # B
        [0, 1, 0, 1, 0, 1, 0, 0, 1, 0], # C
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0], # D
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # E
        [0, 0, 1, 0, 1, 0, 1, 1, 0, 0], # F
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0], # G
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1], # H
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 1], # I
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0]  # J
    ])

    x_0 = numpy.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    a(A, x_0)
    b(A, x_0)
# 
