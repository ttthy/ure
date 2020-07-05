# from https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

import numpy as np
import numpy.random as npr

def alias_setup(probs):
    K       = len(probs)
    q       = np.zeros(K)
    J       = np.zeros(K, dtype=np.int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger  = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    K  = len(J)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand()*K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]


if __name__ == '__main__':
    K = 100
    N = 1000000

    # Get a random probability vector.
    import time

    probs = npr.dirichlet(np.ones(K), 1).ravel()
    print(probs, np.sum(probs))


    # Construct the table.
    s = time.time()
    J, q = alias_setup(probs)
    print('setup %f' % (time.time() - s))

    # Generate variates.
    s = time.time()
    X = np.zeros(K)
    for nn in range(N):
        X[alias_draw(J, q)] += 1
    X = X / N
    print(X)
