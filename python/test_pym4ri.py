import pym4ri
import numpy as np
import numpy.random as npr

num_tests = 1000
for _ in range(num_tests):
    verbose = False

    shape = [(3, 5), (10, 10), (15, 20), (20, 15)][npr.choice(4, p=[0.25, 0.25, 0.25, 0.25])]
    task = npr.choice(['gen2chk', 'chk2gen'])
    in_matrix = npr.randint(2, size=shape, dtype=np.uint8)
    
    if verbose:
        print(shape)
        print(task)
        print(in_matrix)
        input()

    rank = pym4ri.rank(in_matrix)
    
    if verbose:
        print(rank)
        input()
    
    if task == 'gen2chk':      
        out_matrix = pym4ri.gen2chk(in_matrix)

        if verbose:
            print(out_matrix)
            input()

        assert not ((out_matrix @ in_matrix.T) % 2).any()
    else:
        out_matrix = pym4ri.chk2gen(in_matrix)

        if verbose:
            print(out_matrix)
            input()

        assert not ((in_matrix @ out_matrix) % 2).any()

    if verbose:
        print("Success!")
        input()

