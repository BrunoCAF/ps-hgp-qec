import pym4ri
import numpy as np
import numpy.random as npr

num_tests = 10000
for _ in range(num_tests):
    verbose = False

    shape = [(30, 50), (100, 100), (150, 200), (200, 150)][npr.choice(4, p=[0.25, 0.25, 0.25, 0.25])]
    task = npr.choice(['gen2chk', 'chk2gen'])
    in_matrix = npr.randint(2, size=shape, dtype=np.bool_)
    
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
        
        verification = pym4ri.gf2_mul(out_matrix, in_matrix.T)

        if verbose:
            print(verification)
            input()

        assert not verification.any()
    else:
        out_matrix = pym4ri.chk2gen(in_matrix)

        if verbose:
            print(out_matrix)
            input()
        
        verification = pym4ri.gf2_mul(in_matrix, out_matrix)

        if verbose:
            print(verification)
            input()

        assert not verification.any()

    if verbose:
        print("Success!")
        input()

