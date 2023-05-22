import torch
import numpy as np
import pickle
import time
from scipy.sparse import csr_matrix

def lsh_random_projection(A, c=4, m=64):
    """
    A: The input auxillary matrix. A torch tensor.
    c: The code carinality,
    m: Code length
    - c and m dictate the format and memory cost of the output compositional code
    - MUST USE CSR TENSOR
    return: Boolean Tensor
    """
    # A = A.to_dense() Lol, this is 10tb
    # indices = A._indices()
    # row_indices, column_indices = indices
    n, d = A.size()
    A = A.unsqueeze(1).to('cuda')
    # A.to('cuda')
    # WOW I could of done this with COO the entire time. Don't use A[i,:] use A[i]
    n_bits = int(m*np.log2(c))
    compositional_code = torch.zeros((n, n_bits), dtype=torch.bool)
    for i in range(n_bits):
        V = torch.rand(d).unsqueeze(1).to('cuda')  # Fix the dimension issue once. Create a random matrix?
        # V = torch.rand((n,d)).to('cuda') # Yeah, I will need to fix this to be a sparse matrix.
        # This is pain in the ass.
        # If I do then it will fit in memory, the same goes for U.

        # U = torch.empty((n,d)).to('cuda')
        U = torch.empty(n).to('cuda')
        for j in range(n):
             U[j] = torch.sparse.mm(A[j], V)  # There was a dimension issue!!!
        # U = torch.sparse.mm(A, V)

        t = torch.median(U)  # This returns a tensor, I need the value in the tensor.
        for j in range(n):
            if U[j] > t:
                compositional_code[j,i] = True

    return compositional_code

if __name__ == '__main__':

    # I need to get an adjacency matrix to test on.
    # Also figure out values for c and m
    data = pickle.load(open("sampledata.pkl", "rb"))
    A = data.get('full_adj')

    t0 = time.time()
    compositional_code = lsh_random_projection(A)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()

    print(f"{t1-t0:.3} Seconds")
    print(f"{(t1-t0)/60:.3} minutes")

    with open('compositional_code.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(compositional_code, file)
    file.close()

    with open('compositional_code_time.txt', 'w') as file:
        file.write(f"{t1-t0:.3} Seconds\n")
        file.write(f"{(t1-t0)/60:.3} minutes")

    file.close()


    print('Done')