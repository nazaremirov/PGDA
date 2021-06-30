import numpy as np

def Distributed_inv_approx(H):
    
    N = np.size(H, axis = 0)
    
    row_col_max = np.maximum(np.sum(H, axis = 1), np.sum(H, axis = 0))
    
    local_row_col_max = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            if H[i][j]!=0:
                local_row_col_max[i] = np.maximum(local_row_col_max[i], row_col_max[j])
            
    prec_matrix = np.diag(local_row_col_max)
    
    G = np.linalg.matrix_power(prec_matrix,-2)*H.transpose()
    
    return G


def PGDA_eigvector_approx(A, eig_value, max_iter):
    
    N = np.size(A, axis = 0)
    
    H = A - eig_value*np.identity(N)
    
    G = Distributed_inv_approx(H)
    
    x = np.random.rand(N)    
    
    for m in range(max_iter):
        x = np.dot(np.identity(N) - np.matmul(G, H), x)   
    return x
