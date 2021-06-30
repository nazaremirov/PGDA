import numpy as np
#from numpy.linalg import matrix_power

def Distributed_inv_approx(H):
    
    N = np.size(H,axis = 0)
    
    row_col_max = np.maximum(np.sum(H, axis = 1),np.sum(H, axis = 0))
    
    local_row_col_max = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            if H[i][j]!=0:
                local_row_col_max[i] = np.maximum(local_row_col_max[i], row_col_max[j])
            
    prec_matrix = np.diag(local_row_col_max)
    
    G = np.linalg.matrix_power(prec_matrix,-2)*H.transpose()
    
    return G

def Distributed_sym_inv_approx(H):
    
    row_col_max = np.sum(H, axis=1)
    
    G = np.linalg.matrix_power(np.diag(row_col_max),-1)
    
    return G


def PGDA(H,y,max_iter):
    
    if np.allclose(H, H.T, rtol = 1e-05, atol = 1e-08):
        G = Distributed_sym_inv_approx(H)
    else:
        G = Distributed_inv_approx(H)
    
    N = np.size(H, axis = 0)
    
    x = np.zeros(N)
    e = np.zeros(N)
    
    for m in range(max_iter):
        e = np.dot(H,x) - y 
        x = x - np.dot(G,e)
    
    return x
