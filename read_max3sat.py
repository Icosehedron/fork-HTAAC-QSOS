import random
import numpy as np
import torch
import sympy as sp

### parse a given max3sat in cnf format as a 2D matrix
def parse_max3sat(cnf):
    max3sat = np.zeros((len(cnf),3), dtype = int)
    for i in range(len(cnf)):
        temp = cnf[i].strip().split(' ') 
        max3sat[i][0] = temp[0]
        max3sat[i][1] = temp[1]
        max3sat[i][2] = temp[2]
    return max3sat

### generate the weight matrices. Returns w_plus, w_minus (encoding degree-2 term), and W_plus, W_minus (encoding degree-4 terms)
def generate_W_mat(max3sat):
    n = get_max(max3sat)

    a = np.zeros((n+1, n+1))
    b = np.zeros((n+1, n+1))
    a_tilde = np.zeros(((n+1)**2, (n+1)**2))
    b_tilde = np.zeros(((n+1)**2, (n+1)**2))

    print('init!')

    for r in max3sat:
        i, j, k = np.abs(r[0]), np.abs(r[1]), np.abs(r[2])
        i_neg, j_neg, k_neg = np.sign(r[0]), np.sign(r[1]), np.sign(r[2])

        if i_neg > 0 and j_neg > 0 and k_neg > 0:
            b[0][i] += 1
            b[0][j] += 1
            b[0][k] += 1
            a[i][j] += 1
            a[i][k] += 1
            a[j][k] += 1
            b_tilde[j][(n+1)*i+k] += 1
        elif i_neg < 0 and j_neg > 0 and k_neg > 0:
            a[0][i] += 1
            b[0][j] += 1
            b[0][k] += 1
            b[i][j] += 1
            b[i][k] += 1
            a[j][k] += 1
            a_tilde[j][(n+1)*i+k] += 1
        elif i_neg > 0 and j_neg < 0 and k_neg > 0:
            b[0][i] += 1
            a[0][j] += 1
            b[0][k] += 1
            b[i][j] += 1
            a[i][k] += 1
            b[j][k] += 1
            a_tilde[j][(n+1)*i+k] += 1
        elif i_neg > 0 and j_neg > 0 and k_neg < 0:
            b[0][i] += 1
            b[0][j] += 1
            a[0][k] += 1
            a[i][j] += 1
            b[i][k] += 1
            b[j][k] += 1
            a_tilde[j][(n+1)*i+k] += 1
        elif i_neg < 0 and j_neg < 0 and k_neg > 0:
            a[0][i] += 1
            a[0][j] += 1
            b[0][k] += 1
            a[i][j] += 1
            b[i][k] += 1
            b[j][k] += 1
            b_tilde[j][(n+1)*i+k] += 1
        elif i_neg < 0 and j_neg > 0 and k_neg < 0:
            a[0][i] += 1
            b[0][j] += 1
            a[0][k] += 1
            b[i][j] += 1
            a[i][k] += 1
            b[j][k] += 1
            b_tilde[j][(n+1)*i+k] += 1
        elif i_neg > 0 and j_neg < 0 and k_neg < 0:
            b[0][i] += 1
            a[0][j] += 1
            a[0][k] += 1
            b[i][j] += 1
            b[i][k] += 1
            a[j][k] += 1
            b_tilde[j][(n+1)*i+k] += 1
        elif i_neg < 0 and j_neg < 0 and k_neg < 0:
            a[0][i] += 1
            a[0][j] += 1
            a[0][k] += 1
            a[i][j] += 1
            a[i][k] += 1
            a[j][k] += 1
            a_tilde[j][(n+1)*i+k] += 1
        else:
            print('error: ' + str(r))

    a = (a + np.transpose(a)) / 8
    b = (b + np.transpose(b)) / 8
    a_tilde = (a_tilde + np.transpose(a_tilde)) / 8
    b_tilde = (b_tilde + np.transpose(b_tilde)) / 8

    w_plus = a + b
    w_minus = a - b
    W_plus = a_tilde + b_tilde
    W_minus = a_tilde - b_tilde

    return w_plus, w_minus, W_plus, W_minus


### Returns the maximum variable in the max3sat instance 
def get_max(max3sat):
    m = 0
    for r in max3sat:
        temp = max(r)
        if temp > m:
            m = temp
    return m


### reads the problem instance from a given file
def read_max3sat(f_name):
    f = open(f_name)
    max3sat = ''
    lines = f.readlines()
    for line in lines:
        max3sat += line

    return max3sat.split('\n')[3:-1] ##Should be catered to the specific file type


def main():
    ### read the problem from file name
    file_path = './imported/s3v110c700-1.cnf'

    ### parse the max3sat problem to generate the W matrices
    max3sat = read_max3sat(file_path)
    max3sat = parse_max3sat(max3sat)

    print('parsed!')
    
    # variable occurrences
    v1 = max3sat[:,0]
    v2 = max3sat[:,1]
    v3 = max3sat[:,2]

    w_plus, w_minus, W_plus, W_minus = generate_W_mat(max3sat)
    
    ### save problem matrices
    torch.save(torch.from_numpy(w_plus), 'problem/w_plus.pt')
    torch.save(torch.from_numpy(w_minus), 'problem/w_minus.pt')
    torch.save(torch.from_numpy(W_plus), 'problem/W_plus_tilde.pt')
    torch.save(torch.from_numpy(W_minus), 'problem/W_minus_tilde.pt')
    torch.save(torch.from_numpy(np.absolute(v1)), 'problem/v1.pt')
    torch.save(torch.from_numpy(np.absolute(v2)), 'problem/v2.pt')
    torch.save(torch.from_numpy(np.absolute(v3)), 'problem/v3.pt')





if __name__ == '__main__':
    main()