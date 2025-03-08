import random
import numpy as np
import torch
import sympy as sp

### Randomly generate max3sat problem instance
def generate_max3sat(num_var, num_clauses):
    max3sat = ''
    for i in range(num_clauses):
        tmp, clause = generate_clause(num_var)
        while not tmp:
            tmp, clause = generate_clause(num_var)
        max3sat += ' '.join(clause) +  ' 0\n'
    return max3sat


### Randomly generate max3sat problem clause
def generate_clause(num_var):
    clause = []
    for _ in range(3):
        tmp = random.choice(range(1, num_var + 1, 1))
        if (str(tmp) in clause) or (str (-1 * tmp) in clause):
            return False, []
        else:
            clause.append(str(random.choice([1, -1]) * tmp))
    return True, clause


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
    a1_tilde = np.zeros((n+1, n+1))
    a2_tilde = np.zeros((n+1, n+1))
    b1_tilde = np.zeros((n+1, n+1))
    b2_tilde = np.zeros((n+1, n+1))
    a_tilde = np.zeros(((n+1)**2, (n+1)**2))
    b_tilde = np.zeros(((n+1)**2, (n+1)**2))

    def add_to_A():
        nonlocal a_tilde
        nonlocal a1_tilde
        nonlocal a2_tilde

        a_tilde = a_tilde + np.kron(a1_tilde, a2_tilde)

        a1_tilde = np.zeros((n+1, n+1))
        a2_tilde = np.zeros((n+1, n+1))

    def add_to_B():
        nonlocal b_tilde
        nonlocal b1_tilde
        nonlocal b2_tilde

        b_tilde = b_tilde + np.kron(b1_tilde, b2_tilde)

        b1_tilde = np.zeros((n+1, n+1))
        b2_tilde = np.zeros((n+1, n+1))

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

    return max3sat.split('\n')[4:-1] ##Should be catered to the specific file type


def main():
    ### select the size of the problem you want to make
    num_var = 20
    num_clauses = 600

    ### generate the problem and save it as a file
    name = 'v' + str(num_var) + 'c' + str(num_clauses)
    file_path = './gen_max3sat/' + name + '.cnf'
    max3sat = generate_max3sat(num_var, num_clauses)
    file = open(file_path, 'w')
    file.write(max3sat)
    file.close()

    ### parse the max3sat problem to generate the W matrices
    max3sat = read_max3sat(file_path)
    max3sat = parse_max3sat(max3sat)
    
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