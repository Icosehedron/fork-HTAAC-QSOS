import tlquantum as tlq
import torch
import itertools
import os
import matplotlib.pyplot as plt


torch.manual_seed(0) # will make the initialization deterministic (same random every time), comment out for random random

#device = 'cuda:0' # uncomment if you want to run on a GPU
device = 'cpu' # comment if you if you want to run on a GPU
dtype = torch.complex64

instance = 's3v70c700-1.cnf'


### hyperparameters simulation
nepochs = 100 # number of epochs per simulation, you can play with this
reps = 1 # number of repititions of experiment (full runs). At first, you probably just want 1, but crank it up to more reps to compare an ensemble of random initializations and get general understanding

### hyperparamters optimizer
lr = 0.01 # learning rate

### circuit hyperparameters
ncontraq = 1 # number of qubits in core, not relevant for matrices so don't worry about this for qiskit
ncontral = 5 # number of layers in core, not relevant for matrices so don't worry about this for qiskit
gate_rep = 100 # how many time to repeat the gate patern: Rotation Y, CZ-even, Rotation Y, CZ-odd, for more info please see below. You can play with this.

### choose your graph!
data_set = 'skew' # name for saving files
name = './problem/' # graph name to load from file
nqubits = 5 # number of qubits. Depends on nterms
nterms =  21 # number of variables in the problem instance. We can solve 2**nqubits variables or fewer, so nterms < 2**nqubits

### graph hyperparameters
alpha = 0.01 # unitary phase. You can probably leave me alone, but you could try making me ~2x-5x smaller or bigger
coeff_base = 300 # size of coefficient. Bigger makes us enforce the constraints harder. You will want to tune this.
reg = 10.0 # regularizes the strength of population balancing term. Bigger makes us regulate less. You will want to tune this.

### build trivial input state as a Matrix Product State
state = tlq.spins_to_tt_state([0 for i in range(nqubits)], device=device, dtype=dtype)
state = tlq.qubits_contract(state, ncontraq)


### build the Pauli operators for the constraints
pauli_z = torch.tensor([[1,0],[0,-1]]).to(device)
iden = torch.tensor([[1,0],[0,1]]).to(device)

### build the two unitary matrices
### W matrices are generators of the objective function unitaries encoding the SDP.
w_plus = torch.load(name+'w_plus.pt').to(torch.float).to(device)
w_minus = torch.load(name+'w_minus.pt').to(torch.float).to(device)
W_plus = torch.load(name+'W_plus_tilde.pt').to(torch.float).to(device)
W_minus = torch.load(name+'W_minus_tilde.pt').to(torch.float).to(device)

# U encodes degree-2 terms in the objective function 
U = torch.matrix_exp(1j*alpha*(w_minus)).to(device)
U = torch.nn.functional.pad(input=U, pad=(0, 2**nqubits - len(U), 0, 2**nqubits - len(U)), mode='constant', value=0).to(torch.complex64)

# U_tilde encodes degree-4 terms in the objective function
U_tilde = torch.matrix_exp(1j*alpha*(W_minus)).to(device)
U_tilde = torch.nn.functional.pad(input=U_tilde, pad=(0, 2**(nqubits*2) - len(U_tilde), 0, 2**(nqubits*2) - len(U_tilde)), 
                                  mode='constant', value=0).to(torch.complex64)


### generate all 1 and 2-body constraint operators in matrix form (rather than tensor)
### Let's take 2 qubits as an example, then pauli_obs will be a list that contains: Z0*I1, I0*Z1, and Z0*Z1 after this block of code
### A 3-qubit example, pauli_obs will contain: Z0*I1*I2, I0*Z1*I2, I0*I2*Z2, Z0*Z1*I2, Z0*I1*Z2, I0*Z1*Z2
pauli_obs = []
for i in range(1,3):
    for gate_indices in list(itertools.combinations(list(range(nqubits)), i)):
        op = torch.tensor(1., dtype=torch.complex64, device=device)
        for ind in range(nqubits):
            if ind in gate_indices:
                op = torch.kron(op, pauli_z)
            if ind not in gate_indices:
                op = torch.kron(op, iden)
        pauli_obs.append(op)
nconstraints = len(pauli_obs)
coeff = coeff_base*alpha/(nconstraints)
pauli_obs = pauli_obs[:1]

### V is generator of the population balancing unitary U2. This is the matrix that helps the approximate constraints stored in pauli_obs work better (sometimes they need some help)
vertices1 = torch.load(name+'v1.pt').to(torch.int64).to(device)
vertices2 = torch.load(name+'v2.pt').to(torch.int64).to(device)
vertices3 = torch.load(name+'v3.pt').to(torch.int64).to(device)
bins = torch.bincount(torch.cat((torch.cat((vertices1, vertices2)), vertices3)))
bins[0] = nterms
max_bins = torch.max(bins)
V = torch.zeros((2**nqubits, 2**nqubits))
for i in range(2**nqubits):
    if i < nterms:
        V[i,i] = (-(max_bins-bins[i])/reg)
    else:
        V[i,i] = (-max_bins/reg)
U2 = torch.matrix_exp(1j*alpha*V).to(device)

cut_array = torch.zeros((reps, nepochs)) # to store cut values (how well we solve MaxCut)
proper_loss_vec = torch.zeros((reps, nepochs)) # to store the loss values (the cut value is a rounded and weighted variation of this)

for rep in range(reps):
    ### build the unitary gates layer by layer
    print('___________________________')
    # first we define the control-z gates. As these are constant we define them only once and reuse the same ones.
    CZ0 = tlq.BinaryGatesUnitary(nqubits, ncontraq, tlq.cz(device=device, dtype=dtype), 0) # control-z gates where even qubits control odd qubits (first qubit is even here because zero)
    CZ1 = tlq.BinaryGatesUnitary(nqubits, ncontraq, tlq.cz(device=device, dtype=dtype), 1) # control-z gates where odd qubits control even qubits (first qubit is even here because zero)
    unitaries = []
    # now we build the circuits by adding the pattern: Rotation Y, CZ-even, Rotation Y, CZ-odd
    # tlq.UnaryGatesUnitary are the Rotation Ys and they have to be called every time like this because they contain trainable parameters, so each Rotation Y must be unique
    # also, please note that the order things are added is the order that they are applied. So if we have gates A, B, C, D and we write unitaries = [A, B, C, D], then we
    # will apply them to our quantum state |0> as output_state = DCBA|0>
    for r in range(gate_rep):
        unitaries += [tlq.UnaryGatesUnitary(nqubits, ncontraq, device=device, dtype=dtype), CZ0, tlq.UnaryGatesUnitary(nqubits, ncontraq, device=device, dtype=dtype), CZ1]

    ### normally we do all of our work with the TTCircuit class, but here we just use it go get the matrices that will represent our variational quantum circuit
    circuit = tlq.TTCircuit(unitaries, ncontraq, ncontral)

    opt = torch.optim.Adam(circuit.parameters(), lr=lr, amsgrad=True) # basic ADAM AMSGrad optimizer to manage our gradient descent
    
    all_loss = []
    all_rounded_sol = []
    all_unrounded_sol = []
    
    for epoch in range(nepochs):

        ### output state is the action of the variational quantum circuit on the input state
        output_state = circuit.to_ket(state).to(torch.complex64)

        ### calculate the inner product of each constraint
        ### These are just the inner products of our Pauli constraints. So for the 2-qubit constraint Z0*I1, we here build the operator <psi|Z0I1|psi> where |psi> = U_variational |0>
        ### and then we square it to be <psi|Z0I1|psi>**2, so that we can add it to punish the loss function as coeff*<psi|Z0I1|psi>**2. The goal of this is to push us towards <psi|Z0I1|psi> = 0.
        constraints = torch.zeros((len(pauli_obs),)).to(device)
        for i in range(len(pauli_obs)):
            constraints[i] = torch.real(torch.matmul(torch.transpose(output_state, 0, 1), torch.matmul(pauli_obs[i], output_state)))
        constraints = torch.sum(constraints**2)
        
        ### loss from the objective function and population balancing unitaries that we built above
        loss_proper = torch.imag(torch.matmul(torch.transpose(output_state, 0, 1), torch.matmul(U, output_state))) \
                + torch.imag(torch.matmul(torch.transpose(torch.kron(output_state, output_state), 0, 1), torch.matmul(U_tilde, torch.kron(output_state, output_state)))) \
                + torch.imag(torch.matmul(torch.transpose(output_state, 0, 1), torch.matmul(U2, output_state)))
        proper_loss_vec[rep, epoch] = loss_proper.detach().data

        ### add the constraint loss for full backprop
        loss = loss_proper + coeff*constraints

        with torch.no_grad():
            # calculating the unrounded problem solution
            out = torch.real(output_state[0:nterms])
            one_vector = torch.ones((nterms, 1))
            unrounded_obj = (torch.matmul(torch.transpose(one_vector, 0, 1), torch.matmul(w_plus, one_vector)) \
                - torch.matmul(torch.transpose(out, 0, 1), torch.matmul(w_minus, out)) \
                + torch.matmul(torch.transpose(torch.kron(one_vector, one_vector), 0, 1), torch.matmul(W_plus, torch.kron(one_vector, one_vector))) \
                - torch.matmul(torch.transpose(torch.kron(out, out), 0, 1), torch.matmul(W_minus, torch.kron(out, out))))/2
            
            # calculating the rounded problem solution
            rounded_output_state = torch.sign(torch.real(output_state[0:nterms]))
            obj = (torch.matmul(torch.transpose(one_vector, 0, 1), torch.matmul(w_plus, one_vector)) \
                - torch.matmul(torch.transpose(rounded_output_state, 0, 1), torch.matmul(w_minus, rounded_output_state)) \
                + torch.matmul(torch.transpose(torch.kron(one_vector, one_vector), 0, 1), torch.matmul(W_plus, torch.kron(one_vector, one_vector))) \
                - torch.matmul(torch.transpose(torch.kron(rounded_output_state, rounded_output_state), 0, 1), torch.matmul(W_minus, torch.kron(rounded_output_state, rounded_output_state))))/2

            cut_array[rep, epoch] = obj.item()
            print('Epoch Number ' + str(epoch))
            print('HTAACQSOS (unrounded): ' + str(unrounded_obj.item()))
            all_unrounded_sol.append(unrounded_obj)
            print('HTAACQSOS (rounded): ' + str(obj.item()))
            all_rounded_sol.append(obj)
            print('Pure Loss: ' + str(loss.item()))
            all_loss.append(loss.item())
            print()

        loss.backward() # have PyTorch calculate that backwards pass
        opt.step() # update our learned parameters (the angles of the Rotation Y gates)
        opt.zero_grad() # zero out the grad on each pass to reset for the next epoch (training step)


    ### plotting
            
    os.makedirs('./saved_figures/' + instance + '/' + str(coeff_base) + '_' + str(reg), exist_ok=True)  

    plt.plot(range(nepochs), all_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.scatter(all_loss.index(min(all_loss)), min(all_loss), label = 'min: ' + str(min(all_loss)), color = 'red')
    plt.legend()
    plt.savefig('./saved_figures/' + instance + '/' + str(coeff_base) + '_' + str(reg) + '/loss_over_epochs')
    plt.clf() 

    plt.plot(range(nepochs), all_rounded_sol)
    plt.ylabel('HTAACQSOS')
    plt.xlabel('Epoch')
    plt.title('Rounded HTAAC-QSOS Solution')
    plt.scatter(all_rounded_sol.index(max(all_rounded_sol)), max(all_rounded_sol), label = 'max: ' + str(max(all_rounded_sol).item()), color = 'red')  
    plt.legend()
    plt.savefig('./saved_figures/' + instance + '/' + str(coeff_base) + '_' + str(reg) + '/rounded_sol_over_epochs')
    plt.clf()

    plt.plot(range(nepochs), all_unrounded_sol)
    plt.ylabel('HTAACQSOS')
    plt.xlabel('Epoch')
    plt.title('Unrounded HTAAC-QSOS Solution')
    plt.scatter(all_unrounded_sol.index(max(all_unrounded_sol)), max(all_unrounded_sol), label = 'max: ' + str(max(all_unrounded_sol).item()), color = 'red')  
    plt.legend()
    plt.savefig('./saved_figures/' + instance + '/' + str(coeff_base) + '_' + str(reg) + '/unrounded_sol_over_epochs')
    plt.clf()

    plt.scatter(all_loss, all_rounded_sol, s = 10)
    plt.axvline(x = min(all_loss), color = 'r')
    plt.ylabel('HTAACQSOS')
    plt.xlabel('loss')
    plt.title('correlation between loss function and objective')
    plt.savefig('./saved_figures/' + instance + '/' + str(coeff_base) + '_' + str(reg) + '/correlation')
