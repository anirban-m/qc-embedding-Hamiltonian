def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy
import copy
import ipyparallel as ipp
from qiskit.quantum_info import Pauli
from qiskit.chemistry import FermionicOperator
from qiskit.aqua.operators.legacy import op_converter
from openfermion.circuits import slater_determinant_preparation_circuit
from qiskit import QuantumCircuit,execute
from qiskit import Aer
from joblib import Parallel,delayed
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.chemistry.components.variational_forms import UCCSD
import scipy
import time
c =ipp.Client()
num_qubits=8
c.ids
v1 = c.load_balanced_view([0,1,2,3,4,5,6,7])
#Computing energy gradient for newly added operators
def countYgates(pauli_label):
    countYgates = sum(map(lambda x : 1 if 'Y' in x else 0, pauli_label))
    return countYgates
def compute_gradient(data):
    circLast,Op=data
    circ1=circLast.copy()
    circ2=circLast.copy()
    state1=execute(circ1,backend,shots=1024).result().get_statevector() 
    E1=(numpy.conjugate(state1)@Hmat@state1).real
    label=Op.paulis[0][1].to_label()
    Label=numpy.array(list(label))
    qubits_to_act_on=list(numpy.where(Label!='I')[0])
    substring=''.join(list(Label[qubits_to_act_on]))
    UnitaryMat=Pauli.from_label(substring).to_matrix()
    UnitaryMatrix1=numpy.cos(0.01)*numpy.eye(2**len(qubits_to_act_on))-1j*numpy.sin(0.01)*UnitaryMat
    UnitaryMatrix2=numpy.cos(0.01)*numpy.eye(2**len(qubits_to_act_on))+1j*numpy.sin(0.01)*UnitaryMat
    circ1.unitary(UnitaryMatrix1,qubits_to_act_on[::-1],label=label)
    circ2.unitary(UnitaryMatrix2,qubits_to_act_on[::-1],label=label)
    state1=execute(circ1,backend,shots=1024).result().get_statevector() 
    state2=execute(circ2,backend,shots=1024).result().get_statevector() 
    E1=(numpy.conjugate(state1)@Hmat@state1).real
    E2=(numpy.conjugate(state2)@Hmat@state2).real
    grad=(E1-E2)/0.02
    return grad
def commutator(A,B):
        B2=WeightedPauliOperator([[-B.paulis[0][0],B.paulis[0][1]]])
        return A.multiply(B).add(B2.multiply(A))
#commutator pool    
def commutatorPool(qubitH):
    #construct commutator pool from the Hamiltonian
    pool_H=[WeightedPauliOperator([[1j,qubitH.paulis[i][1]]]) for i in range(len(qubitH.paulis))][1:]
    #commutator between operators
    commutator_pool=[WeightedPauliOperator([[1j,commutator(op1,op2).paulis[0][1]]]) for op1 in pool_H for op2 in pool_H if countYgates(commutator(op1,op2).paulis[0][1].to_label())%2==1]
    labels1=[commutator_pool[i].paulis[0][1].to_label() for i in range(len(commutator_pool))]
    unique_labels,indices=numpy.unique(labels1,return_index=True)
    commutator_pool=[commutator_pool[indices[i]] for i in range(len(indices))]
    commutator_pool_2=[WeightedPauliOperator([[1j,commutator(op1,op2).paulis[0][1]]]) for op1 in commutator_pool for op2 in pool_H if countYgates(commutator(op1,op2).paulis[0][1].to_label())%2==1 and commutator(op1,op2).paulis[0][1].to_label() not in labels1]
    labels2=[commutator_pool_2[i].paulis[0][1].to_label() for i in range(len(commutator_pool_2))]
    unique_labels,indices=numpy.unique(labels2,return_index=True)
    commutator_pool_2=[commutator_pool_2[indices[i]] for i in range(len(indices))]
    return numpy.array(commutator_pool),numpy.array(commutator_pool_2)   
def Energy(params):
    circ=ansatz_circuit(PaulisAndMats,params) #var_form_base.construct_circuit(parameters=params) 
    state=execute(circ,backend,shots=1024).result().get_statevector() 
    E=(numpy.conjugate(state)@Hmat@state).real
    return E
def ref_state(*args,**kwargs):
    init_circ=QuantumCircuit(num_qubits)
    for i in range(num_sites//2):
        init_circ.x(i)
        init_circ.x(i+num_sites)
    return init_circ 
def qubitOp(h1,h2):
    qubit_op=FermionicOperator(h1,h2).mapping('jordan_wigner')
    return qubit_op
#Ansatz Circuit Construction  
def PauliStringToMatrix(Label):
    Label1=numpy.array(list(Label))
    LabelInds=numpy.where(Label1!='I')[0]
    substring=''.join(list(Label1[LabelInds]))
    UnitaryMat=Pauli(substring).to_matrix()
    return [UnitaryMat,LabelInds]
def UnitaryMatrixForm(data):
    import numpy
    PaulisAndMats,params,ind=data
    Label=PaulisAndMats[ind][0]
    UnitaryMat,qubits_to_act_on=PaulisAndMats[ind][1]
    UnitaryMatrix=numpy.cos(params[ind])*numpy.eye(2**len(qubits_to_act_on))-1j*numpy.sin(params[ind])*UnitaryMat
    return [UnitaryMatrix,qubits_to_act_on[::-1],Label]
def ansatz_circuit(PaulisAndMats,params):
    circ=HFcirc.copy()
    if(len(PaulisAndMats)!=0):
        inp_data=[(PaulisAndMats,params,i) for i in range(len(params))]
        #result = v1.map_async(UnitaryMatrixForm, inp_data)
        #UnitaryMatArr=result.get()
        UnitaryMatArr=list(map(UnitaryMatrixForm,inp_data))
        for i in range(len(UnitaryMatArr)):
            circ.unitary(UnitaryMatArr[i][0],list(UnitaryMatArr[i][1]),label=UnitaryMatArr[i][2])
        v1.purge_results('all')
        del inp_data    
        del UnitaryMatArr
    return circ     
#construct initial state
def HF_init_state(U,return_MO_rotnOp=False):
    backend=Aer.get_backend('statevector_simulator')
    N=4
    # Qiskit implementation of Givens rotation
    def prepare_givens_rotated_state(givens):
        circ = QuantumCircuit(2*N)
        # Fill first N_f orbitals for each spin
        for i in range(N_f):
            circ.x(i)
            circ.x(i+N)
        for rots in givens:
            for tup in rots:
                #for spin down
                spin=0
                circ.cnot(tup[1]+N*spin,tup[0]+N*spin)
                circ.cry(-2*tup[2],tup[0]+N*spin, tup[1]+N*spin)
                circ.cnot(tup[1]+N*spin, tup[0]+N*spin)
                circ.rz(tup[3],tup[1]+N*spin)
                #for spin up
                spin=1
                circ.cnot(tup[1]+N*spin,tup[0]+N*spin)
                circ.cry(-2*tup[2],tup[0]+N*spin, tup[1]+N*spin)
                circ.cnot(tup[1]+N*spin, tup[0]+N*spin)
                circ.rz(tup[3],tup[1]+N*spin)
        final_state_vector=execute(circ,backend,shots=1024).result().get_statevector()      
        return circ,final_state_vector
    def cost_fn(angles):
        givens=[((1, 2, angles[0], 0.0),), ((0, 1, angles[1], 0.0), (2, 3, angles[2], 0.0)), ((1, 2, angles[3], 0.0),)]  
        #c=Givens_rot_circuit(givens)
        #state = QuantumState(N*2)
        #c.update_quantum_state(state)
        #state_vector=numpy.array(state.get_vector())
        c,state=prepare_givens_rotated_state(givens)
        E=numpy.real(numpy.dot(numpy.dot(numpy.conjugate(state),Hmat),state))
        return E
    with open('../chem_pot_for_Half_Fill.txt','r') as f:
        lines=f.readlines()[1:]
        for line in lines:
            elems=line.split()
            if int(elems[0])==U:
                muHalf=float(elems[1]) #Chem Pot for a given Hubbard U
    #Getting the one body and two body interaction vertexes
    with open('../'+str(U)+'/v1e.dat','r') as f:
            lines=f.readlines()[1:]
            num_sites=4
            chem_pot=numpy.zeros((2*num_sites,2*num_sites))
            eg_h1=numpy.zeros((2*num_sites,2*num_sites))
            for line in lines:
                elems=line.split()
                eg_h1[int(elems[0])][int(elems[1])]=float(elems[2])
                eg_h1[int(elems[0])+num_sites][int(elems[1])+num_sites]=float(elems[2])
            for i in range(2*num_sites):
                chem_pot[i][i]=-muHalf
            eg_h1=eg_h1+chem_pot        
    with open('../'+str(U)+'/v2e.dat','r') as f:
        num_sites=4
        eg_h2=numpy.zeros((2*num_sites,2*num_sites,2*num_sites,2*num_sites))
        for line in f:
            if "#" in line:
                continue
            line = line.split()
            i,j,k,l = map(int, line[:4])
            val = float(line[4])
            eg_h2[i,j,k,l] = eg_h2[i+num_sites,j+num_sites,k,l] = eg_h2[i,j,k+num_sites,l+num_sites]             = eg_h2[i+num_sites,j+num_sites,k+num_sites,l+num_sites] = 0.5*val  # convention with 0.5 factor included.
    qubitH=qubitOp(eg_h1,eg_h2)
    Hmat=op_converter.to_matrix_operator(qubitH).dense_matrix#qubitH.to_matrix(massive=True)
    E,V=numpy.linalg.eigh(eg_h1)
    energy = 2*sum(E[:int(N/2)])
    psi = V[:,:int(N/2)]
    Q = numpy.transpose(numpy.conjugate(psi))
    N_f=len(Q)
    givens_init = slater_determinant_preparation_circuit(Q)
    params_init=[]
    for k in range(len(givens_init)):
        for l in range(len(givens_init[k])):
            params_init.append(givens_init[k][l][2])        
    res = scipy.optimize.minimize(cost_fn, params_init, bounds=[[-numpy.pi,numpy.pi]]*4,method='L-BFGS-B')
    print("Final Hartree Fock Energy",res['fun'])
    #final slater determinant state vector obtained using quantum variational Hartree Fock using Given Rotations
    params_fin=res['x']
    #Final Givens Rotation Circuit obtained after implementing L-BFGS-B algorithm
    givens_fin=[((1, 2, params_fin[0], 0.0),), ((0, 1, params_fin[1], 0.0), (2, 3, params_fin[2], 0.0)), ((1, 2, params_fin[3], 0.0),)]
    HFcirc,HFstateVec=prepare_givens_rotated_state(givens_fin)
    if return_MO_rotnOp==True:
        givensOp=prepare_given_operator(givens_fin)
        return HFcirc,HFstateVec,givensOp
    return HFcirc,HFstateVec
#construct Hamiltonian
def egBandHamiltonian(U):
    #Getting chemical Potential for Half-Filling
    with open('../chem_pot_for_Half_Fill.txt','r') as f:
        lines=f.readlines()[1:]
        for line in lines:
            elems=line.split()
            if int(elems[0])==U:
                muHalf=float(elems[1]) #Chem Pot for a given Hubbard U
    #Getting the one body and two body interaction vertexes
    with open('../'+str(U)+'/v1e.dat','r') as f:
            lines=f.readlines()[1:]
            num_sites=4
            chem_pot=numpy.zeros((2*num_sites,2*num_sites))
            eg_h1=numpy.zeros((2*num_sites,2*num_sites))
            for line in lines:
                elems=line.split()
                eg_h1[int(elems[0])][int(elems[1])]=float(elems[2])
                eg_h1[int(elems[0])+num_sites][int(elems[1])+num_sites]=float(elems[2])
            for i in range(2*num_sites):
                chem_pot[i][i]=-muHalf
            eg_h1=eg_h1+chem_pot       
    with open('../'+str(U)+'/v2e.dat','r') as f:
        num_sites=4
        eg_h2=numpy.zeros((2*num_sites,2*num_sites,2*num_sites,2*num_sites))
        for line in f:
            if "#" in line:
                continue
            line = line.split()
            i,j,k,l = map(int, line[:4])
            val = float(line[4])
            eg_h2[i,j,k,l] = eg_h2[i+num_sites,j+num_sites,k,l] = eg_h2[i,j,k+num_sites,l+num_sites]             = eg_h2[i+num_sites,j+num_sites,k+num_sites,l+num_sites] = 0.5*val  # convention with 0.5 factor included.
    qubitH=qubitOp(eg_h1,eg_h2)
    Hmat=op_converter.to_matrix_operator(qubitH).dense_matrix
    w,v=numpy.linalg.eigh(Hmat)
    Eg=w[0]
    state_g=v[:,0]
    return qubitH,Hmat,Eg,state_g
def qubitOp(h1,h2):
    qubit_op=FermionicOperator(h1,h2).mapping('jordan_wigner')
    return qubit_op
def SMO(cost,params,runs=20,tol=1e-4,save_opt_steps=False):
    index=1
    conv_err=1000
    def E_landscape(ind,ang,cost,params):
        params1=copy.deepcopy(params)
        params1[ind]=params1[ind]+ang #ang
        #circ=var_form_base.construct_circuit(parameters=params1)
        E=cost(params1)
        return E.real
    def determine_unknowns(E,cost,params,ind):
        L1=E#_landscape(ind,0,params,Hmat)
        L2=E_landscape(ind,numpy.pi/4.,cost,params)
        L3=E_landscape(ind,-numpy.pi/4.,cost,params)
        ratio=(L3-L2)/(2*L1-L2-L3)
        a3=(L2+L3)/2.
        a2=2*params[ind]-numpy.arctan(ratio)
        a1=(L1-a3)/numpy.cos(numpy.arctan(ratio))
        return a1,a2,a3
    def update(E,cost,params,ind):
        a1,a2,a3=determine_unknowns(E,cost,params,ind)
        thetaStar=a2/2.+numpy.pi/2. if a1>0 else a2/2.
        newParams=copy.deepcopy(params)
        newParams[ind]=thetaStar
        updEnergy=a3-a1 if a1>0 else a3+a1
        return newParams,updEnergy.real
    while conv_err>tol and index<runs:
        print("looped "+str(index)+" times")
        Eold=cost(params)
        init=0
        for i in range(len(params)):#[::-1][0:1]:
            #first run sequential minimial optimization (SMO)  for a given multiqubit operator using 
            #exact analytical form for cost function
            if init==0:
                E=Eold
            ind=i
            params,E=update(E,cost,params,ind)
            if save_opt_steps==True:
                with open('paramsForQubitAdapt_eg_model.txt','+a') as f:
                    Str=["{:0.16f}".format(params[i].real) for i in range(len(params))]
                    print('['+','.join(Str)+']'+'#'+"{:0.16f}".format(E.real),file=f)
            else:
                continue
            init=init+1  
        conv_err=Eold-E
        print("inner loop error",conv_err)
        index=index+1
    return params,E
U=7
num_sites=4
num_qubits=8
#Constructing Hamiltonian
qubitH,Hmat,Eg,state_g=egBandHamiltonian(U)
#commutator pools
commutator_pool,commutator_pool_2=commutatorPool(qubitH)
backend=Aer.get_backend('statevector_simulator')
#preparing HF state
HFcirc,HFstate=HF_init_state(U)
params=[]
EnergyArr=[]
PaulisAndMats=[]
ExcOps=[]
counts=38
count=1
ti=time.time()
circLast=HFcirc.copy()#ref_state()
backend=Aer.get_backend('statevector_simulator')
params=[]
with open('paramsForQubitAdapt_eg_model.txt','r') as f:
    lines=f.readlines()
    for i in range(len(lines)):
        if lines[i][0]=='l':
            Label=lines[i].split('-')[1][1:-1]
            print(Label)
            PaulisAndMats.append([Label,PauliStringToMatrix(Label)])
    params=eval(lines[-1].split('#')[0])
res=SMO(Energy,params,tol=1e-9,runs=40,save_opt_steps=True)    
# for i in range(counts):
#     grads=Parallel(n_jobs=7,verbose=2)(delayed(compute_gradient)((circLast,commutator_pool[i])) for i in range(len(commutator_pool)))
#     ind=numpy.argsort(numpy.abs(grads))[-1]
#     print("max. grad",grads[ind])
#     PauliOp=commutator_pool[ind]
#     ExcOps.append(PauliOp.paulis[0][1].to_label())
#     print("chosen Op",ExcOps[-1])
#     with open('paramsForQubitAdapt_eg_model.txt','a') as f:
#         print("label-",ExcOps[-1],file=f)
#     params.append(0.0)
#     PaulisAndMats.append([ExcOps[-1],PauliStringToMatrix(ExcOps[-1])])
#     #construct circuit for estimating Hamiltonian
#     params,E=SMO(Energy,params,tol=7e-6,runs=40,save_opt_steps=True)#scipy.optimize.minimize(Energy,params,method='L-BFGS-B')#,jac=fun_jac)
#     EnergyArr.append(Energy(params))
#     print("Energy",Energy(params))
#     print("num. of parameters", len(params))
#     print("time elapsed",time.time()-ti)    
#     error=EnergyArr[-1]-Eg
#     circLast=ansatz_circuit(PaulisAndMats,params)
