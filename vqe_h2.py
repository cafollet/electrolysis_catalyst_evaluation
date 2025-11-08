import pennylane as qml
import numpy as np
from pennylane.templates import ApproxTimeEvolution

coeffs = [
    -0.138754,  
    -0.152989,  
    0.164190,  
    0.144579,  
    0.111373,  
    0.146726,  
    0.169348,  
    -0.035353,  
    0.035353   
]
# The cost hamiltonian.
H0_ops = [
    qml.Identity(0),
    qml.PauliZ(2),
    qml.PauliZ(3),
    qml.PauliZ(0),
    qml.PauliZ(1),
    qml.PauliZ(3) @ qml.PauliZ(2),
    qml.PauliZ(3) @ qml.PauliZ(1),
    qml.PauliZ(2) @ qml.PauliZ(0),
    qml.PauliZ(2) @ qml.PauliZ(1),
    qml.PauliZ(3) @ qml.PauliZ(0),
    qml.PauliZ(1) @ qml.PauliZ(0),
]
H0_coeffs_index_order = [0, 1, 1,2,2,3,4,4,5,5,6]
H0_coeffs = [coeffs[i] for i in H0_coeffs_index_order]
H0 = qml.Hamiltonian(H0_coeffs, H0_ops) # Hamiltonian takes coeffs, observables

# Mixer Hamiltonian V
V_ops = [
    qml.PauliY(3) @ qml.PauliY(2) @ qml.PauliX(1) @ qml.PauliX(0),
    qml.PauliX(3) @ qml.PauliX(2) @ qml.PauliY(1) @ qml.PauliY(0),
    qml.PauliX(3) @ qml.PauliY(2) @ qml.PauliY(1) @ qml.PauliX(0),
    qml.PauliY(3) @ qml.PauliX(2) @ qml.PauliX(1) @ qml.PauliY(0)
]
c8s = [coeffs[7]]*2
c9s = [coeffs[8]]*2
V_coeffs = c8s + c9s
V = qml.Hamiltonian(V_coeffs, V_ops)

dev_vqe = qml.device("default.qubit", wires=4)

def vqe_ansatz(theta):
    qml.BasisState(np.array([0, 0, 1, 1]), wires=[0, 1, 2, 3])
    qve_H = qml.PauliX(3) @ qml.PauliX(2) @ qml.PauliX(1) @ qml.PauliY(0)
    qml.ApproxTimeEvolution(qml.Hamiltonian([1.0], [qve_H]), time=theta[0], n=1)

@qml.qnode(dev_vqe)
def circuit(theta):
    vqe_ansatz(theta)
    return qml.expval(H0 + V)

def vqe_cost(theta):
    return circuit(theta)

def optimization(stepsize: int, num_steps: int, params:qml.numpy.array):
    optimizer = qml.AdamOptimizer(stepsize)
    for i in range(num_steps):
        params = optimizer.step(vqe_cost, params)
        if i % 100 == 0:
            energy = vqe_cost(params)
            print(f"Step {i:3d} | Energy = {energy:.8f} H2")
    energy = vqe_cost(params)
    print(f"Estimated ground state energy (VQE): {energy}")

def main():
    stepsize = 0.2
    num_steps = 1000
    theta = qml.numpy.array([0.02], requires_grad=True)
    optimization(stepsize, num_steps, theta)