import pennylane as qml
from pennylane.qchem import fermionic_hamiltonian
from pennylane import numpy as np
from pennylane import jordan_wigner, bravyi_kitaev
from pennylane.fermi import parity_transform   # Parity mapping (requires n_qubits)

def fermion_to_qb(fermionic_hamiltonian, n_qubits):
    """
    Apply Jordan–Wigner, Bravyi–Kitaev, and Parity mappings to a fermionic Hamiltonian,
    compute arithmetic_depth of each mapped qubit Hamiltonian,
    and return the one with minimal depth.

    Parameters
    ----------
    fermionic_hamiltonian : FermiSentence
        The fermionic Hamiltonian to map.
    n_qubits : int
        Number of spin–orbitals (required by parity_transform).

    Returns
    -------
    best_name : str
        'jordan_wigner', 'bravyi_kitaev', or 'parity'
    best_hamiltonian : qml.Hamiltonian
        Qubit Hamiltonian with minimal arithmetic depth.
    all_results : dict
        All mappings with their depth and qubit Hamiltonians.
    """

    mappings = {
        "jordan_wigner": lambda h: jordan_wigner(h),
        "bravyi_kitaev": lambda h: bravyi_kitaev(h, n_qubits),
        "parity": lambda h: parity_transform(h, n=n_qubits),
    }

    results = {}

    for name, mapping_fn in mappings.items():
        # 1. Map fermions → qubits
        H_qubit = mapping_fn(fermionic_hamiltonian)

        # 2. Measure arithmetic depth
        depth = H_qubit.arithmetic_depth

        results[name] = {
            "hamiltonian": H_qubit,
            "depth": depth,
        }

    # 3. Pick mapping with smallest arithmetic depth
    best_name = min(results, key=lambda k: results[k]["depth"])
    best_hamiltonian = results[best_name]["hamiltonian"]

    return best_name, best_hamiltonian, results

# ---------------------------
# Build molecule
# ---------------------------
symbols  = ['H', 'H']
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
                  [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
args = [alpha]
h = fermionic_hamiltonian(mol)#(*args)
print(h)
fermionic_op = h()     # FermiSentence

# ---------------------------
# Number of qubits
# ---------------------------
n_qubits = len(fermionic_op.wires)

# ---------------------------
# Run the comparison
# ---------------------------
best_name, best_H, all_maps = fermion_to_qb(fermionic_op, n_qubits)

print("Best mapping:", best_name)
print("Best arithmetic depth:", all_maps[best_name]["depth"])
print("Best qubit Hamiltonian:\n", best_H)