from pennylane import qchem
from pennylane import numpy as np
import pennylane as qml
from jax import numpy as np
import optax
import jax
import pyscf
import openfermionpyscf
from typing import Literal
from pennylane.fermi import from_string
from pennylane import jordan_wigner, bravyi_kitaev
from pennylane.fermi import parity_transform

def fermion_to_qb(fermion):
    """
    Apply Jordan–Wigner, Bravyi–Kitaev, and Parity mappings to a fermionic Hamiltonian,
    compute arithmetic_depth of each mapped qubit Hamiltonian,
    and return the one with minimal depth.

    Args:
        fermion (FermiSentence): The fermionic Hamiltonian to map.

    Returns:
        tuple:
            best_hamiltonian (qml.Hamiltonian): Qubit Hamiltonian with minimal depth.
            best_name (str): Name of the mapping producing this Hamiltonian. One of these occupation_number (jordan_wigner), bravyi_kitaev, or parity
    """

    mappings = {
        "jordan_wigner": lambda h: jordan_wigner(h),
        "bravyi_kitaev": lambda h: bravyi_kitaev(h, max(fermion.wires)+1),
        "parity": lambda h: parity_transform(h, n=max(fermion.wires)+1),
    }

    results = {}

    for name, mapping_fn in mappings.items():
        # 1. Map fermions → qubits
        H_qubit = mapping_fn(fermion)

        # 2. Measure arithmetic depth
        depth = H_qubit.arithmetic_depth

        results[name] = {
            "hamiltonian": H_qubit,
            "depth": depth,
        }

    # 3. Pick mapping with smallest arithmetic depth
    best_name = min(results, key=lambda k: results[k]["depth"])
    best_hamiltonian = results[best_name]["hamiltonian"]

    #print("Best mapping:", best_name)
    #print("Best arithmetic depth:", results[best_name]["depth"])
    #print("Best qubit Hamiltonian:\n", best_hamiltonian)
    return best_hamiltonian, best_name

# "default.mixed", "lightning.qubit", "lightning.gpu" do not work yet!
class Chemical:
    def __init__(self, mol: qchem.Molecule,
                 qb: Literal["default.qubit", "default.mixed", "lightning.qubit", "lightning.gpu"] = "default.qubit",
                 name: str = ""):

        self.molecule = mol
        self.name = name

        # Chemical/Fermionic Hamiltonian
        try:
            self.chem_hamiltonian = qchem.fermionic_hamiltonian(mol)()  # Defaults to RHF
            rhf = True
            self.n_qubits = len(self.chem_hamiltonian.wires)
        except ValueError as e:
            print("Unavailable to use RHF method: {}".format(e), "\n changing to OpenFermion...")
            rhf = False


        # Properties used in VQE
        self.n_electrons = self.molecule.n_electrons


        # Apply Fermionic Mapping
        if rhf:
            self.qubit_hamiltonian, self.basis = fermion_to_qb(self.chem_hamiltonian)
        else:
            self.basis = "jordan_wigner" # Defaults to jordan_wigner mapping if fermion_to_qb is not able to run

        # Test for using molecular_hamiltonian
        self.qubit_hamiltonian, self.n_qubits = qchem.molecular_hamiltonian(mol, method="openfermion", mapping=self.basis)

        # Define circuit device
        self.device = qml.device(qb)

    def run_vqe(self, step, n_steps, qpe_refine = False):
        """
        Method to find the lowest energy state, Hamiltonian and ansatz
        :param step: optimizer step-size
        :param n_steps: number of steps to run optimizer
        :param qpe_refine: whether to refine the state with QPE (not implemented yet)
        :return:
        """
        init_state = qchem.hf_state(self.n_electrons, self.n_qubits, basis=self.basis)

        singles, doubles = qchem.excitations(self.n_electrons, self.n_qubits)

        params_len = len(singles+doubles)

        singles_fermi = []
        for ex in singles:
            singles_fermi.append(from_string(f"{ex[1]}+ {ex[0]}-")
                                 - from_string(f"{ex[0]}+ {ex[1]}-"))

        doubles_fermi = []
        for ex in doubles:
            doubles_fermi.append(from_string(f"{ex[3]}+ {ex[2]}+ {ex[1]}- {ex[0]}-")
                                 - from_string(f"{ex[0]}+ {ex[1]}+ {ex[2]}- {ex[3]}-"))

        singles_pauli = []
        for op in singles_fermi:
            singles_pauli.append(fermion_to_qb(op)[0])

        doubles_pauli = []
        for op in doubles_fermi:
            doubles_pauli.append(fermion_to_qb(op)[0])

        @qml.qnode(device=self.device)
        def circuit_1(theta):
            """Failed attempt 1"""
            qml.BasisState(init_state, [x for x in range(self.n_qubits)])
            # Ansatz
            # ansatz(theta)
            final_i = len(doubles_pauli)
            doubles_hamilt_ops = doubles_pauli
            doubles_hamilt_coeffs = theta[:final_i]
            singles_hamilt_ops = singles_pauli
            singles_hamilt_coeffs = theta[final_i:]

            doubles_hamiltonian = qml.Hamiltonian(doubles_hamilt_coeffs, doubles_hamilt_ops)
            singles_hamiltonian = qml.Hamiltonian(singles_hamilt_coeffs, singles_hamilt_ops)

            qml.ApproxTimeEvolution(doubles_hamiltonian, 0.5, 1)
            qml.ApproxTimeEvolution(singles_hamiltonian, 0.5, 1)

            return qml.expval(self.qubit_hamiltonian)

        @qml.qnode(device=self.device)
        def circuit_2(theta):
            "Failed attempt 2"
            qml.BasisState(init_state, [x for x in range(self.n_qubits)])

            # Ansatz
            exit_hamil = qml.Hamiltonian(theta / 2, doubles_pauli+singles_pauli)
            qml.ApproxTimeEvolution(exit_hamil, 1, 25)

            return qml.expval(self.qubit_hamiltonian)

        # Single & Double Excitations for Ansatz
        def ansatz(theta):

            # UCCSD Ansatz (Unitary Coupled-Clusters Singles and Doubles)
            for i, excitation in enumerate(doubles_pauli):
                qml.exp(excitation * theta[i] / 2)

            for j, excitation in enumerate(singles_pauli):
                qml.exp(excitation * theta[i + j + 1] / 2)

        @jax.jit
        @qml.qnode(self.device, interface="jax")
        def circuit(theta):
            qml.BasisState(init_state, wires=range(self.n_qubits))

            ansatz(theta)

            return qml.expval(self.qubit_hamiltonian)


        def vqe_cost(theta):
            return np.real(circuit(theta))

        def optimization(stepsize: int, num_steps: int, param: np.array):

            print(f"Optimizing {self.name if (self.name != "") else self.molecule.symbols}, {self.basis} basis chosen")

            optim = optax.adam(stepsize)

            opt_state = optim.init(param)

            for i in range(num_steps):
                grads = jax.grad(vqe_cost)(param)
                updates, opt_state = optim.update(grads, opt_state)
                param = optax.apply_updates(param, updates)
                if i % int(num_steps / 20) == 0:
                    print(f"Step {i:3d} | Energy = {vqe_cost(param):.8f} | params = {param}")

            energy = vqe_cost(param)
            print(f"Estimated ground state energy (VQE): {energy}")
            return param, energy

        init_params = np.zeros(params_len)
        new_params, energy = optimization(step, n_steps, init_params)

        # QPE refining logic - If we decide to go forward with QPE transfer, expand on this
        if qpe_refine:
            # run optimized circuit, then apply QPE to different circuit, and measure first circuit (tracing out).
            # This leaves us with a new initial state which we can train further if we want

            # PREPARE OUR GROUND STATE
            def prep():
                qml.BasisState(init_state, wires=range(self.n_qubits))

                ansatz(new_params)

            # PHASE ESTIMATION ONTO NEW REGISTER WITH SAME N_QUBITS
            def move_to_new_reg():
                pass


            # MEASURE PREVIOUS REGISTER
            def trace_out():
                pass

        return energy

    # Unfinished - Need a way to create the combined Molecule configuration
    def adsorption_energy(self, step, n_steps):
        # Adsorbate Energy
        H_2 = -1.1357

        # Candidate energy
        candidate = self.run_vqe(step, n_steps)

        # Combination energy
        combo = self.run_vqe(step, n_steps)

def test_hydrogen():
    stepsize = 0.2
    num_steps = 200

    symbols = ['H', 'H']
    geometry = np.array([[0.0, 0.0, -0.69434785],
                          [0.0, 0.0, 0.69434785]])
    # alpha = np.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]])
    hydrogen = qml.qchem.Molecule(symbols, geometry)
    energy = Chemical(hydrogen, name="Hydrogen Molecule").run_vqe(stepsize, num_steps)
    return energy

def test_lithium():
    stepsize = 0.2
    num_steps = 1000

    symbols = ['Li']
    geometry = np.array([[0.0, 0.0, 0.0]])
    lithium = qml.qchem.Molecule(symbols, geometry, mult=2)
    energy = Chemical(lithium, name="Lithium Atom").run_vqe(stepsize, num_steps)
    return energy

def test_lithium2():
    stepsize = 0.2
    num_steps = 1000

    symbols = ['Li', 'Li']
    geometry = np.array([[0.0, 0.0, 0.0],
                         [0.0, 0.0, 2.6730]])
    lithium = qml.qchem.Molecule(symbols, geometry, mult=2)
    energy = Chemical(lithium, name="Lithium Atom").run_vqe(stepsize, num_steps)
    return energy

def test_li_ion():
    stepsize = 0.2
    num_steps = 1000

    symbols = ['Li']
    geometry = np.array([[0.0, 0.0, 0.0]])
    lithium = qml.qchem.Molecule(symbols, geometry, charge=1)
    energy = Chemical(lithium).run_vqe(stepsize, num_steps)
    return energy

def test_beryllium():
    """ Simple test of the Beryllium Ground state energy"""
    stepsize = 0.2
    num_steps = 1000

    symbols = ['Be']
    geometry = np.array([[0.0, 0.0, 0.0]])
    # alpha = np.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]], requires_grad = False)
    lithium = qml.qchem.Molecule(symbols, geometry)
    energy = Chemical(lithium).run_vqe(stepsize, num_steps)
    return energy

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    h_nrg = test_hydrogen()
    li_nrg = test_lithium() # Multiplicity doesnt work for this, only changing it to Li+ does (ion)
    li_ion_nrg = test_li_ion()
    be_nrg = test_beryllium()
    print(h_nrg, be_nrg)

    """Restricted Hartree-Fock (RHF) methods, often used by default, do not support open-shell systems 
    (those with unpaired electrons). 
    Suggestion for future improvement: Try using PySCF with the appropriate settings 
    (e.g., Unrestricted Hartree-Fock, UHF) to handle open-shell configurations."""