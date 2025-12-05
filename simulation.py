from pennylane import qchem
import pennylane as qml
import numpy as np
from jax import numpy as jnp
import optax
import jax
from qiskit_ibm_runtime import QiskitRuntimeService, fake_provider, SamplerV2 as Sampler
import ffsim
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
import qiskit_addon_sqd as sqd
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.fermion import (
    bitstring_matrix_to_ci_strs,
    solve_fermion,
)
from qiskit_addon_sqd.subsampling import postselect_and_subsample, postselect_by_hamming_right_and_left, subsample
import pyscf
from pyscf import ao2mo, tools
import openfermionpyscf
import openfermion
from openfermion import MolecularData
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
    def __init__(self, mol: qchem.Molecule | pyscf.gto.Mole,
                 qb: Literal["default.qubit", "default.mixed", "lightning.qubit", "lightning.gpu"] = "default.qubit",
                 name: str = ""):

        self.molecule = mol
        self.name = name

        # Chemical/Fermionic Hamiltonian
        if type(self.molecule) == qchem.Molecule:
            try:
                self.chem_hamiltonian = qchem.fermionic_hamiltonian(mol)()  # Defaults to RHF
                rhf = True
                self.n_qubits = len(self.chem_hamiltonian.wires)
            except ValueError as e:
                print("Unavailable to use RHF method: {}".format(e), "\n changing to OpenFermion...")
                rhf = False

            # Build molecule with scf
            self.mol_scf = pyscf.gto.Mole()
            self.mol_scf.build(
                atom=[[mol.symbols[a], mol.coordinates[a]] for a in range(len(mol.symbols))],
                basis=mol.basis_name,
                spin=(mol.mult-1)/2,
                charge=mol.charge,
            )



            # Properties used in VQE
            self.n_electrons = self.molecule.n_electrons

            # Apply Fermionic Mapping
            if rhf:
                self.qubit_hamiltonian, self.basis = fermion_to_qb(self.chem_hamiltonian)
            else:
                self.basis = "jordan_wigner"  # Defaults to jordan_wigner mapping if fermion_to_qb is not able to run

            # Test for using molecular_hamiltonian
            self.qubit_hamiltonian, self.n_qubits = qchem.molecular_hamiltonian(mol, method="openfermion",
                                                                                mapping=self.basis)



        else:
            self.n_electrons = mol.nelectron
            self.n_qubits = mol.nelectron*2
            self.basis = 'jordan_wigner'
            self.mol_scf = mol
        # Define circuit device
        self.device = qml.device(qb)



        # IF input molecule is pyscf.gto.Mole, then vqe cant be run, default to sqd sim





    def run_simulation(self, step, n_steps, qpe_refine = False,
                       method: Literal["vqe", "sqd"]="vqe", key: str | None = None):
        """
        Method to find the lowest energy state, Hamiltonian and ansatz
        :param step: optimizer step-size
        :param n_steps: number of steps to run optimizer
        :param qpe_refine: whether to refine the state with QPE (not implemented yet)
        :return:
        """
        if type(self.molecule) is qml.qchem.Molecule:
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
            return jnp.real(circuit(theta))

        def optimization(stepsize: int, num_steps: int, param: jnp.array):

            print(f"Optimizing {self.name if (self.name != "") else self.molecule.symbols}, {self.basis} basis chosen")

            optim = optax.adam(stepsize)

            opt_state = optim.init(param)

            for i in range(num_steps):
                grads = jax.grad(vqe_cost)(param)
                updates, opt_state = optim.update(grads, opt_state)
                param = optax.apply_updates(param, updates)
                if i % int(num_steps / 5) == 0:
                    print(f"Step {i:3d} | Energy = {vqe_cost(param):.8f} | params = {param}")

            energy = vqe_cost(param)
            print(f"Estimated ground state energy (VQE): {energy}")
            return param, energy

        if method == "vqe" and not (type(self.molecule) is pyscf.gto.Mole):
            init_params = jnp.zeros(params_len)
            new_params, energy = optimization(step, n_steps, init_params)

        elif method[:3] == "sqd":
            print(f"Running {method.upper()}")


            open_shell = self.molecule.spin != 0
            spin_sq = 0
            mol = self.mol_scf

            if open_shell:
                scf = pyscf.scf.UHF(mol).run()
            else:
                scf = pyscf.scf.RHF(mol).run()

            ccsd = pyscf.cc.CCSD(
                scf
            )

            # Define active space
            ccsd.set_frozen()
            n_frozen = ccsd.frozen  # number of orbitals not contributing to possible configurations
            active_space = range(n_frozen, mol.nao_nr())

            num_orbitals = len(active_space)

            if len(scf.mo_occ.shape) > 1:
                n_electrons = int(sum(sum(x[active_space]) for x in scf.mo_occ))
            else:
                n_electrons = int(sum(scf.mo_occ[active_space]))

            num_elec_a = (n_electrons + mol.spin) // 2
            num_elec_b = (n_electrons - mol.spin) // 2


            cas = pyscf.mcscf.CASCI(scf, num_orbitals, (num_elec_a, num_elec_b))
            mo = cas.sort_mo(active_space, base=0)
            hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
            eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), num_orbitals)

            # Get t1 and t2 amplitudes from CCSD for ansatz
            ccsd.run()

            t1 = ccsd.t1
            t2 = ccsd.t2

            n_reps = 2
            alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)]
            alpha_beta_indices = [(p, p) for p in range(0, num_orbitals, 4)]

            if open_shell:
                ucj_op = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
                    t2=t2,
                    t1=t1,
                    n_reps=n_reps,
                )
            else:
                ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                    t2=t2,
                    t1=t1,
                    n_reps=n_reps,
                    interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
                )

            nelec = (num_elec_a, num_elec_b)

            # Create the circuit
            qubits = QuantumRegister(2 * num_orbitals, name="q")
            circuit = QuantumCircuit(qubits)

            # Initialize circuit
            circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), qubits)

            # Apply UCJ (Balanced if molecule is closed-shell, Unbalanced otherwise)
            if open_shell:
                circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op), qubits)
            else:
                circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
            circuit.measure_all()

            # determine what backend to use to run circuit
            if key != None:
                service = QiskitRuntimeService(token=key)
                backend = service.least_busy(operational=True, simulator=False)
            else:
                backend = AerSimulator(method='automatic')

            # If running on real IBM QC, you can create this compilation layout
            # for the spin up electrons and spin down electrons...
            # Not used in current configuration
            spin_a_layout = [0, 14]
            spin_b_layout = [2, 3]
            initial_layout = spin_a_layout + spin_b_layout

            # if initial layout needed, add it as a keyword to function below (initial_layout=initial_layout)
            pass_manager = qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager(
                optimization_level=3, backend=backend,
            )

            pass_manager.pre_init = ffsim.qiskit.PRE_INIT
            isa_circuit = pass_manager.run(circuit)


            sampler = Sampler(mode=backend)
            sampler.options.dynamical_decoupling.enable = True

            job = sampler.run([isa_circuit], shots=10_000)

            primitive_result = job.result()
            pub_result = primitive_result[0]
            counts = pub_result.data.meas.get_counts()

            # Convert counts into bitstring and probability arrays
            bitstring_matrix_full, probs_arr_full = sqd.counts.counts_to_arrays(counts)


            rng = np.random.default_rng(24)
            # SQD options
            iterations = n_steps

            # Eigenstate solver options
            n_batches = 5
            samples_per_batch = step
            max_davidson_cycles = 300

            # Self-consistent configuration recovery loop
            e_hist = np.zeros((iterations, n_batches))  # energy history
            s_hist = np.zeros((iterations, n_batches))  # spin history
            occupancy_hist = []
            avg_occupancy = None
            prnt = False

            for i in range(iterations):
                if i % (iterations//5) == 0:
                    print(f"Starting configuration recovery iteration {i}")
                    prnt = True

                # Noisy configurations
                if avg_occupancy is None:
                    bs_mat_tmp = bitstring_matrix_full
                    probs_arr_tmp = probs_arr_full

                # If we have average orbital occupancy information, we use it to refine the full set of noisy configurations
                else:
                    bs_mat_tmp, probs_arr_tmp = recover_configurations(
                        bitstring_matrix_full,
                        probs_arr_full,
                        avg_occupancy,
                        num_elec_a,
                        num_elec_b,
                        rand_seed=rng,
                    )

                # Create batches of subsamples. We post-select here to remove configurations
                # with incorrect hamming weight during iteration 0, since no config recovery was performed.
                batches = postselect_by_hamming_right_and_left(
                    bs_mat_tmp,
                    probs_arr_tmp,
                    hamming_right=num_elec_a,
                    hamming_left=num_elec_b,
                )

                new_bs_mat, new_probs_arr = postselect_by_hamming_right_and_left(
                    bs_mat_tmp,
                    probs_arr_tmp,
                    hamming_right=num_elec_a,
                    hamming_left=num_elec_b,
                )

                batches = subsample(
                    new_bs_mat,
                    new_probs_arr,
                    samples_per_batch = samples_per_batch,
                    num_batches = n_batches,
                    rand_seed = rng,
                )
                # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
                e_tmp = np.zeros(n_batches)
                s_tmp = np.zeros(n_batches)
                occs_tmp = []
                coeffs = []
                for j in range(n_batches):
                    strs_a, strs_b = bitstring_matrix_to_ci_strs(batches[j])

                    if prnt:
                        print(f"  Batch {j} subspace dimension: {len(strs_a) * len(strs_b)}")


                    energy_sci, coeffs_sci, avg_occs, spin = solve_fermion(
                        batches[j],
                        hcore,
                        eri,
                        open_shell=open_shell,
                        spin_sq=spin_sq,
                        max_cycle=max_davidson_cycles,
                    )

                    energy_sci += nuclear_repulsion_energy
                    e_tmp[j] = energy_sci
                    s_tmp[j] = spin
                    occs_tmp.append(avg_occs)
                    coeffs.append(coeffs_sci)

                if prnt:
                    prnt = False

                # Combine batch results
                avg_occupancy = tuple(np.mean(occs_tmp, axis=0))

                # Track optimization history
                e_hist[i, :] = e_tmp
                s_hist[i, :] = s_tmp
                occupancy_hist.append(avg_occupancy)

            print("Last found energy:", e_hist[-1])

            energy = e_hist[-1]

        return energy


# def adsorption_energy(step, n_steps, adsorbate:, substrate, combo, method: Literal["vqe", "sqd"]="sqd"):


def test_hydrogen():
    stepsize = 0.2
    num_steps = 200

    symbols = ['H', 'H']
    geometry = jnp.array([[0.0, 0.0, -0.69434785],
                          [0.0, 0.0, 0.69434785]])
    # alpha = jnp.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]])
    hydrogen = qml.qchem.Molecule(symbols, geometry)
    energy = Chemical(hydrogen, name="Hydrogen Molecule").run_simulation(stepsize, num_steps)
    return energy

def test_lithium():
    stepsize = 0.2
    num_steps = 1000

    symbols = ['Li']
    geometry = jnp.array([[0.0, 0.0, 0.0]])
    lithium = qml.qchem.Molecule(symbols, geometry, mult=2)
    energy = Chemical(lithium, name="Lithium Atom").run_simulation(stepsize, num_steps)
    return energy

def test_lithium2():
    stepsize = 0.2
    num_steps = 1000

    symbols = ['Li', 'Li']
    geometry = jnp.array([[0.0, 0.0, 0.0],
                         [0.0, 0.0, 2.6730]])
    lithium = qml.qchem.Molecule(symbols, geometry, mult=2)
    energy = Chemical(lithium, name="Lithium Atom").run_simulation(stepsize, num_steps)
    return energy

def test_li_ion():
    stepsize = 0.2
    num_steps = 50

    symbols = ['Li']
    geometry = jnp.array([[0.0, 0.0, 0.0]])
    lithium = qml.qchem.Molecule(symbols, geometry, charge=1)
    energy = Chemical(lithium).run_simulation(stepsize, num_steps)
    return energy

def test_beryllium():
    """ Simple test of the Beryllium Ground state energy"""
    stepsize = 0.2
    num_steps = 200

    symbols = ['Be']
    geometry = jnp.array([[0.0, 0.0, 0.0]])
    # alpha = jnp.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]], requires_grad = False)
    lithium = qml.qchem.Molecule(symbols, geometry)
    energy = Chemical(lithium).run_simulation(stepsize, num_steps)
    return energy

def test_hydrogen_sqd():
    stepsize = 500
    num_steps = 100

    hydrogen = pyscf.gto.Mole()
    hydrogen.build(
        atom=[["H", (0, 0, -0.69434785)], ["H", (0, 0, 0.69434785)]],
        basis="sto-3g",
    )
    # hydrogen = qml.qchem.Molecule(symbols, geometry)
    energy = Chemical(hydrogen, name="Hydrogen Molecule").run_simulation(stepsize, num_steps, method="sqd")
    return energy

def test_li_ion_sqd():
    stepsize = 500
    num_steps = 100

    hydrogen = pyscf.gto.Mole()
    hydrogen.build(
        atom=[["Li", (0, 0, 0)]],
        basis="sto-3g",
        charge=1,
    )
    # hydrogen = qml.qchem.Molecule(symbols, geometry)
    energy = Chemical(hydrogen, name="Hydrogen Molecule").run_simulation(stepsize, num_steps, method="sqd")
    return energy

def test_aluminum():
    stepsize = 50
    num_steps = 100

    aluminum = pyscf.gto.Mole()
    aluminum.build(
        atom=[["Al", (0, 0, 0)]],
        basis="sto-3g",
        spin = 1,
    )

    energy = Chemical(aluminum, name="Aluminum Atom").run_simulation(stepsize, num_steps, method="sqd")
    return energy

def test_titanium():
    stepsize = 50
    num_steps = 100

    titanium = pyscf.gto.Mole()
    titanium.build(
        atom=[["Ti", (0, 0, 0)]],
        basis="def2-SVP",
    )
    energy = Chemical(titanium, name="Titanium Atom").run_simulation(stepsize, num_steps, method="sqd")
    return energy

def test_platinum():
    stepsize = 50
    num_steps = 100

    platinum = pyscf.gto.Mole()
    platinum.build(
        atom=[["Pt", (0, 0, 0)]],
        basis="def2-SVP",
        ecp="def2-SVP"
    )
    energy = Chemical(platinum, name="Hydrogen Molecule").run_simulation(stepsize, num_steps, method="sqd")
    return energy

def test_molybdenum():
    stepsize = 50
    num_steps = 100

    molybdenum = pyscf.gto.Mole()
    molybdenum.build(
        atom=[["Mo", (0, 0, 0)]],
        basis="def2-SVP",
        ecp="def2-SVP"
    )
    energy = Chemical(molybdenum, name="Molybdenum Atom").run_simulation(stepsize, num_steps, method="sqd")
    return energy


if __name__ == "__main__":
    import time
    jax.config.update("jax_enable_x64", True)

    # Testing time to run VQE vs SQD
    time_start = time.time()
    h_nrg = test_hydrogen()
    vqe_time_h = time.time() - time_start

    time_start = time.time()
    h_nrg_2 = test_hydrogen_sqd()
    sqd_time_h = time.time() - time_start

    time_start = time.time()
    li_ion_nrg = test_li_ion()
    vqe_time_li_ion = time.time() - time_start

    time_start = time.time()
    li_ion_nrg_2 = test_li_ion_sqd()
    sqd_time_li_ion = time.time() - time_start

    print(f"Times to compile - Energy: ")
    print(f"\nHydrogen")
    print(f"VQE: {round(vqe_time_h, 2)}s - {h_nrg} H, "
          f"SQD: {round(sqd_time_h, 2)}s - {min(h_nrg_2)} H")
    print(f"\nLithium")
    print(f"VQE time: {round(vqe_time_li_ion, 2)}s - {li_ion_nrg} H, "
          f"SQD Time: {round(sqd_time_li_ion, 2)}s - {min(li_ion_nrg_2)} H")

    print(f"\nSQD Average accuracy: {round(((min(li_ion_nrg_2)/li_ion_nrg)+(min(h_nrg_2)/h_nrg))*50, 0)}%")
    print(f"\nSQD Li ion speedup: {round(vqe_time_li_ion/sqd_time_li_ion, 0)}x")

    h_nrg = test_hydrogen_sqd()
    print("Hydrogen: ", min(h_nrg), "H")

    al_nrg = test_aluminum()
    print("Aluminum: ", min(al_nrg), "H")

    # mo_nrg = test_molybdenum()
    # print("Molybdenum: ", min(mo_nrg), "H")



    # ti_nrg = test_titanium()
    # print("Titanium: ", min(ti_nrg), "H")



    """Restricted Hartree-Fock (RHF) methods, often used by default, do not support open-shell systems 
    (those with unpaired electrons). 
    Suggestion for future improvement: Try using PySCF with the appropriate settings 
    (e.g., Unrestricted Hartree-Fock, UHF) to handle open-shell configurations."""