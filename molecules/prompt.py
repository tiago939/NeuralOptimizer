"""Group-standard API prompt, plus optional molecular/chemical context.

STRICT COMPARABILITY: when chemical_info=None, SYSTEM and build_user_prompt()
match the coworker's original prompt.py byte-for-byte for the same arguments.
Use build_messages(..., chemical_info=...) for the informed condition.
"""
import json

# The original group text is intentionally unchanged.
SYSTEM = '''You optimize the expectation value of a supplied Hamiltonian using a fixed parameterized quantum circuit. Choose one complete absolute numerical parameter vector in radians. Do not modify the circuit or Hamiltonian. Infer parameter order from the OpenQASM 3 input declarations in their textual order. Qiskit Pauli labels run from q[n-1] on the left to q[0] on the right. Use only the provided circuit, Hamiltonian, optional noise model, and previous angle-energy observations. Return only a JSON object with keys "theta" (numeric array) and "mechanism" (brief, testable reasoning).'''

_CHEMICAL_SYSTEM = SYSTEM.replace(
    'Use only the provided circuit, Hamiltonian, optional noise model, and previous angle-energy observations.',
    'Use only the provided circuit, Hamiltonian, chemical context (when present), optional noise model, and previous angle-energy observations.',
)
assert _CHEMICAL_SYSTEM != SYSTEM


def chemical_context(problem, *, initial_theta=None, shots=0, noise=False):
    """Build chemical metadata from the *computed* molecular problem.

    Do not call this with invented orbitals/electron counts. Reference FCI
    energies are NEVER included in the model-facing prompt.
    """
    molecule = problem.molecule
    return {
        'molecule': molecule.name,
        'geometry_angstrom': [list(row) for row in molecule.atoms],
        'basis': molecule.basis.upper().replace('STO3G', 'STO-3G'),
        'charge': molecule.charge,
        'multiplicity': molecule.multiplicity,
        'full_spatial_orbitals': problem.full_spatial_orbitals,
        'full_num_particles': list(problem.full_num_particles),
        'active_spatial_orbitals': problem.num_spatial_orbitals,
        'active_num_particles': list(problem.num_particles),
        'active_orbital_indices': (molecule.active_space or {}).get('orbital_indices'),
        'n_qubits': problem.H.num_qubits,
        'ansatz': problem.ansatz.upper(),
        'mapping': 'Jordan-Wigner',
        'reference_state': 'Hartree-Fock; preparation included in the OpenQASM',
        'hf_basis_index': problem.hf_index,
        'initial_theta': list(map(float, initial_theta)) if initial_theta is not None else None,
        'shots_per_energy_query': int(shots),
        'noise': bool(noise),
        'energy_convention': ('Hamiltonian Pauli coefficients describe electronic energy. '
                              'Observed feedback uses electronic energy, excluding additive '
                              'nuclear and frozen-core offsets.'),
    }


def build_system_prompt(*, chemical_info=None):
    return SYSTEM if chemical_info is None else _CHEMICAL_SYSTEM


def build_user_prompt(qasm, terms, history=None, disclosed_noise=None, *, chemical_info=None):
    """Original benchmark prompt plus an *optional* chemical-context section."""
    # START UNMODIFIED group-standard body.
    sections = [
        'CIRCUIT_OPENQASM_3:',
        qasm,
        'HAMILTONIAN_PAULI_TERMS:',
        json.dumps(terms, indent=2),
    ]
    if disclosed_noise is not None:
        sections += ['NOISE_MODEL:', json.dumps(disclosed_noise, indent=2)]
    if history:
        sections += ['PREVIOUS_EVALUATIONS:', json.dumps(history, indent=2)]
    # END UNMODIFIED group-standard body.
    if chemical_info is not None:
        if not isinstance(chemical_info, dict):
            raise TypeError('chemical_info must be a dictionary')
        required = {'molecule', 'geometry_angstrom', 'basis', 'charge', 'multiplicity',
                    'active_spatial_orbitals', 'active_num_particles', 'n_qubits'}
        missing = required - set(chemical_info)
        if missing:
            raise ValueError(f'Missing chemical fields: {sorted(missing)}')
        # Separate section: do not edit the original ordering or serialization.
        sections += ['CHEMICAL_CONTEXT:', json.dumps(chemical_info, indent=2)]
    return '\n\n'.join(sections)


def build_messages(qasm, terms, history=None, disclosed_noise=None, *, chemical_info=None):
    return {
        'instructions': build_system_prompt(chemical_info=chemical_info),
        'input': build_user_prompt(qasm, terms, history, disclosed_noise,
                                   chemical_info=chemical_info),
    }
