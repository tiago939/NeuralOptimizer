"""Shared blind prompt for Hamiltonian expectation minimization.

Only circuit OpenQASM, Hamiltonian Pauli terms, optional disclosed noise model,
optional disclosed measurement shot count, and previous (theta, energy) observations
enter the model-facing user prompt.
No molecule/instance/ground-energy metadata is added here.

Keep SYSTEM and field order fixed across experiments for fair comparisons.
"""

import json

SYSTEM = '''You optimize the expectation value of a supplied Hamiltonian using a fixed parameterized quantum circuit. Choose one complete absolute numerical parameter vector in radians. Do not modify the circuit or Hamiltonian. Infer parameter order from the OpenQASM 3 input declarations in their textual order. Qiskit Pauli labels run from q[n-1] on the left to q[0] on the right. Use only the provided circuit, Hamiltonian, optional noise model, optional measurement specification, and previous angle-energy observations. Return only a JSON object with keys "theta" (numeric array) and "mechanism" (brief, testable reasoning).'''


def build_user_prompt(qasm, terms, history=None, disclosed_noise=None, disclosed_shots=None):
    """Return the model-facing prompt; compatible with the original parent benchmark.

    Args:
        qasm: Complete OpenQASM 3 source.
        terms: Sequence of {'pauli': str, 'coefficient': number} dictionaries.
        history: Sequence of {'theta': [floats], 'energy': float} observations.
        disclosed_noise: Optional noise-model dictionary; None hides the model.
        disclosed_shots: Optional integer total shots per full-H expectation; None hides it.
    """
    sections = [
        'CIRCUIT_OPENQASM_3:',
        qasm,
        'HAMILTONIAN_PAULI_TERMS:',
        json.dumps(terms, indent=2),
    ]
    if disclosed_noise is not None:
        sections += ['NOISE_MODEL:', json.dumps(disclosed_noise, indent=2)]
    if disclosed_shots is not None:
        if isinstance(disclosed_shots, bool) or not isinstance(disclosed_shots, int) or disclosed_shots <= 0:
            raise ValueError('disclosed_shots must be a positive integer')
        sections += ['MEASUREMENT_SPECIFICATION:', json.dumps({
            'shots_per_energy_evaluation': disclosed_shots}, indent=2)]
    if history:
        sections += ['PREVIOUS_EVALUATIONS:', json.dumps(history, indent=2)]
    return '\n\n'.join(sections)
