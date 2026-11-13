"""Full-rotation-block ansatz for parent-Hamiltonian ITQPP experiments.

For L entangling layers: B0 -> E0 -> B1 -> ... -> E(L-1) -> BL.
Each Bj applies RY-RZ-RY to every qubit, in that chronological order.
Each Ej applies fixed RXX(pi/2) to disjoint nearest-neighbor pairs;
edge parity alternates between layers. There are 3*n*(L+1) parameters.

The ansatz defines U(theta); hamiltonian.py constructs H=U(theta*)H0U(theta*)^dagger.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ROTATION_STYLES = ('ry-rz-ry', 'rx-rz-rx')

@dataclass(frozen=True)
class Gate:
    kind: str
    qubits: tuple[int, ...]
    angle: float | None = None
    parameter: int | None = None


def make_template(n=4, layers=2, active_per_layer=None, topology_seed=42,
                  rotation_style='ry-rz-ry'):
    """Construct a deterministic full-coverage ansatz.

    active_per_layer is a legacy compatibility argument; it must be None or n.
    The old sparsity setting is not meaningful for a full rotation-block ansatz.
    topology_seed is accepted to preserve the calling convention, but fixed
    alternating nearest-neighbor connectivity is deliberately seed-independent.
    """
    if not 2 <= n <= 200 or layers < 1:
        raise ValueError('Require 2 <= n <= 16 and layers >= 1')
    if active_per_layer not in (None, n):
        raise ValueError('Full-coverage ansatz requires active_per_layer=n (or None). '
                         'Old sparse active_per_layer settings are no longer supported.')
    if rotation_style not in ROTATION_STYLES:
        raise ValueError(f'rotation_style must be one of {ROTATION_STYLES}')
    _ = topology_seed  # fixed topology; for future randomized layouts use a new protocol
    axes = rotation_style.split('-')
    gates: list[Gate] = []
    p = 0
    for block in range(layers + 1):
        for q in range(n):
            for kind in axes:
                gates.append(Gate(kind, (q,), parameter=p))
                p += 1
        if block < layers:
            for q in range(block % 2, n - 1, 2):
                gates.append(Gate('rxx', (q, q + 1), angle=float(np.pi / 2)))
    return tuple(gates)


def parameter_names(gates):
    ids = [g.parameter for g in gates if g.parameter is not None]
    if ids != list(range(len(ids))):
        raise ValueError('Parameter indices must be consecutive and ordered')
    return [f'theta_{i:04d}' for i in ids]


def canonical_angles(theta, gates):
    v = np.asarray(theta, dtype=float)
    if v.shape != (len(parameter_names(gates)),) or not np.isfinite(v).all():
        raise ValueError(f'Expected {len(parameter_names(gates))} finite angles')
    return (v + np.pi) % (2 * np.pi) - np.pi


def initial_parameters(gates, seed=123, scale=0.3):
    if scale < 0:
        raise ValueError('scale must be nonnegative')
    return np.random.default_rng(seed).uniform(-scale, scale, len(parameter_names(gates)))


def bound_circuit(gates, n, theta):
    from qiskit import QuantumCircuit
    theta = canonical_angles(theta, gates)
    qc = QuantumCircuit(n)
    for g in gates:
        a = float(theta[g.parameter]) if g.parameter is not None else g.angle
        getattr(qc, g.kind)(a, *g.qubits)
    return qc


def parameterized_circuit(gates, n):
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    qc = QuantumCircuit(n)
    pars = [Parameter(name) for name in parameter_names(gates)]
    for g in gates:
        a = pars[g.parameter] if g.parameter is not None else g.angle
        getattr(qc, g.kind)(a, *g.qubits)
    return qc


def export_qasm3(gates, n):
    from qiskit import qasm3
    return qasm3.dumps(parameterized_circuit(gates, n))


def native_depth(gates, n):
    from qiskit import transpile
    from noise import NATIVE_BASIS
    # Nonzero test angles prevent depth artifacts from identity rotations.
    angles = np.full(len(parameter_names(gates)), 0.37)
    qc = bound_circuit(gates, n, angles)
    return transpile(qc, basis_gates=NATIVE_BASIS,
                     optimization_level=0).depth()


def gate_metadata(gates):
    return [{'kind': g.kind, 'qubits': list(g.qubits), 'angle': g.angle,
             'parameter': g.parameter} for g in gates]
