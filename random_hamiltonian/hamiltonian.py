"""Exact Pauli conjugation, sparse parent-Hamiltonian generation and energy oracle.

H = U(theta*) [sum_i (I-Z_i)/2] U(theta*)^dagger, E_ground=0, gap=1.
Secret theta* is mostly Clifford with a configurable number of generic angles.
No dense matrices or statevectors are used to BUILD the Hamiltonian.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from circuit import make_template, parameter_names, bound_circuit, native_depth

# Pauli multiplication: A*B = phase*C.
_PRODUCTS = {
('I','I'):(1,'I'),('I','X'):(1,'X'),('I','Y'):(1,'Y'),('I','Z'):(1,'Z'),
('X','I'):(1,'X'),('Y','I'):(1,'Y'),('Z','I'):(1,'Z'),
('X','X'):(1,'I'),('Y','Y'):(1,'I'),('Z','Z'):(1,'I'),
('X','Y'):(1j,'Z'),('Y','Z'):(1j,'X'),('Z','X'):(1j,'Y'),
('Y','X'):(-1j,'Z'),('Z','Y'):(-1j,'X'),('X','Z'):(-1j,'Y')}


def product(a,b):
    phase = 1+0j
    out=[]
    for x,y in zip(a,b):
        v,p=_PRODUCTS[(x,y)]
        phase*=v
        out.append(p)
    return phase,''.join(out)


def pauli_for_gate(g, n):
    label=['I']*n
    op={'rx':'X','ry':'Y','rz':'Z','rxx':'X'}[g.kind]
    for q in g.qubits: label[n-q-1]=op
    return ''.join(label)


def conjugate_terms(terms, generator, angle, atol=1e-12):
    """Apply exp(-i angle P/2) H exp(+i angle P/2)."""
    out={}
    for label,c in terms.items():
        phase, prod=product(generator,label)
        reverse_phase,_=product(label,generator)
        if abs(phase-reverse_phase)<1e-9:
            out[label]=out.get(label,0)+c
        else:
            out[label]=out.get(label,0)+c*np.cos(angle)
            # -i P Q is Hermitian whenever P,Q anticommute.
            factor=(-1j)*phase
            out[prod]=out.get(prod,0)+c*np.sin(angle)*factor
    return {p:float(complex(c).real) for p,c in out.items()
            if abs(c)>atol and abs(complex(c).imag)<1e-8}


def construct_parent(gates,n,hidden_theta,atol=1e-11):
    hidden=np.asarray(hidden_theta,dtype=float)
    if len(hidden)!=len(parameter_names(gates)): raise ValueError('Hidden parameter count mismatch')
    identity='I'*n
    # Track each conjugated Z_i separately for an independently testable derivation.
    result={identity:n/2}
    for q in range(n):
        p=['I']*n; p[n-q-1]='Z'
        terms={''.join(p):-0.5}
        for g in gates:
            a=float(hidden[g.parameter]) if g.parameter is not None else g.angle
            terms=conjugate_terms(terms,pauli_for_gate(g,n),a,atol=atol*0.01)
        for label,c in terms.items():result[label]=result.get(label,0)+c
    return dict(sorted((p,float(c)) for p,c in result.items() if abs(c)>atol))


def descriptions(terms):
    return [{'pauli':p,'coefficient':float(c)} for p,c in sorted(terms.items())]


def to_sparse_pauli(terms):
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp.from_list(list(terms.items())).simplify(atol=1e-11)


@dataclass
class ParentProblem:
    n: int
    gates: tuple
    terms: dict
    hidden_theta: np.ndarray
    topology_seed: int
    instance_seed: int
    native_depth: int
    attempts: int
    non_clifford_count: int = 1
    non_clifford_indices: tuple[int, ...] = ()
    @property
    def H(self):return to_sparse_pauli(self.terms)
    @property
    def instance_id(self):
        import hashlib
        material=json.dumps({'n':self.n,'gates':[(g.kind,g.qubits,g.angle,g.parameter) for g in self.gates],
             'terms':descriptions(self.terms)},sort_keys=True)
        return hashlib.sha256(material.encode()).hexdigest()[:16]


CLIFFORD_ANGLES = np.array([0.0, np.pi / 2, -np.pi / 2, np.pi])


def sample_hidden_angles(n_parameters, rng, non_clifford_count=1,
                         min_clifford_distance=0.15):
    """Sample a mostly Clifford target without restricting the optimization ansatz.

    Exactly ``non_clifford_count`` entries are separated from every multiple
    of pi/2 by at least ``min_clifford_distance`` radians. This makes the
    declared count meaningful instead of accidentally drawing Clifford-like
    angles. The parameter vector remains unrestricted during optimization.
    """
    if not 0 <= non_clifford_count <= n_parameters:
        raise ValueError('non_clifford_count must lie between 0 and parameter count')
    if not 0 < min_clifford_distance < np.pi / 4:
        raise ValueError('min_clifford_distance must lie in (0, pi/4)')
    theta = rng.choice(CLIFFORD_ANGLES, size=n_parameters).astype(float)
    indices = rng.choice(n_parameters, size=non_clifford_count, replace=False)
    for idx in indices:
        # Rejection sampling terminates rapidly for this allowable interval.
        while True:
            a = float(rng.uniform(-np.pi, np.pi))
            distance = abs((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
            if distance >= min_clifford_distance:
                theta[int(idx)] = a
                break
    return theta, tuple(sorted(int(i) for i in indices))


def _all_non_clifford_angles_affect_parent(gates, n, hidden, indices, terms):
    """Reject generic angles that only contribute an unobservable state phase."""
    for index in indices:
        alternate = hidden.copy()
        alternate[index] = CLIFFORD_ANGLES[np.argmin(
            abs((alternate[index] - CLIFFORD_ANGLES + np.pi) % (2 * np.pi) - np.pi))]
        changed = construct_parent(gates, n, alternate)
        difference = sum((terms.get(p, 0.0) - changed.get(p, 0.0)) ** 2
                         for p in set(terms) | set(changed)) ** 0.5
        if difference < 1e-6:
            return False
    return True


def make_problem(n=4, layers=2, active_per_layer=None, k=16, seed=42,
                 max_depth=100, max_attempts=250, non_clifford_count=1):
    """Build a sparse parent while retaining the full variational ansatz.

    ``non_clifford_count`` controls only the secret target parameters.
    Existing positional arguments and energy-oracle interfaces are preserved.
    """
    if active_per_layer not in (None, n):
        raise ValueError('Full rotation-block ansatz requires active_per_layer=n or None')
    if k < n + 1:
        raise ValueError('K must be at least n+1 for this generator')
    if max_attempts < 1:
        raise ValueError('max_attempts must be >= 1')
    param_count = 3 * n * (layers + 1)
    if not 0 <= non_clifford_count <= param_count:
        raise ValueError(f'non_clifford_count must lie in [0, {param_count}]')
    for attempt in range(max_attempts):
        template_seed = seed + 1009 * attempt
        gates = make_template(n, layers, active_per_layer, template_seed)
        depth = native_depth(gates, n)
        if depth > max_depth:
            # Depth depends on ansatz, not the sampled hidden angles.
            raise RuntimeError(f'Native depth {depth} exceeds max_depth={max_depth}; '
                               'reduce n/layers or increase max_depth.')
        rng = np.random.default_rng(seed + 200003 + 1009 * attempt)
        hidden, indices = sample_hidden_angles(len(parameter_names(gates)), rng,
                                               non_clifford_count)
        terms = construct_parent(gates, n, hidden)
        if len(terms) <= k and _all_non_clifford_angles_affect_parent(
                gates, n, hidden, indices, terms):
            return ParentProblem(n, gates, terms, hidden, template_seed, seed,
                                 depth, attempt + 1, non_clifford_count, indices)
    raise RuntimeError(f'No instance met K={k} with non_clifford_count={non_clifford_count} '
                       f'in {max_attempts} attempts. Increase K or reduce '
                       'non_clifford_count; check the compiled depth separately.')


class EnergyOracle:
    """One full-H query per energy(); shots counts actual Pauli measurement shots.

    shots=0: exact statevector (ideal) or density matrix (noisy), purely diagnostic.
    shots>0: Pauli measurements, one circuit per non-identity term (no grouping).
    """
    def __init__(self,problem,shots=0,noise=False,seed=2026):
        from qiskit.quantum_info import SparsePauliOp
        self.problem=problem;self.shots=int(shots);self.noise=bool(noise)
        self.evaluations=0;self.total_shots=0;self.rng=np.random.default_rng(seed)
        self.terms=[(p,c) for p,c in problem.terms.items() if set(p)!={'I'}]
        if noise:
            from noise import simulator
            self.backend=simulator(seed)
        else:
            self.backend=None
        if self.shots<0 or 0<self.shots<len(self.terms):
            raise ValueError(f'Shots must be 0 or >= {len(self.terms)} (one per nonidentity term)')
    def ideal_unbilled(self,theta):
        from qiskit.quantum_info import Statevector
        return float(np.real(Statevector.from_instruction(bound_circuit(self.problem.gates,self.problem.n,theta)).expectation_value(self.problem.H)))
    def _density_unbilled(self,theta):
        from qiskit.quantum_info import DensityMatrix
        from noise import prepare_noisy_circuit
        qc=prepare_noisy_circuit(bound_circuit(self.problem.gates,self.problem.n,theta))
        qc.save_density_matrix()
        rho=DensityMatrix(self.backend.run(qc,shots=1).result().data(0)['density_matrix'])
        return float(np.real(rho.expectation_value(self.problem.H)))
    def _allocation(self):
        weights=np.abs([c for _,c in self.terms]); weights=weights/weights.sum()
        remain=self.shots-len(self.terms)
        raw=weights*remain;counts=1+np.floor(raw).astype(int)
        left=self.shots-int(counts.sum())
        if left: counts[np.argsort(-(raw-np.floor(raw)))[:left]]+=1
        return counts
    def _sample_noisy(self,theta,label,count):
        # Basis-change AFTER native noisy compilation: measurement basis rotations
        # must themselves be compiled to native gates and be subject to noise.
        qc=bound_circuit(self.problem.gates,self.problem.n,theta)
        for q,c in enumerate(reversed(label)):
            if c=='X':qc.h(q)
            elif c=='Y':qc.sdg(q);qc.h(q)
        from noise import prepare_noisy_circuit
        qc=prepare_noisy_circuit(qc)
        qc.measure_all()
        seed=int(self.rng.integers(1,2**31-1))
        results=self.backend.run(qc,shots=int(count),seed_simulator=seed).result().get_counts()
        mask=sum(1<<q for q,c in enumerate(reversed(label)) if c!='I')
        return sum((1 if (int(bits.replace(' ',''),2)&mask).bit_count()%2==0 else -1)*v
                   for bits,v in results.items())/count
    def energy(self,theta):
        from qiskit.quantum_info import Statevector,SparsePauliOp
        self.evaluations+=1
        if self.shots==0:return self._density_unbilled(theta) if self.noise else self.ideal_unbilled(theta)
        alloc=self._allocation();self.total_shots+=self.shots
        if self.noise:
            exp=[self._sample_noisy(theta,p,int(k)) for (p,_),k in zip(self.terms,alloc)]
        else:
            state=Statevector.from_instruction(bound_circuit(self.problem.gates,self.problem.n,theta))
            exp=[]
            for (p,_),k in zip(self.terms,alloc):
                expectation=float(np.clip(np.real(state.expectation_value(SparsePauliOp.from_list([(p,1)]))),-1,1))
                successes=self.rng.binomial(int(k),(1+expectation)/2)
                exp.append(2*successes/int(k)-1)
        return float(self.problem.terms.get('I'*self.problem.n,0)+sum(c*v for (_,c),v in zip(self.terms,exp)))
