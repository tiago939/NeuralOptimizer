"""Edit MOLECULES below, then run this file to save reproducible molecule JSONs.

Only this file needs to be edited to change atoms, geometry, charge, multiplicity,
basis, or the active space. JSON stores a portable molecular *specification*;
PySCF builds the actual Mole object from it when hamiltonian.py runs.

Default LiH: active (2 electrons, 3 spatial orbitals), to match the small
LiH benchmark; this freezes the lowest doubly occupied MO and excludes the
remaining virtuals. Default H4: full STO-3G (4 electrons, 4 orbitals).
"""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

# ---------------- USER-EDITABLE MOLECULAR DEFINITIONS ---------------------
# Cartesian coordinates are in ANGSTROM. Values are illustrative, not claimed
# experimental or equilibrium structures. Change them to your study geometry.
MOLECULES = {
    'lih': {
        'name': 'LiH',
        'atoms': [['Li', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 1.60]],
        'basis': 'sto3g',
        'charge': 0,
        'multiplicity': 1,
        'active_space': {'electrons': 2, 'spatial_orbitals': 3},
        # Optional: specify active MO indices explicitly, e.g. [1, 2, 3].
        # Omitting it lets Qiskit Nature select around the Fermi level.
    },
    'h4': {
        'name': 'H4',
        'atoms': [
            ['H', 0.0, 0.0, 0.0],
            ['H', 1.0, 0.0, 0.0],
            ['H', 0.0, 1.20, 0.20],
            ['H', 1.05, 1.20, -0.20],
        ],
        'basis': 'sto3g',
        'charge': 0,
        'multiplicity': 1,
        'active_space': None,  # Full STO-3G: 4 electrons / 4 spatial orbitals.
    },
}
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    atoms: tuple  # ((symbol, x, y, z), ...)
    basis: str
    charge: int
    multiplicity: int
    active_space: dict | None = None

    @classmethod
    def from_dict(cls, record):
        name = str(record['name']).strip()
        atoms = record['atoms']
        basis = str(record['basis']).strip().lower().replace('-', '')
        charge = record['charge']
        multiplicity = record['multiplicity']
        if not name or not isinstance(atoms, (list, tuple)) or len(atoms) < 2:
            raise ValueError('Supply a name and at least two atoms.')
        if basis != 'sto3g':
            raise ValueError('This controlled study requires STO-3G.')
        if type(charge) is not int or type(multiplicity) is not int or multiplicity != 1:
            raise ValueError('This singlet benchmark requires integer charge and multiplicity=1.')
        parsed = []
        for atom in atoms:
            if len(atom) != 4 or not isinstance(atom[0], str) or not atom[0].strip():
                raise ValueError('Atoms must be [symbol, x, y, z].')
            coords = [float(x) for x in atom[1:]]
            if not all(math.isfinite(x) for x in coords):
                raise ValueError('Atom coordinates must be finite.')
            parsed.append((atom[0].strip(), *coords))
        for i, a in enumerate(parsed):
            if any(sum((a[k] - b[k])**2 for k in (1, 2, 3)) < 1e-10
                   for b in parsed[i + 1:]):
                raise ValueError('Two atoms occupy nearly the same position.')
        space = record.get('active_space')
        if space is not None:
            if not isinstance(space, dict):
                raise ValueError('active_space must be a dict or None')
            allowed = {'electrons', 'spatial_orbitals', 'orbital_indices'}
            if set(space) - allowed:
                raise ValueError(f'Unknown active-space keys: {set(space) - allowed}')
            ne = space.get('electrons')
            no = space.get('spatial_orbitals')
            if type(ne) is not int or type(no) is not int or ne < 2 or ne % 2 or no < 1 or ne > 2 * no:
                raise ValueError('Singlet active space needs even electrons and enough orbitals')
            if 'orbital_indices' in space:
                indices = space['orbital_indices']
                if (not isinstance(indices, list) or len(indices) != no
                    or any(type(i) is not int or i < 0 for i in indices)
                    or len(set(indices)) != no):
                    raise ValueError('orbital_indices must be unique nonnegative integers')
        return cls(name, tuple(parsed), basis, charge, multiplicity, space)

    def to_dict(self):
        return {
            'name': self.name,
            'atoms': [list(atom) for atom in self.atoms],
            'basis': self.basis,
            'charge': self.charge,
            'multiplicity': self.multiplicity,
            'active_space': self.active_space,
            'units': 'angstrom',
        }

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + '\n', encoding='utf8')
        return path

    def pyscf_atom(self):
        return '; '.join(f'{sym} {x:.12f} {y:.12f} {z:.12f}' for sym, x, y, z in self.atoms)


def load_molecule(source):
    if isinstance(source, MoleculeSpec):
        return source
    if isinstance(source, dict):
        return MoleculeSpec.from_dict(source)
    data = json.loads(Path(source).read_text(encoding='utf8'))
    if data.get('units', 'angstrom').lower() != 'angstrom':
        raise ValueError('Only Angstrom coordinates are supported.')
    return MoleculeSpec.from_dict(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=(*MOLECULES.keys(), 'all'), default='all')
    parser.add_argument('--output-dir', default='molecules')
    args = parser.parse_args()
    names = MOLECULES if args.name == 'all' else (args.name,)
    for name in names:
        path = MoleculeSpec.from_dict(MOLECULES[name]).write(
            Path(args.output_dir) / f'{name}.json')
        print(f'Saved {path} (edit MOLECULES in molecule.py to change it)')


if __name__ == '__main__':
    main()
