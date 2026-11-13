# NeuralOptimizer

This repository accompanies the Programming Quantum Circuit Parameters with Large Language Models study. It contains the code and recorded model responses for experiments on whether a frozen, general-purpose large language model (LLM) can propose parameters for variational quantum circuits directly from symbolic descriptions of the circuit and its target Hamiltonian.

## Experiment code

- [Random Hamiltonians](random_hamiltonian/) — Random Hamiltonian generation, circuit construction, LLM prompts and inference, and the variational quantum eigensolver (VQE) baseline.
- [Parent Hamiltonians](parent_hamiltonian/) — Parent Hamiltonian and circuit construction, LLM prompts, and inference code.
- [Molecular systems](molecules/) — Code and prompts for the LiH and H₄ ground-state energy experiments.

## Saved LLM responses

The [data/](data/) directory contains the recorded LLM responses in JSON format, organized by experiment:

- [Random Hamiltonian responses](data/random_hamiltonian/)
- [Parent Hamiltonian responses](data/parent_hamiltonian/), grouped by `r`
- [Molecular responses](data/molecules/), including standard (blind) and physics-aware prompting runs

## Citation

If you use this code or the accompanying data in your research, please cite the paper.

```bibtex
@misc{farias2026programmingquantumcircuitparameters,
  title         = {Programming Quantum Circuit Parameters with Large Language Models},
  author        = {Tiago de Souza Farias and Caio Moraes Porto and Matheus da Silva Fonseca and Nicolas Armando Cabrera Carpi and Anderson de Araujo Oliveira and Celso Jorge Villas-Boas},
  year          = {2026},
  eprint        = {2611.17246},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = {https://arxiv.org/abs/2611.17246},
}
