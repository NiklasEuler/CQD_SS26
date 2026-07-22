# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is the codebase for "Computational Quantum Dynamics" (SS2026, Jena, Prof. Gärttner). It holds the `Comp_Quant_Dynam` Python package plus weekly exercise sheets (`Notebooks/exerciseN.ipynb`) and their sample solutions (`Solutions/exerciseN_sol.ipynb`). New sheets are added weekly; solution notebooks depend on the library functions being implemented as the course progresses, so library modules are organized chronologically by exercise sheet (see "Architecture" below).

## Setup

```bash
pip install -e .[test]
```

Requires Python >=3.9. Core deps: numpy, scipy, matplotlib, qutip, jax, flax, optax, jupyter (see `pyproject.toml`).

## Commands

Run the full test suite:
```bash
pytest --cov
```

Run a single test file / class / test:
```bash
pytest tests/hamiltonians_test.py
pytest tests/hamiltonians_test.py::Test_HO_eigenstates_exact
pytest tests/hamiltonians_test.py::Test_HO_eigenstates_exact::test_HO_ground
```

Notebook execution tests (`tests/exercise_sheets_test.py`) are **skipped by default** — they convert each exercise/solution notebook to a script via `jupyter nbconvert` and execute it, which is slow. Enable them with:
```bash
RUN_EXERCISE_SHEET_TESTS=1 pytest tests/exercise_sheets_test.py
```
In CI this only runs when the commit message / PR body contains `[notebooks]`, or on manual `workflow_dispatch` (see `.github/workflows/python_CI.yml`).

CI runs on ubuntu/macos/windows across Python 3.9 and 3.13, and uploads coverage + test results to Codecov.

## Architecture

The package `Comp_Quant_Dynam/` (imported as `import Comp_Quant_Dynam` or per-module, e.g. `from Comp_Quant_Dynam.utility import expectation_value`) is organized by physics/numerics role, not by exercise sheet — but within each file, functions are grouped in chronological blocks labeled `###### Exercise/Solution sheet N ######`. When adding new solutions, append to the relevant module under a new such block rather than creating new files.

- **`utility.py`** — foundational, dependency-free helpers used everywhere else: grid construction (`create_xvals`, `create_tvecs`), Fourier transforms, `expectation_value`/`_check_if_sized` (accepts either a single operator or an iterable of operators, used pervasively as the "observable" argument convention across the codebase), basis-index conversions (`idx2state`/`state2idx`, `n_party_idx2state`), coherent/CSS states, Husimi-distribution plotting-data generators, entanglement entropy / partial trace, and JAX/Flax-based variational Monte Carlo (NQS) machinery for the TFIM (sheet 10: `Jastrow`, `FFNN`, `MCMC_Sampler_Metropolis_Hastings`, `grad_E_theta_MC_*`, `perform_gs_search*`).
- **`operators.py`** — builds sparse operators (scipy.sparse). Central primitive is `n_party_op_sparse(local_dims, idxs, ops)`, which embeds one or more local operators into an n-party tensor-product Hilbert space via Kronecker products; nearly every multi-particle Hamiltonian builder depends on it. Also: ladder operators (`a_operator_sparse`, `adag_operator_sparse`, `n_operator_sparse`), collective/individual spin operators (`Sx_sparse`/`Sy_sparse`/`Sz_sparse`, Pauli matrices, symmetric-subspace reductions `Sx_symm`/`Sz2_symm`), spin-1 operators for AKLT, and MPS helper routines (`get_coeff_MPS`, `build_E_mat_MPS`, `corr_func_MPS`).
- **`hamiltonians.py`** — builds concrete Hamiltonians (harmonic oscillator, coupled HOs, TFIM in several variants — collective/symmetric/individual-spin/all-to-all/tilted —, AKLT, three-level EIT system) plus their exact/analytical solutions where known, using `operators.py` and `utility.py`.
- **`unitaries.py`** — time-evolution via eigenbasis propagation (`t_evol_eigenbasis`) and exact diagonalization (`calc_expv_ED`), plus the split-step Fourier method for wavepacket propagation.
- **`integrators.py`** — numerical ODE/Schrödinger-equation integrators sharing a common stepper signature `stepper_func(y, H_mat, dt, stepper_args) -> y_next` (Euler, RK2/RKn, Crank-Nicolson, Arnoldi/Krylov, wrapped scipy ODE). `integrate_ODE` and `loop_time_step` drive these steppers over a time grid and compare convergence against exact diagonalization. Also holds the mean-field TFIM equations and (sheet 11) the Lindblad master-equation RHS.
- **`open_systems.py`** — open quantum system / master equation code (sheet 11): steady-state solving (`rho_ss` via reduced Liouvillian `tr_reduce_L`), full master-equation RHS (`ME_RHS`), EIT system operators/Liouvillian construction, non-Hermitian effective Hamiltonians, and quantum-trajectory (quantum jump) Monte Carlo (`trajectory_step`, `get_trajectory`).
- **`plotting.py`** — matplotlib/qutip-Bloch-sphere visualization helpers used from notebooks (animation callbacks, eigenstate/Husimi/Bloch-sphere plots, ED-vs-integrator comparison plots). Not covered by unit tests.

Cross-module dependency direction is roughly: `utility` → `operators` → `hamiltonians` → `unitaries`/`integrators` → `open_systems`, with `plotting` consuming from all of them. Respect this layering when adding new functions (e.g. don't import `hamiltonians` from `operators`).

## Testing conventions

- One test file per module in `tests/` (`<module>_test.py`), mirroring `Comp_Quant_Dynam/*.py`.
- Tests are grouped into `Test_<function_name>` classes containing one or more `test_*` methods, often with shared fixture-like class attributes (e.g. precomputed grids) defined directly on the class body.
- Numerical checks predominantly use `np.allclose`/`np.isclose` against either exact analytical formulas (e.g. `HO_eigenenergies_exact`, `coupled_HO_E0_exact`, `E_TFIM_individual_exact`) or independent reference implementations (e.g. `qutip`, `scipy.sparse.linalg.eigsh`), not hardcoded output snapshots.
- `tests/exercise_sheets_test.py` is a smoke test that only checks each notebook executes without raising — it does not check numerical correctness of notebook output.
