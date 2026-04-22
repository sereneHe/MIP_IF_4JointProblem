"""MOSEK-based NCPOP solver extracted from `_ncpol_Test (1).ipynb`.

The notebook contains several `sdp.solve(solver="mosek")` snippets. The
standalone, executable path is the `NCPOLR.estimate()` solver, extracted here
with a small helper for reuse.
"""

import numpy as np
from ncpol2sdpa import SdpRelaxation, flatten, generate_operators


def solve_with_mosek(
    variables,
    objective,
    inequalities,
    level=1,
    verbose=1,
    write_files=False,
):
    """Build an SDP relaxation and solve it with MOSEK."""
    sdp = SdpRelaxation(variables=variables, verbose=verbose)
    sdp.get_relaxation(level, objective=objective, inequalities=inequalities)
    sdp.solve(solver="mosek")

    if write_files:
        sdp.write_to_file("solutions.csv")
        sdp.write_to_file("example.dat-s")
        sdp.find_solution_ranks()

    print(sdp.primal, sdp.dual, sdp.status)
    return sdp


class NCPOLRMosek(object):
    """NCPOP regressor solved through MOSEK.

    This is the executable MOSEK path from `NCPOLR.estimate()` in the notebook.
    Given arrays `X` and `Y`, it fits `f = G * X + noise` and returns the
    estimated residual/noise values when the relaxation is feasible.
    """

    def __init__(self, level=1, verbose=1):
        self.level = level
        self.verbose = verbose

    def estimate(self, X, Y, return_sdp=False):
        """Estimate residuals for `Y ~= G * X` with an SDP solved by MOSEK."""
        T = len(Y)

        # Decision variables from the notebook.
        G = generate_operators("G", n_vars=1, hermitian=True, commutative=False)[0]
        f = generate_operators("f", n_vars=T, hermitian=True, commutative=False)
        noise = generate_operators("m", n_vars=T, hermitian=True, commutative=False)
        p = generate_operators("p", n_vars=T, hermitian=True, commutative=False)

        # Objective: squared fit error plus L1-like residual penalty.
        objective = sum((Y[i] - f[i]) ** 2 for i in range(T))
        objective += 0.5 * sum(p[i] for i in range(T))

        # Constraints:
        #   f[i] = G * X[i] + noise[i]
        #   p[i] >= noise[i]
        #   p[i] >= -noise[i]
        inequalities = []
        inequalities += [f[i] - G * X[i] - noise[i] for i in range(T)]
        inequalities += [-f[i] + G * X[i] + noise[i] for i in range(T)]
        inequalities += [p[i] - noise[i] for i in range(T)]
        inequalities += [p[i] + noise[i] for i in range(T)]

        sdp = solve_with_mosek(
            variables=flatten([G, f, noise, p]),
            objective=objective,
            inequalities=inequalities,
            level=self.level,
            verbose=self.verbose,
        )

        if sdp.status == "infeasible":
            print("Cannot find feasible solution.")
            return (None, sdp) if return_sdp else None

        print("ok.")
        residuals = [sdp[noise[i]] for i in range(T)]
        print(residuals)
        return (residuals, sdp) if return_sdp else residuals


if __name__ == "__main__":
    X = np.array([1, 2, 3], dtype=float)
    Y = np.array([1, 2, 3], dtype=float)
    NCPOLRMosek().estimate(X, Y)
