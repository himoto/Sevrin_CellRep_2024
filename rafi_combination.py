from typing import ClassVar, Final

import numpy as np
from pysb.core import ComplexPattern, Model, Monomer, MonomerPattern, as_complex_pattern
from pysb.simulator import ScipyOdeSimulator
from tqdm import tqdm

CODI_CIDO: Final[dict[str, float]] = {
    # equilibrium dissociation constant
    "Kd_BRAF_RAFi1": 4.980079681,
    "Kd_CRAF_RAFi1": 12.06487342,
    "Kd_ARAF_RAFi1": 12.06487342,
    "Kd_BRAF_RAFi2": 4.980079681,
    "Kd_CRAF_RAFi2": 12.06487342,
    "Kd_ARAF_RAFi2": 12.06487342,
    # thermodynamic factors
    "fa": 0.01,
    "fb": 0.005,
    "g1a": 0.429206672,
    "g1b": 0.043807972,
    "g2a": 102.9515048,
    "g2b": 4.345979826,
    "g3a": 1,
    "g3b": 1,
}


class _OdeProblem(object):

    def __init__(self, model: Model, integrator_options: dict | None = None):
        self.model = model
        if integrator_options is None:
            integrator_options = {}
        integrator_options.setdefault("atol", 1e-8)
        integrator_options.setdefault("rtol", 1e-8)
        self.simulator = ScipyOdeSimulator(
            model,
            integrator="vode",
            integrator_options=integrator_options,
            compiler="cython",
        )

    def get_species_index(self, pattern: Monomer | MonomerPattern | ComplexPattern) -> int:
        """
        Return the index for a given species in the model.
        """
        pattern = as_complex_pattern(pattern)
        for i, s in enumerate(self.model.species):
            if s.is_equivalent_to(pattern):
                return i
        assert False, f"Pattern '{pattern}' does not exist in the model."

    def equilibrate(self) -> np.ndarray:
        """
        Run the model from the baseline initial conditions until equilibrium is reached.

        Returns
        -------
        initials_pre : numpy.ndarray
            Steady state level of each species in the model.
        """
        tspan_initial_eq = np.linspace(0, 100 * 3600, 101)
        initial_equilibrium = self.simulator.run(tspan=tspan_initial_eq)
        assert np.allclose(*initial_equilibrium.species[[-1, -2]].view(float).reshape(2, -1), rtol=1e-3)
        initials_pre = initial_equilibrium.species[-1]
        # Filter out noise
        initials_pre = np.where(np.abs(initials_pre) < 1e-10, 0.0, initials_pre)
        return initials_pre


class TwoRafiCombination(_OdeProblem):
    """
    Analyzing the effect of a combination of type I1/2 and type II RAF inhibitors.

    Parameters
    ----------
    model : pysb.Model
        A pysb model.
    output_observables : list[str]
        Observables as model output.
    upper_bounds : tuple[float, float]
        Upper bounds of Type I1/2  and Type II RAFi concentration.
    n_doses : int, default: 20
        Number of dose points to calculate.
    integrator_options : dict, optional
        Keyword arguments to pass to the "vode" integrator provided by SciPy.
        For details, please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
    """

    rafi_combination: ClassVar = CODI_CIDO

    def __init__(
        self,
        model: Model,
        *,
        output_observables: list[str],
        upper_bounds: tuple[float, float],
        n_doses: int = 20,
        integrator_options: dict | None = None,
    ):
        super().__init__(model, integrator_options)
        self.output_observables = output_observables
        for obs_name in output_observables:
            if obs_name not in self.model.observables.keys():
                raise ValueError(f"Observable '{obs_name}' is not defined in the model.")
        assert len(upper_bounds) == 2, "`upper_bounds` must be a tuple of two elements."
        # numpy.geomspace is used to capture the paradoxical ERK activation in the range of low RAFi concentrations.
        self.RAFi1_conc = np.append(
            0,
            np.geomspace(
                self.rafi_combination["Kd_BRAF_RAFi1"] * 1e-2,
                self.rafi_combination["Kd_BRAF_RAFi1"] * upper_bounds[0],
                n_doses - 1,
            ),
        )
        self.RAFi2_conc = np.append(
            0,
            np.geomspace(
                self.rafi_combination["Kd_BRAF_RAFi2"] * 1e-2,
                self.rafi_combination["Kd_BRAF_RAFi2"] * upper_bounds[1],
                n_doses - 1,
            ),
        )

    def save_simulation_results(self, fname: str) -> None:
        """
        Calculate combination effect of RAFi1 and RAFi2 on ERK activity.
        The simulation result is save as a .npz file.

        Parameters
        ----------
        fname : str
            Name of the file to be saved, i.e., {fname}.npz.
        """
        RAFi1_index = self.get_species_index(self.model.monomers.RAFi1(raf=None))
        RAFi2_index = self.get_species_index(self.model.monomers.RAFi2(raf=None))
        # First, equilibrate the system
        initials_pre = self.equilibrate()
        assert initials_pre.ndim == 1 and len(initials_pre) == len(self.model.species)
        # Then, add RAF inhibitors
        combination = np.empty((len(self.output_observables), len(self.RAFi1_conc), (len(self.RAFi2_conc))))
        simulation_condition = dict(**self.rafi_combination)
        simulation_tspan = np.linspace(0, 24 * 3600, 25)
        for j, dose1 in enumerate(tqdm(self.RAFi1_conc, desc="Calculating drug combination effect")):
            for k, dose2 in enumerate(self.RAFi2_conc):
                initials_pre[RAFi1_index] = dose1
                initials_pre[RAFi2_index] = dose2
                res = self.simulator.run(
                    tspan=simulation_tspan,
                    initials=initials_pre,
                    param_values=simulation_condition.copy(),
                )
                for i, obs_name in enumerate(self.output_observables):
                    combination[i, j, k] = res.observables[obs_name][-1]
        simulation_results: dict[str, np.ndarray] = {}
        for i, obs_name in enumerate(self.output_observables):
            simulation_results[obs_name] = combination[i]
        np.savez(fname, **simulation_results)
