"""Étude du temps d'exécution du pricer trinomial."""

from market import Market
from option import Option
from parameters import PricerParameters
from tree import Tree
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, List, Tuple


def _collect_backward_times(pricer_parameters: PricerParameters,
                            market_data: Market, option_data: Option,
                            steps: List[int],
                            n_repeats: int) -> List[float]:
    """Mesure le temps backward pricing pour chaque N."""
    times: List[float] = []
    for n in steps:
        pricer_parameters.nb_steps = n
        runs: List[float] = []
        for _ in range(n_repeats):
            begin = time.perf_counter()
            tree = Tree(pricer_parameters, market_data, option_data)
            tree.backward_pricing()
            runs.append(time.perf_counter() - begin)
        times.append(min(runs))
    return times


def _collect_recursive_times(pricer_parameters: PricerParameters,
                             market_data: Market, option_data: Option,
                             steps: List[int],
                             n_repeats: int) -> List[float]:
    """Mesure le temps recursive pricing (NaN si RecursionError)."""
    times: List[float] = []
    for n in steps:
        pricer_parameters.nb_steps = n
        runs: List[float] = []
        failed = False
        for _ in range(n_repeats):
            begin = time.perf_counter()
            tree = Tree(pricer_parameters, market_data, option_data)
            try:
                tree.recursive_pricing()
            except (RecursionError, Exception):
                failed = True
                break
            runs.append(time.perf_counter() - begin)
        if failed or not runs:
            times.append(np.nan)
        else:
            times.append(min(runs))
    return times


def _collect_pruned_times(pricer_parameters: PricerParameters,
                          market_data: Market, option_data: Option,
                          steps: List[int],
                          n_repeats: int) -> List[float]:
    """Mesure le temps avec pruning activé."""
    times: List[float] = []
    original = pricer_parameters.pruning
    try:
        pricer_parameters.pruning = True
        for n in steps:
            pricer_parameters.nb_steps = n
            runs: List[float] = []
            for _ in range(n_repeats):
                begin = time.perf_counter()
                tree = Tree(pricer_parameters, market_data, option_data)
                tree.backward_pricing()
                runs.append(time.perf_counter() - begin)
            times.append(min(runs))
    finally:
        pricer_parameters.pruning = original
    return times


def _collect_unpruned_times(pricer_parameters: PricerParameters,
                            market_data: Market, option_data: Option,
                            steps: List[int],
                            n_repeats: int) -> List[float]:
    """Mesure le temps avec pruning désactivé."""
    times: List[float] = []
    original = pricer_parameters.pruning
    try:
        pricer_parameters.pruning = False
        for n in steps:
            pricer_parameters.nb_steps = n
            runs: List[float] = []
            for _ in range(n_repeats):
                begin = time.perf_counter()
                tree = Tree(pricer_parameters, market_data, option_data)
                tree.backward_pricing()
                runs.append(time.perf_counter() - begin)
            times.append(min(runs))
    finally:
        pricer_parameters.pruning = original
    return times


def _log_spaced_steps(max_steps: int, nb_points: int) -> List[int]:
    """Retourne une grille log-espacée d'entiers entre 1 et max_steps."""
    raw = np.unique(np.round(
        np.logspace(0, np.log10(max_steps), nb_points)).astype(int))
    return [int(s) for s in raw if 1 <= s <= max_steps]


class ExecutionTime:
    """Étude du temps d'exécution du pricer trinomial."""

    def __init__(self, pricer_parameters: PricerParameters,
                 market_data: Market, option_data: Option) -> None:
        """Stocke les paramètres."""
        self.pricer_parameters: PricerParameters = pricer_parameters
        self.market_data: Market = market_data
        self.option_data: Option = option_data
        self.nb_steps_max: int = pricer_parameters.nb_steps

    def compute_execution_time(self, pricing_function: Callable,
                               n_repeats: int = 3) -> List[float]:
        """Mesure le temps pour N = 1..nb_steps_max."""
        if isinstance(pricing_function, str):
            pricing_function = self._resolve_method(pricing_function)
        ex_time: List[float] = []
        for i in range(1, self.nb_steps_max + 1):
            runs: List[float] = []
            for _ in range(n_repeats):
                begin = time.perf_counter()
                pricing_function(self, i)
                runs.append(time.perf_counter() - begin)
            ex_time.append(min(runs))
        self._plot_exec_time(pricing_function, ex_time)
        return ex_time

    def _resolve_method(self, name: str) -> Callable:
        """Retourne la fonction wrapper selon le nom."""
        if name in ("recursive_pricing", "recursive pricing"):
            return recursive_pricing
        return backward_pricing

    def _plot_exec_time(self, pricing_function: Callable,
                        ex_time: List[float]) -> None:
        """Trace la courbe du temps d'exécution."""
        if pricing_function == backward_pricing:
            method = "backward pricing"
        else:
            method = "recursive pricing"
        if self.pricer_parameters.pruning:
            pruning_str = (f"pruning p_min = "
                           f"{self.pricer_parameters.p_min:.2e}")
        else:
            pruning_str = "no pruning"
        plt.figure()
        plt.plot(ex_time)
        plt.title(f"Execution time ({method}, {pruning_str})")
        plt.xlabel("Number of steps")
        plt.ylabel("Time (seconds)")

    def compare_methods(self, max_steps: int = 200, n_repeats: int = 3,
                        log_spaced: bool = True
                        ) -> Tuple[plt.Figure, List[int], List[float],
                                    List[float]]:
        """Compare les temps backward et recursive."""
        if log_spaced:
            steps = _log_spaced_steps(max_steps, 30)
        else:
            steps = list(range(1, max_steps + 1))
        original_steps = self.pricer_parameters.nb_steps
        try:
            t_back = _collect_backward_times(
                self.pricer_parameters, self.market_data, self.option_data,
                steps, n_repeats)
            t_rec = _collect_recursive_times(
                self.pricer_parameters, self.market_data, self.option_data,
                steps, n_repeats)
        finally:
            self.pricer_parameters.nb_steps = original_steps
        fig = self._plot_compare_methods(steps, t_back, t_rec, n_repeats)
        return fig, steps, t_back, t_rec

    def _plot_compare_methods(self, steps: List[int],
                              t_back: List[float], t_rec: List[float],
                              n_repeats: int) -> plt.Figure:
        """Trace la comparaison backward vs recursive."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, t_back, label='Backward', color='steelblue',
                linewidth=2, marker='o', markersize=4)
        t_rec_arr = np.array(t_rec, dtype=float)
        mask = ~np.isnan(t_rec_arr)
        if mask.any():
            ax.plot(np.array(steps)[mask], t_rec_arr[mask],
                    label='Recursive', color='orange',
                    linewidth=2, marker='o', markersize=4)
        ax.set_xlabel("Number of steps")
        ax.set_ylabel("Run time (s)")
        ax.set_title(f"Backward vs. recursive pricing\n"
                     f"(log-spaced, min over {n_repeats} runs per point)")
        ax.set_xscale('log')
        ax.grid(True, which='both', alpha=0.4)
        ax.legend()
        plt.tight_layout()
        return fig

    def compare_pruning(self, max_steps: int = 300, n_repeats: int = 2,
                        log_spaced: bool = True
                        ) -> Tuple[plt.Figure, List[int], List[float],
                                    List[float]]:
        """Compare les temps avec/sans pruning."""
        if log_spaced:
            steps = _log_spaced_steps(max_steps, 25)
        else:
            steps = list(range(1, max_steps + 1))
        original_steps = self.pricer_parameters.nb_steps
        try:
            t_pruned = _collect_pruned_times(
                self.pricer_parameters, self.market_data, self.option_data,
                steps, n_repeats)
            t_unpruned = _collect_unpruned_times(
                self.pricer_parameters, self.market_data, self.option_data,
                steps, n_repeats)
        finally:
            self.pricer_parameters.nb_steps = original_steps
        fig = self._plot_pruning(steps, t_pruned, t_unpruned)
        return fig, steps, t_pruned, t_unpruned

    def _plot_pruning(self, steps: List[int], t_pruned: List[float],
                      t_unpruned: List[float]) -> plt.Figure:
        """Trace le graphe linéaire avec zone remplie."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(steps, t_pruned, t_unpruned,
                        color='red', alpha=0.15,
                        label='Time saved by pruning')
        ax.plot(steps, t_pruned, label='With pruning (p_min = 1e-8)',
                color='steelblue', linewidth=2.5, marker='o', markersize=4)
        ax.plot(steps, t_unpruned, label='Without pruning',
                color='red', linewidth=2.5, marker='o', markersize=4)
        ax.set_xlabel("Number of steps (N)", fontsize=11)
        ax.set_ylabel("Run time (seconds)", fontsize=11)
        ax.set_title("Pruning impact on runtime — the larger N, the bigger "
                     "the win", fontsize=13, fontweight='bold')
        self._annotate_pruning_speedup(ax, steps, t_pruned, t_unpruned)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        plt.tight_layout()
        return fig

    def _annotate_pruning_speedup(self, ax: plt.Axes, steps: List[int],
                                  t_pruned: List[float],
                                  t_unpruned: List[float]) -> None:
        """Annule le gain observé au dernier point (si N assez grand)."""
        if not steps:
            return
        last_n, last_t_p, last_t_u = steps[-1], t_pruned[-1], t_unpruned[-1]
        if last_t_u <= last_t_p:
            return
        speedup = last_t_u / last_t_p
        ax.annotate(f'×{speedup:.1f} faster\nat N={last_n}',
                    xy=(last_n, (last_t_p + last_t_u) / 2),
                    xytext=(last_n * 0.8, last_t_u * 0.8),
                    fontsize=10, color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred',
                                    lw=1.5))


def recursive_pricing(execution_object: ExecutionTime,
                      nb_steps: int) -> float:
    """Wrapper recursive pour compute_execution_time."""
    execution_object.pricer_parameters.nb_steps = nb_steps
    tree = Tree(execution_object.pricer_parameters,
                execution_object.market_data, execution_object.option_data)
    return tree.recursive_pricing()


def backward_pricing(execution_object: ExecutionTime,
                     nb_steps: int) -> float:
    """Wrapper backward pour compute_execution_time."""
    execution_object.pricer_parameters.nb_steps = nb_steps
    tree = Tree(execution_object.pricer_parameters,
                execution_object.market_data, execution_object.option_data)
    return tree.backward_pricing()