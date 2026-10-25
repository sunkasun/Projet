from market import Market
from option import Option
from parameters import PricerParameters
from tree import Tree
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union, List, Tuple


class ExecutionTime:
    """
    Study of the execution time of the trinomial pricer.
    """
    def __init__(self, pricer_parameters: PricerParameters, market_data: Market, option_data: Option) -> None:
        self.pricer_parameters: PricerParameters = pricer_parameters
        self.market_data: Market = market_data
        self.option_data: Option = option_data
        self.nb_steps_max: int = pricer_parameters.nb_steps

    def compute_execution_time(self, pricing_function, n_repeats: int = 3) -> List[float]:
        if isinstance(pricing_function, str):
            if pricing_function in ("recursive_pricing", "recursive pricing"):
                pricing_function = recursive_pricing
            else:
                pricing_function = backward_pricing
        ex_time: List[float] = []
        for i in range(1, self.nb_steps_max + 1):
            runs: List[float] = []
            for _ in range(n_repeats):
                begin = time.perf_counter()
                pricing_function(self, i)
                runs.append(time.perf_counter() - begin)
            ex_time.append(min(runs))

        method: str = "backward pricing" if pricing_function == backward_pricing else "recursive pricing"
        if self.pricer_parameters.pruning:
            pruning_str = f"pruning p_min = {self.pricer_parameters.p_min:.2e}"
        else:
            pruning_str = "no pruning"
        plt.figure()
        plt.plot(ex_time)
        plt.title(f"Execution time ({method}, {pruning_str})")
        plt.xlabel("Number of steps")
        plt.ylabel("Time (seconds)")
        return ex_time

    def compare_methods(self, max_steps: int = 200,
                        n_repeats: int = 3,
                        log_spaced: bool = True
                        ) -> Tuple[plt.Figure, List[int], List[float], List[float]]:
        """
        Compare les temps d'exécution de backward et recursive.

        - log_spaced=True : échantillonnage log (30 points) au lieu de
          linéaire (max_steps points). Réduit fortement le coût total.
        - n_repeats : chaque point est mesuré n fois, on garde le minimum
          pour éliminer le bruit GC/scheduler.
        """
        if log_spaced:
            raw = np.unique(np.round(np.logspace(0, np.log10(max_steps), 30)).astype(int))
            steps: List[int] = [int(s) for s in raw if 1 <= s <= max_steps]
        else:
            steps = list(range(1, max_steps + 1))

        t_back: List[float] = []
        t_rec: List[float] = []

        original_steps = self.pricer_parameters.nb_steps
        try:
            for n in steps:
                setattr(self.pricer_parameters, "nb_steps", n)

                runs_back: List[float] = []
                for _ in range(n_repeats):
                    begin = time.perf_counter()
                    tree = Tree(self.pricer_parameters,
                                self.market_data, self.option_data)
                    tree.backward_pricing()
                    runs_back.append(time.perf_counter() - begin)
                t_back.append(min(runs_back))

                runs_rec: List[float] = []
                recursion_failed = False
                for _ in range(n_repeats):
                    begin = time.perf_counter()
                    tree = Tree(self.pricer_parameters,
                                self.market_data, self.option_data)
                    try:
                        tree.recursive_pricing()
                    except (RecursionError, Exception):
                        recursion_failed = True
                        break
                    runs_rec.append(time.perf_counter() - begin)
                if recursion_failed or not runs_rec:
                    t_rec.append(np.nan)
                else:
                    t_rec.append(min(runs_rec))
        finally:
            self.pricer_parameters.nb_steps = original_steps

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
        return fig, steps, t_back, t_rec

    def compare_pruning(self, max_steps: int = 300,
                        n_repeats: int = 2,
                        log_spaced: bool = True
                        ) -> Tuple[plt.Figure, List[int], List[float], List[float]]:
        """
        Compare le temps d'exécution avec et sans pruning.
        Utilise une échelle LINÉAIRE et remplit la zone entre les deux courbes
        pour rendre l'impact visuel maximal.
        """
        if log_spaced:
            raw = np.unique(np.round(np.logspace(0, np.log10(max_steps), 25)).astype(int))
            steps = [int(s) for s in raw if 1 <= s <= max_steps]
        else:
            steps = list(range(1, max_steps + 1))

        t_pruned: List[float] = []
        t_unpruned: List[float] = []

        original_steps = self.pricer_parameters.nb_steps
        original_pruning = self.pricer_parameters.pruning
        try:
            for n in steps:
                setattr(self.pricer_parameters, "nb_steps", n)

                # --- Avec pruning ---
                runs_p = []
                setattr(self.pricer_parameters, "pruning", True)
                for _ in range(n_repeats):
                    begin = time.perf_counter()
                    tree = Tree(self.pricer_parameters, self.market_data, self.option_data)
                    tree.backward_pricing()
                    runs_p.append(time.perf_counter() - begin)
                t_pruned.append(min(runs_p))

                # --- Sans pruning ---
                runs_u = []
                setattr(self.pricer_parameters, "pruning", False)
                for _ in range(n_repeats):
                    begin = time.perf_counter()
                    tree = Tree(self.pricer_parameters, self.market_data, self.option_data)
                    tree.backward_pricing()
                    runs_u.append(time.perf_counter() - begin)
                t_unpruned.append(min(runs_u))
        finally:
            setattr(self.pricer_parameters, "nb_steps", original_steps)
            setattr(self.pricer_parameters, "pruning", original_pruning)

        # --- Plot lin-lin avec zone remplie ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Remplissage de l'écart (le "temps gagné")
        ax.fill_between(steps, t_pruned, t_unpruned, 
                        color='red', alpha=0.15, 
                        label='Time saved by pruning')
        
        # Courbes
        ax.plot(steps, t_pruned, label='With pruning (p_min = 1e-8)',
                color='steelblue', linewidth=2.5, marker='o', markersize=4)
        ax.plot(steps, t_unpruned, label='Without pruning',
                color='red', linewidth=2.5, marker='o', markersize=4)
        
        ax.set_xlabel("Number of steps (N)", fontsize=11)
        ax.set_ylabel("Run time (seconds)", fontsize=11)
        ax.set_title("Pruning impact on runtime — the larger N, the bigger the win",
                     fontsize=13, fontweight='bold')
        
        # Annoter l'écart au dernier point (si N est suffisamment grand)
        if len(steps) > 0:
            last_n = steps[-1]
            last_t_p = t_pruned[-1]
            last_t_u = t_unpruned[-1]
            if last_t_u > last_t_p:
                speedup = last_t_u / last_t_p
                ax.annotate(f'×{speedup:.1f} faster\nat N={last_n}',
                            xy=(last_n, (last_t_p + last_t_u) / 2),
                            xytext=(last_n * 0.8, last_t_u * 0.8),
                            fontsize=10, color='darkred',
                            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        plt.tight_layout()
        return fig, steps, t_pruned, t_unpruned


def recursive_pricing(execution_object: ExecutionTime, nb_steps: int) -> float:
    setattr(execution_object.pricer_parameters, "nb_steps", nb_steps)
    tree = Tree(execution_object.pricer_parameters, execution_object.market_data, execution_object.option_data)
    return tree.recursive_pricing()


def backward_pricing(execution_object: ExecutionTime, nb_steps: int) -> float:
    setattr(execution_object.pricer_parameters, "nb_steps", nb_steps)
    tree = Tree(execution_object.pricer_parameters, execution_object.market_data, execution_object.option_data)
    return tree.backward_pricing()
