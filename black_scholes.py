"""Pricing Black & Scholes et étude de convergence."""

from option import Option
from market import Market
from parameters import PricerParameters
from tree import Tree
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple


class Black_Scholes:
    """Pricing Black & Scholes et étude de convergence."""

    def __init__(self, pricer_parameters: PricerParameters,
                 market_data: Market, option_data: Option) -> None:
        """Initialise B&S et calcule d1/d2 (ou marque le cas dégénéré)."""
        if option_data.is_american() or market_data.dividend != 0:
            raise TypeError(
                "The dividend must be zero and the option European")
        self.pricer_parameters: PricerParameters = pricer_parameters
        self.option_data: Option = option_data
        self.market_data: Market = market_data
        self.T: float = ((option_data.maturity
                          - pricer_parameters.pricing_date).days / 365)
        # Cas dégénéré (vol≈0 ou T≈0) : on évite la division par zéro.
        self._degenerate: bool = (self.market_data.vol <= 1e-10
                                   or self.T <= 1e-10
                                   or self.market_data.s0 <= 1e-10)
        if self._degenerate:
            self.d1 = 0.0
            self.d2 = 0.0
        else:
            self._compute_d1_d2()

    def _compute_d1_d2(self) -> None:
        """Calcule d1 et d2 selon la formule B&S."""
        self.d1 = ((np.log(self.market_data.s0 / self.option_data.K)
                    + (self.market_data.rate
                       + 0.5 * self.market_data.vol ** 2) * self.T)
                   / (self.market_data.vol * np.sqrt(self.T)))
        self.d2 = self.d1 - self.market_data.vol * np.sqrt(self.T)

    def pricing_BS(self) -> float:
        """Prix B&S (ou payoff déterministe actualisé en cas dégénéré)."""
        if self._degenerate:
            return self._pricing_degenerate()
        if self.option_data.call_put == "call":
            return (self.market_data.s0 * norm.cdf(self.d1)
                    - self.option_data.K * np.exp(
                        -self.market_data.rate * self.T)
                    * norm.cdf(self.d2))
        return (self.option_data.K * np.exp(
                    -self.market_data.rate * self.T)
                * norm.cdf(-self.d2)
                - self.market_data.s0 * norm.cdf(-self.d1))

    def _pricing_degenerate(self) -> float:
        """Calcule le payoff déterministe actualisé si vol ou T ≈ 0."""
        fwd = self.market_data.s0 * np.exp(self.market_data.rate * self.T)
        if self.option_data.call_put == "call":
            intrinsic = max(fwd - self.option_data.K, 0.0)
        else:
            intrinsic = max(self.option_data.K - fwd, 0.0)
        return intrinsic * np.exp(-self.market_data.rate * self.T)

    def convergence_BS(self, nb_steps_max: int,
                       method: str = "backward") -> List[float]:
        """Étude de convergence (tree − BS) × N."""
        BS_price: float = self.pricing_BS()
        original_steps: int = self.pricer_parameters.nb_steps
        try:
            differences = self._convergence_loop(
                nb_steps_max, method, BS_price)
        finally:
            self.pricer_parameters.nb_steps = original_steps
        self._plot_convergence(differences)
        return differences

    def _convergence_loop(self, nb_steps_max: int, method: str,
                          BS_price: float) -> List[float]:
        """Boucle de calcul isolée pour permettre try/finally."""
        differences: List[float] = []
        for i in range(1, nb_steps_max + 1):
            self.pricer_parameters.nb_steps = i
            tree = Tree(self.pricer_parameters, self.market_data,
                        self.option_data)
            if method == "recursive":
                lattice_price = tree.recursive_pricing()
                if lattice_price is None:
                    break
            else:
                lattice_price = tree.backward_pricing()
            differences.append((lattice_price - BS_price) * i)
        return differences

    def _plot_convergence(self, differences: List[float]) -> None:
        """Trace le graphe de convergence."""
        plt.figure()
        plt.plot(range(1, len(differences) + 1), differences)
        plt.axhline(y=0, color='red', linestyle='-')
        plt.xlabel("Number of steps")
        plt.ylabel("(Tree − B&S) × nb_steps")
        plt.title(f"Convergence of the two models for a European "
                  f"{self.option_data.call_put}")
        plt.tight_layout()

    def calculate_prices_and_slopes(self, K_min: int, K_max: int,
                                    nb_steps: Optional[int] = None
                                    ) -> Tuple[Dict[str, List[float]],
                                                Dict[str, List[float]],
                                                List[float],
                                                List[int], plt.Figure]:
        """Balaye les strikes et trace prix + pentes (tree vs B&S)."""
        original_steps = self.pricer_parameters.nb_steps
        original_K = self.option_data.K
        if nb_steps is not None:
            self.pricer_parameters.nb_steps = int(nb_steps)
        BS, tree_dict, differences, strike_range = \
            self._sweep_strikes(K_min, K_max)
        self.pricer_parameters.nb_steps = original_steps
        self.option_data.K = original_K
        fig = self.plot_slopes(BS, tree_dict, differences, strike_range)
        return BS, tree_dict, differences, strike_range, fig

    def _sweep_strikes(self, K_min: int, K_max: int
                       ) -> Tuple[Dict[str, List[float]],
                                  Dict[str, List[float]],
                                  List[float], List[int]]:
        """Boucle intérieure : construit les courbes prix et pentes."""
        BS: Dict[str, List[float]] = {"Prices BS": [], "Slope BS": []}
        tree_dict: Dict[str, List[float]] = {
            "Prices Tree": [], "Slope Tree": []}
        differences: List[float] = []
        strike_range: List[int] = list(range(int(K_min), int(K_max) + 1))
        for i, K in enumerate(strike_range):
            self.option_data.K = K
            price_BS = Black_Scholes(self.pricer_parameters,
                                     self.market_data,
                                     self.option_data).pricing_BS()
            BS["Prices BS"].append(price_BS)
            tree = Tree(self.pricer_parameters, self.market_data,
                        self.option_data)
            price_tree = tree.backward_pricing()
            tree_dict["Prices Tree"].append(price_tree)
            if i > 0:
                self._append_slopes(BS, tree_dict, strike_range, i,
                                    price_BS, price_tree)
            differences.append(price_tree - price_BS)
        return BS, tree_dict, differences, strike_range

    def _append_slopes(self, BS: Dict[str, List[float]],
                       tree_dict: Dict[str, List[float]],
                       strike_range: List[int], i: int,
                       price_BS: float, price_tree: float) -> None:
        """Calcule et ajoute les pentes discrètes."""
        delta_K = strike_range[i] - strike_range[i - 1]
        tree_dict["Slope Tree"].append(
            (price_tree - tree_dict['Prices Tree'][i - 1]) / delta_K)
        BS["Slope BS"].append((price_BS - BS['Prices BS'][i - 1]) / delta_K)

    def plot_slopes(self, BS: Dict[str, List[float]],
                    tree: Dict[str, List[float]],
                    diff: List[float],
                    strikes: List[int]) -> plt.Figure:
        """Construit la figure prix/pentes vs strike."""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(strikes, BS["Prices BS"], color='red',
                 label='Price B&S', linestyle='-')
        ax1.plot(strikes, tree["Prices Tree"], color='green',
                 label='Price trinomial tree', linestyle='-')
        ax1.set_xlabel("Strike price")
        ax1.set_ylabel("Option price", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax2 = ax1.twinx()
        self._plot_slopes_axis(ax2, BS, tree, diff, strikes)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title(f"Prices and slopes vs. strike — trinomial tree with "
                  f"{self.pricer_parameters.nb_steps} steps")
        plt.tight_layout()
        return fig

    def _plot_slopes_axis(self, ax2: plt.Axes,
                          BS: Dict[str, List[float]],
                          tree: Dict[str, List[float]],
                          diff: List[float],
                          strikes: List[int]) -> None:
        """Trace les pentes et différences sur l'axe secondaire."""
        ax2.plot(strikes[1:], BS["Slope BS"], color='red',
                 label='Slope B&S', linestyle='--')
        ax2.plot(strikes[1:], tree["Slope Tree"], color='green',
                 label='Slope trinomial tree', linestyle='--')
        ax2.plot(strikes, diff, color='purple',
                 label='Tree − B&S', linestyle=':')
        ax2.set_ylabel('Slopes and differences', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

    def max_error_curve(self, nb_steps_max: int = 10000,
                        nb_points: int = 200,
                        target_error: Optional[float] = None
                        ) -> plt.Figure:
        """Trace l'erreur théorique Gap(N) = C/N en log-log."""
        T, S0, r, sigma = (self.T, self.market_data.s0,
                           self.market_data.rate, self.market_data.vol)
        N_values = np.unique(np.round(
            np.logspace(0, np.log10(nb_steps_max), nb_points)).astype(int))
        C = (3 * S0 * np.exp(r * T)) / (8 * np.sqrt(2 * np.pi)) \
            * (sigma ** 2) / np.sqrt(np.exp(sigma ** 2 * T) - 1) * T
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(N_values, C / N_values, color='steelblue',
                  label='Theoretical gap Gap(N)')
        if target_error is not None and target_error > 0:
            self._plot_error_target(ax, C, target_error)
        self._format_error_axes(ax)
        plt.tight_layout()
        return fig

    def _plot_error_target(self, ax: plt.Axes, C: float,
                           target_error: float) -> None:
        """Trace la cible d'erreur et la ligne verticale correspondante."""
        N_required = int(np.ceil(C / target_error))
        ax.axhline(y=target_error, color='red', linestyle='--',
                   label=f'Target error = {target_error}')
        ax.axvline(x=N_required, color='green', linestyle=':',
                   label=f'Steps needed = {N_required}')
        ax.plot([N_required], [target_error], 'o', color='green',
                markersize=10)

    def _format_error_axes(self, ax: plt.Axes) -> None:
        """Formate les axes et la légende du graphe d'erreur."""
        ax.set(
            xlabel="Number of steps  N  (log scale)",
            ylabel="Absolute error  |Gap|  (log scale)",
            title="Number of steps vs. pricing error "
                  "(trinomial, European, no dividend)")
        ax.grid(True, which="both", linestyle=':', alpha=0.5)
        ax.legend()