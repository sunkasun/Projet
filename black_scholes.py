from option import Option
from market import Market
from parameters import PricerParameters
from tree import Tree
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List, Dict, Tuple


class Black_Scholes:
    """
    Black & Scholes pricing and convergence study.
    """
    def __init__(self, pricer_parameters: PricerParameters,
                 market_data: Market, option_data: Option) -> None:
        if option_data.is_american() or market_data.dividend != 0:
            raise TypeError("The dividend must be zero and the option European")
        self.pricer_parameters: PricerParameters = pricer_parameters
        self.option_data: Option = option_data
        self.market_data: Market = market_data
        self.T: float = (option_data.maturity - pricer_parameters.pricing_date).days / 365
        self.q = getattr(market_data, "continuous_yield", 0.0)

        # FIX : cas dégénéré (vol≈0 ou T≈0) → on évite la division par zéro.
        # On marque l'objet et on laisse pricing_BS() retourner l'intrinsèque.
        self._degenerate = (self.market_data.vol <= 1e-10 or self.T <= 1e-10
                            or self.market_data.s0 <= 1e-10)
        if self._degenerate:
            self.d1 = 0.0
            self.d2 = 0.0
        else:
            self.d1 = ((np.log(self.market_data.s0 / self.option_data.K) +
                        (self.market_data.rate - self.q + 0.5 * self.market_data.vol ** 2) * self.T)
                       / (self.market_data.vol * np.sqrt(self.T)))
            self.d2 = self.d1 - self.market_data.vol * np.sqrt(self.T)

    def pricing_BS(self) -> float:
        # FIX : cas dégénéré → payoff déterministe actualisé
        if self._degenerate:
            fwd = self.market_data.s0 * np.exp((self.market_data.rate - self.q) * self.T)
            if self.option_data.call_put == "call":
                intrinsic = max(fwd - self.option_data.K, 0.0)
            else:
                intrinsic = max(self.option_data.K - fwd, 0.0)
            return intrinsic * np.exp(-self.market_data.rate * self.T)

        if self.option_data.call_put == "call":
            return (self.market_data.s0 * np.exp(-self.q * self.T) * norm.cdf(self.d1) 
                    - self.option_data.K * np.exp(-self.market_data.rate * self.T) * norm.cdf(self.d2))
        else:
            return (self.option_data.K * np.exp(-self.market_data.rate * self.T) * norm.cdf(-self.d2) 
                    - self.market_data.s0 * np.exp(-self.q * self.T) * norm.cdf(-self.d1))

    def convergence_BS(self, nb_steps_max: int, method: str = "backward") -> List[float]:
        differences: List[float] = []
        BS_price: float = self.pricing_BS()
        original_steps: int = self.pricer_parameters.nb_steps
        try:
            for i in range(1, nb_steps_max + 1):
                self.pricer_parameters.nb_steps = i
                tree: Tree = Tree(self.pricer_parameters, self.market_data, self.option_data)
                if method == "recursive":
                    lattice_price: float = tree.recursive_pricing()
                    if lattice_price is None:
                        print(f"Recursive pricing failed at step {i}, stopping convergence.")
                        break
                else:
                    lattice_price: float = tree.backward_pricing()
                differences.append((lattice_price - BS_price) * i)
        finally:
            self.pricer_parameters.nb_steps = original_steps

        plt.figure()
        plt.plot(range(1, len(differences) + 1), differences)
        plt.axhline(y=0, color='red', linestyle='-')
        plt.xlabel("Number of steps")
        plt.ylabel("(Tree − B&S) × nb_steps")
        plt.title(
            f"Convergence of the two models for a European {self.option_data.call_put}"
        )
        plt.tight_layout()
        return differences

    def calculate_prices_and_slopes(self, K_min: int, K_max: int, nb_steps: int = None):
        original_steps = self.pricer_parameters.nb_steps
        original_K = self.option_data.K
        if nb_steps is not None:
            self.pricer_parameters.nb_steps = int(nb_steps)

        BS: Dict[str, List[float]] = {"Prices BS": [], "Slope BS": []}
        tree_dict: Dict[str, List[float]] = {"Prices Tree": [], "Slope Tree": []}
        differences: List[float] = []
        strike_range: List[int] = list(range(int(K_min), int(K_max) + 1))

        try:
            for i, K in enumerate(strike_range):
                self.option_data.K = K
                price_BS: float = Black_Scholes(self.pricer_parameters, self.market_data,
                                                self.option_data).pricing_BS()
                BS["Prices BS"].append(price_BS)
                tree: Tree = Tree(self.pricer_parameters, self.market_data, self.option_data)
                price_tree: float = tree.backward_pricing()
                tree_dict["Prices Tree"].append(price_tree)
                if i > 0:
                    delta_K: float = K - strike_range[i - 1]
                    tree_dict["Slope Tree"].append(
                        (price_tree - tree_dict['Prices Tree'][i - 1]) / delta_K)
                    BS["Slope BS"].append(
                        (price_BS - BS['Prices BS'][i - 1]) / delta_K)
                differences.append(price_tree - price_BS)
        finally:
            self.pricer_parameters.nb_steps = original_steps
            self.option_data.K = original_K

        fig = self.plot_slopes(BS, tree_dict, differences, strike_range)
        return BS, tree_dict, differences, strike_range, fig

    def plot_slopes(self, BS: Dict[str, List[float]],
                    tree: Dict[str, List[float]], diff: List[float], strikes: List[int]):
        """
        Plots Tree vs B&S prices and slopes as functions of the strike.
        Returns the matplotlib figure (caller displays or saves it).
        """
        # Détection du mode pour le titre et la légende
        mode_str = "Binomial"
        if getattr(self.pricer_parameters, "tree_mode", "") == "trinomial_everywhere":
            mode_str = "Trinomial"
        elif self.market_data.dividend > 0:
            mode_str = "Binomial with div switch"
        
        tree_label = f"{mode_str} tree"

        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Primary axis: prices
        ax1.plot(strikes, BS["Prices BS"], color='red', label='Price B&S', linestyle='-')
        ax1.plot(strikes, tree["Prices Tree"], color='green', 
                 label=f'Price {tree_label}', linestyle='-')
        ax1.set_xlabel("Strike price")
        ax1.set_ylabel("Option price", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Secondary axis: slopes and difference
        ax2 = ax1.twinx()
        ax2.plot(strikes[1:], BS["Slope BS"], color='red', label='Slope B&S', linestyle='--')
        ax2.plot(strikes[1:], tree["Slope Tree"], color='green', 
                 label=f'Slope {tree_label}', linestyle='--')
        ax2.plot(strikes, diff, color='purple', label='Tree − B&S', linestyle=':')
        ax2.set_ylabel('Slopes and differences', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.title(
            f"Prices and slopes vs. strike — {tree_label} with "
            f"{self.pricer_parameters.nb_steps} steps"
        )
        plt.tight_layout()
        return fig

    def max_error_curve(self, nb_steps_max: int = 10000, nb_points: int = 200,
                        target_error: float = None):
        T = self.T
        S0 = self.market_data.s0
        r = self.market_data.rate
        sigma = self.market_data.vol

        N_values = np.unique(np.round(
            np.logspace(0, np.log10(nb_steps_max), nb_points)
        ).astype(int))

        C = (3 * S0 * np.exp(r * T)) / (8 * np.sqrt(2 * np.pi)) \
            * (sigma ** 2) / np.sqrt(np.exp(sigma ** 2 * T) - 1) * T
        gaps = C / N_values

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(N_values, gaps, color='steelblue', label='Theoretical gap Gap(N)')

        # FIX : on ne fait la ligne cible que si target_error > 0
        if target_error is not None and target_error > 0:
            ax.axhline(y=target_error, color='red', linestyle='--',
                       label=f'Target error = {target_error}')
            N_required = int(np.ceil(C / target_error))
            ax.axvline(x=N_required, color='green', linestyle=':',
                       label=f'Steps needed = {N_required}')
            ax.plot([N_required], [target_error], 'o', color='green', markersize=10)

        ax.set_xlabel("Number of steps  N  (log scale)")
        ax.set_ylabel("Absolute error  |Gap|  (log scale)")
        ax.set_title("Number of steps vs. pricing error (trinomial, European, no dividend)")
        ax.grid(True, which="both", linestyle=':', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        return fig
