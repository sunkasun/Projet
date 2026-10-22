"""Greeks B&S et empiriques (différences finies sur l'arbre)."""

from market import Market
from option import Option
from parameters import PricerParameters
from tree import Tree
from black_scholes import Black_Scholes
from typing import Any, Callable, List, Optional, Union, Tuple
from scipy.stats import norm
import numpy as np
import copy
import datetime as dt


class Parameters:
    """Instantané de tous les paramètres nécessaires au repricing."""

    def __init__(self, pricer_parameters: PricerParameters,
                 market_data: Market, option_data: Option) -> None:
        """Copie les paramètres depuis les objets métier."""
        # --- Copie des données métier ---
        self._copy_market(market_data)
        self._copy_option(option_data)
        self._copy_pricer(pricer_parameters)
        # --- Paramètres dérivés ---
        self._compute_delta_t()

    def _copy_market(self, market_data: Market) -> None:
        """Copie les données de marché."""
        self.s0: float = market_data.s0
        self.rate: float = market_data.rate
        self.vol: float = market_data.vol
        self.dividend: float = market_data.dividend
        self.ex_div_date: dt.datetime = market_data.ex_div_date

    def _copy_option(self, option_data: Option) -> None:
        """Copie les données de l'option."""
        self.K: float = option_data.K
        self.maturity: dt.datetime = option_data.maturity
        self.call_put: str = option_data.call_put
        self.eur_am: str = option_data.eur_am

    def _copy_pricer(self, pricer_parameters: PricerParameters) -> None:
        """Copie les paramètres du pricer."""
        self.pricing_date: dt.datetime = pricer_parameters.pricing_date
        self.nb_steps: int = pricer_parameters.nb_steps
        self.pruning: bool = pricer_parameters.pruning
        self.p_min: float = pricer_parameters.p_min
        self.tree_mode: str = getattr(
            pricer_parameters, "tree_mode", "binomial_with_div_switch")

    def _compute_delta_t(self) -> None:
        """Calcule Δt en années."""
        self.delta_t: float = (
            (self.maturity - self.pricing_date).days
            / (self.nb_steps * 365))


class OneDimDerivative:
    """Dérivée première empirique par rapport à une seule variable."""

    def __init__(self, pricing_function: Callable,
                 parameters: Parameters) -> None:
        """Stocke la fonction de pricing et les paramètres."""
        self.pricing_function: Callable = pricing_function
        self.parameters: Parameters = parameters

    def first_derivative(self, param_to_shift: str,
                         shift_value: float,
                         is_gamma: bool = False
                         ) -> Union[Tuple[float, float], float]:
        """Calcule la dérivée première (centrée ou one-sided si gamma)."""
        self.shift: float = shift_value
        modified_up = copy.deepcopy(self.parameters)
        modified_down = copy.deepcopy(self.parameters)
        setattr(modified_up, param_to_shift,
                getattr(self.parameters, param_to_shift) + self.shift)
        setattr(modified_down, param_to_shift,
                getattr(self.parameters, param_to_shift) - self.shift)
        # --- Différences finies ---
        if is_gamma:
            return self._gamma_differences(modified_up, modified_down)
        return ((self.pricing_function(modified_up)
                 - self.pricing_function(modified_down))
                / (2 * self.shift))

    def _gamma_differences(self, modified_up: Parameters,
                            modified_down: Parameters
                            ) -> Tuple[float, float]:
        """Différences one-sided pour le calcul de gamma."""
        up: float = ((self.pricing_function(modified_up)
                      - self.pricing_function(self.parameters)) / self.shift)
        down: float = ((self.pricing_function(self.parameters)
                        - self.pricing_function(modified_down)) / self.shift)
        return up, down


def price_tree_one_dim(params: Parameters) -> float:
    """Prix tree à partir d'un snapshot Parameters."""
    pricer_parameters = PricerParameters(
        params.pricing_date, params.nb_steps, params.pruning, params.p_min,
        tree_mode=params.tree_mode)
    option_data = Option(params.K, params.maturity,
                         params.call_put, params.eur_am)
    market_data = Market(params.s0, params.rate, params.vol,
                         params.dividend, params.ex_div_date)
    tree = Tree(pricer_parameters, market_data, option_data)
    return tree.backward_pricing()


class GreeksTree:
    """Greeks empiriques calculés via l'arbre (différences finies)."""

    def __init__(self, pricer_parameters: PricerParameters,
                 market_data: Market, option_data: Option) -> None:
        """Crée le snapshot des paramètres."""
        self.params: Parameters = Parameters(
            pricer_parameters, market_data, option_data)

    def delta(self, shift: float) -> float:
        """∂V/∂S0, différence centrée."""
        return OneDimDerivative(price_tree_one_dim,
                                self.params).first_derivative("s0", shift)

    def gamma(self, shift: float) -> float:
        """∂²V/∂S0², différence seconde."""
        delta_up, delta_down = OneDimDerivative(
            price_tree_one_dim, self.params
        ).first_derivative("s0", shift, True)
        return (delta_up - delta_down) / shift

    def rho(self, shift: float) -> float:
        """∂V/∂r, par 1% de taux."""
        return OneDimDerivative(price_tree_one_dim, self.params
                                ).first_derivative("rate", shift) / 100

    def vega(self, shift: float) -> float:
        """∂V/∂σ, par 1% de vol."""
        return OneDimDerivative(price_tree_one_dim, self.params
                                ).first_derivative("vol", shift) / 100

    def theta(self, shift_days: int) -> float:
        """∂V/∂t, par jour calendaire (avance la maturité)."""
        params_shocked = copy.deepcopy(self.params)
        params_shocked.maturity = (self.params.maturity
                                   - dt.timedelta(days=shift_days))
        params_shocked.delta_t = (
            (params_shocked.maturity - params_shocked.pricing_date).days
            / (params_shocked.nb_steps * 365))
        # --- Repricing sur maturité avancée ---
        v_shocked = price_tree_one_dim(params_shocked)
        v_now = price_tree_one_dim(self.params)
        return (v_shocked - v_now) / shift_days

    def vomma(self, shift_vol: float) -> float:
        """∂²V/∂σ² divisé par 100² (cohérent avec Vega)."""
        params_up = copy.deepcopy(self.params)
        params_down = copy.deepcopy(self.params)
        params_up.vol = self.params.vol + shift_vol
        params_down.vol = self.params.vol - shift_vol
        # --- Trois repricings pour la dérivée seconde ---
        v_up = price_tree_one_dim(params_up)
        v_now = price_tree_one_dim(self.params)
        v_down = price_tree_one_dim(params_down)
        return (v_up - 2 * v_now + v_down) / (shift_vol ** 2) / 10000

    def vanna(self, shift_s: float, shift_vol: float) -> float:
        """∂²V/∂S0∂σ divisé par 100 (différence croisée)."""
        p_pp = self._shifted_params(+shift_s, +shift_vol)
        p_pm = self._shifted_params(+shift_s, -shift_vol)
        p_mp = self._shifted_params(-shift_s, +shift_vol)
        p_mm = self._shifted_params(-shift_s, -shift_vol)
        # --- Quatre repricings croisés ---
        v_pp = price_tree_one_dim(p_pp)
        v_pm = price_tree_one_dim(p_pm)
        v_mp = price_tree_one_dim(p_mp)
        v_mm = price_tree_one_dim(p_mm)
        return ((v_pp - v_pm - v_mp + v_mm)
                / (4 * shift_s * shift_vol) / 100)

    def _shifted_params(self, ds: float, dv: float) -> Parameters:
        """Crée un snapshot avec s0 += ds et vol += dv."""
        p = copy.deepcopy(self.params)
        p.s0 = self.params.s0 + ds
        p.vol = self.params.vol + dv
        return p


class GreeksBS:
    """Greeks B&S en forme fermée."""

    def __init__(self, pricer_parameters: PricerParameters,
                 market_data: Market, option_data: Option) -> None:
        """Stocke l'instance B&S sous-jacente."""
        self.BS: Black_Scholes = Black_Scholes(
            pricer_parameters, market_data, option_data)

    def delta(self) -> float:
        """∂V/∂S0 (call: N(d1), put: −N(−d1))."""
        if self.BS.option_data.call_put == "call":
            return norm.cdf(self.BS.d1)
        return -norm.cdf(-self.BS.d1)

    def gamma(self) -> float:
        """∂²V/∂S0² (identique call et put)."""
        return (norm.pdf(self.BS.d1)
                / (self.BS.market_data.s0 * self.BS.market_data.vol
                   * np.sqrt(self.BS.T)))

    def vega(self) -> float:
        """∂V/∂σ, par 1% de vol."""
        return (self.BS.market_data.s0 * norm.pdf(self.BS.d1)
                * np.sqrt(self.BS.T) / 100)

    def theta(self) -> float:
        """∂V/∂t, par jour calendaire."""
        first_part = (-(self.BS.market_data.s0 * norm.pdf(self.BS.d1)
                        * self.BS.market_data.vol)
                      / (2 * np.sqrt(self.BS.T)))
        if self.BS.option_data.call_put == "call":
            return self._theta_call(first_part)
        return self._theta_put(first_part)

    def _theta_call(self, first_part: float) -> float:
        """Composante call de theta."""
        second_part = (self.BS.market_data.rate * self.BS.option_data.K
                       * np.exp(-self.BS.market_data.rate * self.BS.T)
                       * norm.cdf(self.BS.d2))
        return (first_part - second_part) / 365

    def _theta_put(self, first_part: float) -> float:
        """Composante put de theta."""
        second_part = (self.BS.market_data.rate * self.BS.option_data.K
                       * np.exp(-self.BS.market_data.rate * self.BS.T)
                       * norm.cdf(-self.BS.d2))
        return (first_part + second_part) / 365

    def rho(self) -> float:
        """∂V/∂r, par 1% de taux."""
        if self.BS.option_data.call_put == "call":
            return (self.BS.option_data.K * self.BS.T
                    * np.exp(-self.BS.market_data.rate * self.BS.T)
                    * norm.cdf(self.BS.d2)) / 100
        return -(self.BS.option_data.K * self.BS.T
                 * np.exp(-self.BS.market_data.rate * self.BS.T)
                 * norm.cdf(-self.BS.d2)) / 100

    def vanna(self) -> float:
        """∂²V/∂S0∂σ = −φ(d1)·d2/σ, par 1% de vol."""
        return (-norm.pdf(self.BS.d1) * self.BS.d2
                / self.BS.market_data.vol / 100)

    def vomma(self) -> float:
        """∂²V/∂σ² = Vega·d1·d2/σ, par 1%² de vol."""
        vega_unit = (self.BS.market_data.s0 * norm.pdf(self.BS.d1)
                     * np.sqrt(self.BS.T))
        return (vega_unit * self.BS.d1 * self.BS.d2
                / self.BS.market_data.vol / (100 ** 2))

    def compute_all_greeks(self) -> str:
        """Retourne tous les Greeks B&S sous forme de chaîne formatée."""
        results: List[str] = []
        results.append("First-order Greeks (Black & Scholes):")
        results.append(f"Delta: {self.delta()}")
        results.append(f"Gamma: {self.gamma()}")
        results.append(f"Theta: {self.theta()}")
        results.append(f"Rho:   {self.rho()}")
        results.append(f"Vega:  {self.vega()}")
        results.append("")
        results.append("Second-order Greeks:")
        results.append(f"Vanna: {self.vanna()}")
        results.append(f"Vomma: {self.vomma()}")
        return "\n".join(results)