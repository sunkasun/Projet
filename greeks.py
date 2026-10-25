from market import Market
from option import Option
from parameters import PricerParameters
from tree import Tree
from black_scholes import Black_Scholes
from typing import Callable
from scipy.stats import norm
import numpy as np, copy, datetime as dt
from typing import Union, Tuple, List


class Parameters:
    def __init__(self, pricer_parameters, market_data, option_data):
        # market data
        self.s0 = market_data.s0
        self.rate = market_data.rate
        self.vol = market_data.vol
        self.dividend = market_data.dividend
        self.ex_div_date = market_data.ex_div_date
        # option data
        self.K = option_data.K
        self.maturity = option_data.maturity
        self.call_put = option_data.call_put
        self.eur_am = option_data.eur_am
        self.pricing_date = pricer_parameters.pricing_date
        # pricer information
        self.nb_steps = pricer_parameters.nb_steps
        self.pruning = pricer_parameters.pruning
        self.p_min = pricer_parameters.p_min
        self.tree_mode = getattr(pricer_parameters, "tree_mode", "binomial_with_div_switch")
        self.delta_t = (self.maturity - self.pricing_date).days / (self.nb_steps * 365)


class OneDimDerivative:
    """
    Empirical first derivative with respect to a single variable.
    """
    def __init__(self, pricing_function: Callable[[object, float], float], parameters: Parameters) -> None:
        self.pricing_function: Callable[[Parameters, float], float] = pricing_function
        self.parameters: Parameters = parameters

    def first_derivative(self, param_to_shift: str,
                         shift_value: float, is_gamma: bool = False) -> Union[Tuple[float, float], float]:
        self.shift: float = shift_value
        modified_up, modified_down = copy.deepcopy(self.parameters), copy.deepcopy(self.parameters)
        setattr(modified_up, param_to_shift, getattr(self.parameters, param_to_shift) + self.shift)
        setattr(modified_down, param_to_shift, getattr(self.parameters, param_to_shift) - self.shift)
        if is_gamma:
            up: float = (self.pricing_function(modified_up) - self.pricing_function(self.parameters)) / (self.shift)
            down: float = (self.pricing_function(self.parameters) - self.pricing_function(modified_down)) / (self.shift)
            return up, down
        return (self.pricing_function(modified_up) - self.pricing_function(modified_down)) / (2 * self.shift)


def price_tree_one_dim(params: Parameters) -> float:
    pricer_parameters = PricerParameters(
        params.pricing_date, params.nb_steps, params.pruning, params.p_min,
        tree_mode=params.tree_mode,
    )
    option_data = Option(params.K, params.maturity, params.call_put, params.eur_am)
    market_data = Market(params.s0, params.rate, params.vol,
                         params.dividend, params.ex_div_date)
    tree = Tree(pricer_parameters, market_data, option_data)
    return tree.backward_pricing()


class GreeksTree:
    """
    Empirical Greeks computed via trinomial pricing (finite differences).
    """
    def __init__(self, pricer_parameters: PricerParameters, market_data: Market, option_data: Option) -> None:
        self.params: Parameters = Parameters(pricer_parameters, market_data, option_data)

    def delta(self, shift: float) -> float:
        """∂V/∂S0, centered difference."""
        return OneDimDerivative(price_tree_one_dim, self.params).first_derivative("s0", shift)

    def gamma(self, shift: float) -> float:
        """∂²V/∂S0², second-order difference."""
        delta_up, delta_down = OneDimDerivative(price_tree_one_dim, self.params).first_derivative("s0", shift, True)
        return (delta_up - delta_down) / (shift)

    def rho(self, shift: float) -> float:
        """∂V/∂r, per 1% rate change."""
        return OneDimDerivative(price_tree_one_dim, self.params).first_derivative("rate", shift) / 100

    def vega(self, shift: float) -> float:
        """∂V/∂σ, per 1% vol change."""
        return OneDimDerivative(price_tree_one_dim, self.params).first_derivative("vol", shift) / 100

    def theta(self, shift_days: int) -> float:
        """
        ∂V/∂t, per calendar day. We shift the maturity back by `shift_days`
        (advancing calendar time) and reprice. Standard convention: theta < 0
        for a typical long option.
        """
        params_shocked: Parameters = copy.deepcopy(self.params)
        params_shocked.maturity = self.params.maturity - dt.timedelta(days=shift_days)
        params_shocked.delta_t = (params_shocked.maturity - params_shocked.pricing_date).days \
                                  / (params_shocked.nb_steps * 365)
        v_shocked: float = price_tree_one_dim(params_shocked)
        v_now: float = price_tree_one_dim(self.params)
        return (v_shocked - v_now) / shift_days

    def vomma(self, shift_vol: float) -> float:
        """
        ∂²V/∂σ², second-order centered difference. Divided by 100² to be
        consistent with Vega (which is per 1% vol).
        """
        params_up: Parameters = copy.deepcopy(self.params)
        params_down: Parameters = copy.deepcopy(self.params)
        params_up.vol = self.params.vol + shift_vol
        params_down.vol = self.params.vol - shift_vol
        v_up: float = price_tree_one_dim(params_up)
        v_now: float = price_tree_one_dim(self.params)
        v_down: float = price_tree_one_dim(params_down)
        return (v_up - 2 * v_now + v_down) / (shift_vol ** 2) / 10000

    def vanna(self, shift_s: float, shift_vol: float) -> float:
        """
        ∂²V/∂S0∂σ, cross finite difference. Divided by 100 (per 1% vol).
        """
        p_pp = copy.deepcopy(self.params)  # S+, vol+
        p_pm = copy.deepcopy(self.params)  # S+, vol-
        p_mp = copy.deepcopy(self.params)  # S-, vol+
        p_mm = copy.deepcopy(self.params)  # S-, vol-
        p_pp.s0 = self.params.s0 + shift_s; p_pp.vol = self.params.vol + shift_vol
        p_pm.s0 = self.params.s0 + shift_s; p_pm.vol = self.params.vol - shift_vol
        p_mp.s0 = self.params.s0 - shift_s; p_mp.vol = self.params.vol + shift_vol
        p_mm.s0 = self.params.s0 - shift_s; p_mm.vol = self.params.vol - shift_vol
        v_pp: float = price_tree_one_dim(p_pp)
        v_pm: float = price_tree_one_dim(p_pm)
        v_mp: float = price_tree_one_dim(p_mp)
        v_mm: float = price_tree_one_dim(p_mm)
        return (v_pp - v_pm - v_mp + v_mm) / (4 * shift_s * shift_vol) / 100


class GreeksBS:
    """
    Black & Scholes closed-form Greeks.
    """
    def __init__(self, pricer_parameters: PricerParameters, market_data: Market, option_data: Option) -> None:
        self.BS: Black_Scholes = Black_Scholes(pricer_parameters, market_data, option_data)

    def delta(self) -> float:
        if self.BS.option_data.call_put == "call":
            return norm.cdf(self.BS.d1)
        return -norm.cdf(-self.BS.d1)

    def gamma(self) -> float:
        return norm.pdf(self.BS.d1) / (self.BS.market_data.s0 * self.BS.market_data.vol * np.sqrt(self.BS.T))

    def vega(self) -> float:
        return self.BS.market_data.s0 * norm.pdf(self.BS.d1) * np.sqrt(self.BS.T) / 100

    def theta(self) -> float:
        first_part: float = -(self.BS.market_data.s0 * norm.pdf(self.BS.d1)
                              * self.BS.market_data.vol) / (2 * np.sqrt(self.BS.T))
        if self.BS.option_data.call_put == "call":
            second_part: float = self.BS.market_data.rate * self.BS.option_data.K \
                * np.exp(-self.BS.market_data.rate * self.BS.T) * norm.cdf(self.BS.d2)
            return (first_part - second_part) / 365
        else:
            second_part: float = self.BS.market_data.rate * self.BS.option_data.K \
                * np.exp(-self.BS.market_data.rate * self.BS.T) * norm.cdf(-self.BS.d2)
            return (first_part + second_part) / 365

    def rho(self) -> float:
        if self.BS.option_data.call_put == "call":
            return (self.BS.option_data.K * self.BS.T
                    * np.exp(-self.BS.market_data.rate * self.BS.T) * norm.cdf(self.BS.d2)) / 100
        else:
            return -(self.BS.option_data.K * self.BS.T
                     * np.exp(-self.BS.market_data.rate * self.BS.T) * norm.cdf(-self.BS.d2)) / 100

    def vanna(self) -> float:
        """
        ∂²V/∂S0∂σ = -φ(d1) · d2 / σ.  Divided by 100 (per 1% vol).
        Same formula for calls and puts.
        """
        return -norm.pdf(self.BS.d1) * self.BS.d2 / self.BS.market_data.vol / 100

    def vomma(self) -> float:
        """
        ∂²V/∂σ² = Vega · d1 · d2 / σ.  Divided by 100² (per 1%² vol).
        Same formula for calls and puts.
        """
        vega_unit: float = self.BS.market_data.s0 * norm.pdf(self.BS.d1) * np.sqrt(self.BS.T)
        return vega_unit * self.BS.d1 * self.BS.d2 / self.BS.market_data.vol / (100 ** 2)

    def compute_all_greeks(self) -> str:
        """
        Returns all B&S Greeks as a formatted string.
        """
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
