import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from market import Market
from option import Option
from parameters import PricerParameters
from tree import Tree
import datetime as dt


def american_european_gap_vs_rate(
    pricer_parameters: PricerParameters,
    market_data: Market,
    option_data: Option,
    rate_min: float,
    rate_max: float,
    nb_points: int = 20,
    nb_steps_for_study: int = None,
) -> Tuple[plt.Figure, List[float], List[float]]:
    """
    Sweeps the risk-free rate over [rate_min, rate_max] and prices the same
    option (American vs European) on a trinomial tree at each point.

    nb_steps_for_study : si fourni, utilise ce nombre de steps pour l'étude
                         (indépendamment de pricer_parameters.nb_steps).
                         Sinon, utilise pricer_parameters.nb_steps.

    Returns (figure, rates, gaps) où gap = V_American − V_European.
    """
    rates = np.linspace(rate_min, rate_max, nb_points)
    gaps: List[float] = []
    original_steps = pricer_parameters.nb_steps
    try:
        if nb_steps_for_study is not None:
            pricer_parameters.nb_steps = int(nb_steps_for_study)

        for r in rates:
            new_market = Market(
                market_data.s0, float(r), market_data.vol,
                market_data.dividend, market_data.ex_div_date,
            )
            opt_a = Option(option_data.K, option_data.maturity,
                           option_data.call_put, "american")
            tree_a = Tree(pricer_parameters, new_market, opt_a)
            v_a = tree_a.backward_pricing()

            opt_e = Option(option_data.K, option_data.maturity,
                           option_data.call_put, "european")
            tree_e = Tree(pricer_parameters, new_market, opt_e)
            v_e = tree_e.backward_pricing()

            gaps.append(v_a - v_e)
    finally:
        pricer_parameters.nb_steps = original_steps

    steps_used = nb_steps_for_study if nb_steps_for_study is not None \
                 else pricer_parameters.nb_steps

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rates * 100, gaps, marker='o', color='steelblue',
            linewidth=2, markersize=5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel("Risk-free rate r (%)")
    ax.set_ylabel("American − European price")
    ax.set_title(
        f"Early-exercise premium vs. rate "
        f"({option_data.call_put}, K={option_data.K}, "
        f"{steps_used} steps)"
    )
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    return fig, list(rates), gaps


def price_vs_ex_div_date(
    pricer_parameters: PricerParameters,
    market_data: Market,
    option_data: Option,
    nb_points: int = 100,
    nb_steps_for_study: int = None,
) -> Tuple[plt.Figure, List[int], List[float], List[float], List[float]]:
    """
    Étudie le prix de l'option en fonction de la date ex-div.

    - Fait varier la date ex-div entre (pricing_date + 1 jour) et maturity.
    - Trace le prix pour les deux modes : binomial-with-switch et trinomial pur.
    - Ajoute une ligne de référence : prix sans dividende.
    - Un saut brutal entre deux dates proches est un signe de bug de discrétisation.

    Returns: (figure, days_after_pricing, prices_switch, prices_trinomial, price_no_div)
    """
    from market import Market as _Market
    from option import Option as _Option
    from tree import Tree

    original_steps = pricer_parameters.nb_steps
    original_mode = pricer_parameters.tree_mode
    try:
        if nb_steps_for_study is not None:
            pricer_parameters.nb_steps = int(nb_steps_for_study)

        # --- Prix sans dividende (référence) ---
        no_div_market = _Market(
            market_data.s0, market_data.rate, market_data.vol,
            0.0, market_data.ex_div_date,
        )
        pricer_parameters.tree_mode = "binomial_with_div_switch"
        tree_nd = Tree(pricer_parameters, no_div_market, option_data)
        price_no_div = tree_nd.backward_pricing()

        # --- Plage de dates ex-div : de pricing+1j à maturity ---
        start = pricer_parameters.pricing_date + dt.timedelta(days=1)
        end = option_data.maturity
        total_days = (end - start).days
        if total_days < 1:
            raise ValueError("La maturité doit être au moins 2 jours après la pricing date.")

        day_offsets = np.linspace(1, total_days, nb_points, dtype=int)
        day_offsets = np.unique(day_offsets)  # évite les doublons

        days_after: List[int] = []
        prices_switch: List[float] = []
        prices_trinom: List[float] = []

        for d in day_offsets:
            ex_div = pricer_parameters.pricing_date + dt.timedelta(days=int(d))
            if ex_div >= end:
                break

            # Nouveau market avec cette date ex-div
            market_v = _Market(
                market_data.s0, market_data.rate, market_data.vol,
                market_data.dividend, ex_div,
            )

            # Mode switch
            pricer_parameters.tree_mode = "binomial_with_div_switch"
            tree_s = Tree(pricer_parameters, market_v, option_data)
            prices_switch.append(tree_s.backward_pricing())

            # Mode trinomial pur
            pricer_parameters.tree_mode = "trinomial_everywhere"
            tree_t = Tree(pricer_parameters, market_v, option_data)
            prices_trinom.append(tree_t.backward_pricing())

            days_after.append(int(d))
    finally:
        pricer_parameters.nb_steps = original_steps
        pricer_parameters.tree_mode = original_mode

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(y=price_no_div, color='black', linestyle='--', linewidth=1.5,
               label=f'No dividend ({price_no_div:.4f})')
    ax.plot(days_after, prices_switch, color='steelblue', linewidth=2,
            label='Binomial with div switch')
    ax.plot(days_after, prices_trinom, color='orange', linewidth=2,
            linestyle='--', label='Trinomial everywhere')

    # Marqueur de la date ex-div actuelle de l'UI
    current_offset = (market_data.ex_div_date - pricer_parameters.pricing_date).days
    if 0 < current_offset <= max(days_after):
        ax.axvline(x=current_offset, color='red', linestyle=':', linewidth=1.5,
                   label=f'Current ex-div (D+{current_offset})')

    ax.set_xlabel("Days after pricing date")
    ax.set_ylabel("Option price")
    ax.set_title(f"Option price as a function of the ex-dividend date\n"
                 f"(N={pricer_parameters.nb_steps}, "
                 f"{option_data.eur_am.capitalize()} {option_data.call_put})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    return fig, days_after, prices_switch, prices_trinom, price_no_div

def greeks_profiles(
    pricer_parameters: PricerParameters,
    market_data: Market,
    option_data: Option,
    nb_points: int = 15,
) -> plt.Figure:
    """
    Trace les Greeks (Tree vs B&S) en fonction de leur paramètre principal.
    Retourne une figure 2x2.
    """
    from greeks import GreeksTree, GreeksBS, Parameters, price_tree_one_dim
    import copy

    # 1. Récupération des shifts adaptatifs
    s0 = market_data.s0
    vol = market_data.vol
    N = pricer_parameters.nb_steps
    scale = (100.0 / max(N, 10)) ** 0.25
    h_s = min(0.25 * s0, max(0.01 * s0, 0.05 * s0 * scale))
    h_vol = min(0.50 * vol, max(0.005, 0.15 * vol * scale))
    
    # Helper pour calculer les grecs sur un point
    def compute_greeks_at(mkt, opt, prm):
        gt = GreeksTree(prm, mkt, opt)
        # On doit lui passer les shifts manuellement
        g_tree = {
            "Delta": gt.delta(h_s),
            "Gamma": gt.gamma(h_s),
            "Vega": gt.vega(h_vol),
            "Vomma": gt.vomma(h_vol),
            "Rho": gt.rho(0.01),
            "Theta": gt.theta(1)
        }
        
        bs_ok = (not opt.is_american()) and mkt.dividend == 0
        g_bs = None
        if bs_ok:
            gbs = GreeksBS(prm, mkt, opt)
            g_bs = {
                "Delta": gbs.delta(), "Gamma": gbs.gamma(),
                "Vega": gbs.vega(), "Vomma": gbs.vomma(),
                "Rho": gbs.rho(), "Theta": gbs.theta()
            }
        return g_tree, g_bs

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_d, ax_v, ax_r, ax_t = axes.flatten()
    
    # --- 1. Delta & Gamma vs S0 ---
    s0_vals = np.linspace(0.8 * s0, 1.2 * s0, nb_points)
    d_t, d_b, g_t, g_b = [], [], [], []
    
    for s in s0_vals:
        m = Market(s, market_data.rate, market_data.vol, market_data.dividend, market_data.ex_div_date)
        o = Option(option_data.K, option_data.maturity, option_data.call_put, option_data.eur_am)
        gt, gb = compute_greeks_at(m, o, pricer_parameters)
        d_t.append(gt["Delta"]); g_t.append(gt["Gamma"])
        if gb: d_b.append(gb["Delta"]); g_b.append(gb["Gamma"])
        
    ax_d.plot(s0_vals, d_t, label="Delta Tree", color="steelblue", lw=2)
    ax_d.plot(s0_vals, g_t, label="Gamma Tree", color="orange", lw=2)
    if d_b:
        ax_d.plot(s0_vals, d_b, label="Delta BS", color="steelblue", ls="--")
        ax_d.plot(s0_vals, g_b, label="Gamma BS", color="orange", ls="--")
    ax_d.set_title("Delta & Gamma vs S₀")
    ax_d.legend(); ax_d.grid(True, alpha=0.3)

    # --- 2. Vega & Vomma vs Volatility ---
    vol_vals = np.linspace(0.5 * vol, 1.5 * vol, nb_points)
    v_t, v_b, vm_t, vm_b = [], [], [], []
    
    for v in vol_vals:
        m = Market(market_data.s0, market_data.rate, v, market_data.dividend, market_data.ex_div_date)
        o = Option(option_data.K, option_data.maturity, option_data.call_put, option_data.eur_am)
        gt, gb = compute_greeks_at(m, o, pricer_parameters)
        v_t.append(gt["Vega"]); vm_t.append(gt["Vomma"])
        if gb: v_b.append(gb["Vega"]); vm_b.append(gb["Vomma"])
        
    ax_v.plot(vol_vals, v_t, label="Vega Tree", color="green", lw=2)
    ax_v.plot(vol_vals, vm_t, label="Vomma Tree", color="red", lw=2)
    if v_b:
        ax_v.plot(vol_vals, v_b, label="Vega BS", color="green", ls="--")
        ax_v.plot(vol_vals, vm_b, label="Vomma BS", color="red", ls="--")
    ax_v.set_title("Vega & Vomma vs Volatility")
    ax_v.legend(); ax_v.grid(True, alpha=0.3)

    # --- 3. Rho vs Interest Rate ---
    r_vals = np.linspace(-0.05, 0.10, nb_points)
    r_t, r_b = [], []
    
    for r in r_vals:
        m = Market(market_data.s0, r, market_data.vol, market_data.dividend, market_data.ex_div_date)
        o = Option(option_data.K, option_data.maturity, option_data.call_put, option_data.eur_am)
        gt, gb = compute_greeks_at(m, o, pricer_parameters)
        r_t.append(gt["Rho"])
        if gb: r_b.append(gb["Rho"])
        
    ax_r.plot(r_vals * 100, r_t, label="Rho Tree", color="purple", lw=2)
    if r_b: ax_r.plot(r_vals * 100, r_b, label="Rho BS", color="purple", ls="--")
    ax_r.set_title("Rho vs Interest Rate (%)")
    ax_r.legend(); ax_r.grid(True, alpha=0.3)

    # --- 4. Theta vs Time to Maturity ---
    # On fait varier la maturité de 0.2*T à 1.5*T
    T_base = (option_data.maturity - pricer_parameters.pricing_date).days
    t_vals = np.linspace(max(1, int(0.2 * T_base)), int(1.5 * T_base), nb_points)
    th_t, th_b = [], []
    
    for days in t_vals:
        mat_date = pricer_parameters.pricing_date + dt.timedelta(days=int(days))
        m = Market(market_data.s0, market_data.rate, market_data.vol, market_data.dividend, market_data.ex_div_date)
        o = Option(option_data.K, mat_date, option_data.call_put, option_data.eur_am)
        gt, gb = compute_greeks_at(m, o, pricer_parameters)
        th_t.append(gt["Theta"])
        if gb: th_b.append(gb["Theta"])
        
    ax_t.plot(t_vals / 365.0, th_t, label="Theta Tree", color="brown", lw=2)
    if th_b: ax_t.plot(t_vals / 365.0, th_b, label="Theta BS", color="brown", ls="--")
    ax_t.set_title("Theta vs Time to Maturity (Years)")
    ax_t.legend(); ax_t.grid(True, alpha=0.3)

    fig.suptitle("Greeks Profiles: Tree (solid) vs Black-Scholes (dashed)", fontsize=14)
    plt.tight_layout()
    return fig
