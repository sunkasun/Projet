"""Études avancées qui ne rentrent pas dans une classe spécifique."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from market import Market
from option import Option
from parameters import PricerParameters
from tree import Tree
import datetime as dt


def _compute_gaps_vs_rate(pricer_parameters: PricerParameters,
                          market_data: Market, option_data: Option,
                          rates: np.ndarray) -> List[float]:
    """Calcule V_américain − V_européen pour chaque taux de la grille."""
    gaps: List[float] = []
    for r in rates:
        new_market = Market(market_data.s0, float(r), market_data.vol,
                            market_data.dividend, market_data.ex_div_date)
        opt_a = Option(option_data.K, option_data.maturity,
                       option_data.call_put, "american")
        v_a = Tree(pricer_parameters, new_market, opt_a).backward_pricing()
        opt_e = Option(option_data.K, option_data.maturity,
                       option_data.call_put, "european")
        v_e = Tree(pricer_parameters, new_market, opt_e).backward_pricing()
        gaps.append(v_a - v_e)
    return gaps


def _plot_gaps_vs_rate(rates: np.ndarray, gaps: List[float],
                       option_data: Option,
                       steps_used: int) -> plt.Figure:
    """Construit la figure prime d'exercice vs taux."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rates * 100, gaps, marker='o', color='steelblue',
            linewidth=2, markersize=5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel("Risk-free rate r (%)")
    ax.set_ylabel("American − European price")
    ax.set_title(f"Early-exercise premium vs. rate "
                 f"({option_data.call_put}, K={option_data.K}, "
                 f"{steps_used} steps)")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    return fig


def _sweep_gap_vs_rate(pricer_parameters: PricerParameters,
                       market_data: Market, option_data: Option,
                       rate_min: float, rate_max: float, nb_points: int,
                       nb_steps_for_study: int) -> Tuple[np.ndarray,
                                                          List[float], int]:
    """Balaye les taux et retourne (rates, gaps, steps_used)."""
    rates = np.linspace(rate_min, rate_max, nb_points)
    original_steps = pricer_parameters.nb_steps
    try:
        # --- Forçage du nombre de pas pour l'étude ---
        if nb_steps_for_study is not None:
            pricer_parameters.nb_steps = int(nb_steps_for_study)
        gaps = _compute_gaps_vs_rate(pricer_parameters, market_data,
                                     option_data, rates)
    finally:
        pricer_parameters.nb_steps = original_steps
    if nb_steps_for_study is not None:
        steps_used = nb_steps_for_study
    else:
        steps_used = pricer_parameters.nb_steps
    return rates, gaps, steps_used


def american_european_gap_vs_rate(
        pricer_parameters: PricerParameters,
        market_data: Market,
        option_data: Option,
        rate_min: float,
        rate_max: float,
        nb_points: int = 20,
        nb_steps_for_study: int = None) -> Tuple[plt.Figure, List[float],
                                                  List[float]]:
    """Balaye le taux et price l'option (américaine et européenne)."""
    rates, gaps, steps_used = _sweep_gap_vs_rate(
        pricer_parameters, market_data, option_data,
        rate_min, rate_max, nb_points, nb_steps_for_study)
    fig = _plot_gaps_vs_rate(rates, gaps, option_data, steps_used)
    return fig, list(rates), gaps


def _compute_price_no_div(pricer_parameters: PricerParameters,
                          market_data: Market,
                          option_data: Option) -> float:
    """Prix de référence sans dividende."""
    no_div_market = Market(market_data.s0, market_data.rate, market_data.vol,
                           0.0, market_data.ex_div_date)
    original_mode = pricer_parameters.tree_mode
    pricer_parameters.tree_mode = "binomial_with_div_switch"
    try:
        tree_nd = Tree(pricer_parameters, no_div_market, option_data)
        return tree_nd.backward_pricing()
    finally:
        pricer_parameters.tree_mode = original_mode


def _compute_prices_vs_exdiv(pricer_parameters: PricerParameters,
                             market_data: Market, option_data: Option,
                             day_offsets: np.ndarray,
                             end: dt.datetime) -> Tuple[List[int],
                                                          List[float],
                                                          List[float]]:
    """Boucle intérieure : balaye les dates ex-div candidates."""
    days_after: List[int] = []
    prices_switch: List[float] = []
    prices_trinom: List[float] = []
    for d in day_offsets:
        ex_div = (pricer_parameters.pricing_date
                  + dt.timedelta(days=int(d)))
        if ex_div >= end:
            break
        # --- Marché avec la date ex-div candidate ---
        market_v = Market(market_data.s0, market_data.rate,
                          market_data.vol, market_data.dividend, ex_div)
        # --- Pricing en mode switch binomial/trinomial ---
        pricer_parameters.tree_mode = "binomial_with_div_switch"
        tree_s = Tree(pricer_parameters, market_v, option_data)
        prices_switch.append(tree_s.backward_pricing())
        # --- Pricing en mode trinomial pur ---
        pricer_parameters.tree_mode = "trinomial_everywhere"
        tree_t = Tree(pricer_parameters, market_v, option_data)
        prices_trinom.append(tree_t.backward_pricing())
        days_after.append(int(d))
    return days_after, prices_switch, prices_trinom


def _plot_price_vs_exdiv(days_after: List[int],
                         prices_switch: List[float],
                         prices_trinom: List[float],
                         price_no_div: float,
                         market_data: Market,
                         pricer_parameters: PricerParameters,
                         option_data: Option) -> plt.Figure:
    """Construit la figure prix vs date ex-div."""
    fig, ax = plt.subplots(figsize=(11, 6))
    # --- Courbes principales ---
    ax.axhline(y=price_no_div, color='black', linestyle='--', linewidth=1.5,
               label=f'No dividend ({price_no_div:.4f})')
    ax.plot(days_after, prices_switch, color='steelblue', linewidth=2,
            label='Binomial with div switch')
    ax.plot(days_after, prices_trinom, color='orange', linewidth=2,
            linestyle='--', label='Trinomial everywhere')
    # --- Marqueur date ex-div courante ---
    current_offset = ((market_data.ex_div_date
                       - pricer_parameters.pricing_date).days)
    if days_after and 0 < current_offset <= max(days_after):
        ax.axvline(x=current_offset, color='red', linestyle=':',
                   linewidth=1.5,
                   label=f'Current ex-div (D+{current_offset})')
    # --- Habillage ---
    ax.set_xlabel("Days after pricing date")
    ax.set_ylabel("Option price")
    ax.set_title(f"Option price as a function of the ex-dividend date\n"
                 f"(N={pricer_parameters.nb_steps}, "
                 f"{option_data.eur_am.capitalize()} {option_data.call_put})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    return fig


def _compute_exdiv_sweep_data(pricer_parameters: PricerParameters,
                              market_data: Market, option_data: Option,
                              nb_points: int, nb_steps_for_study: int
                              ) -> Tuple[float, List[int], List[float],
                                         List[float]]:
    """Calcule le prix sans div + la grille de dates ex-div."""
    price_no_div = _compute_price_no_div(pricer_parameters, market_data,
                                         option_data)
    # --- Préparation de la grille de dates ex-div ---
    start = pricer_parameters.pricing_date + dt.timedelta(days=1)
    end = option_data.maturity
    total_days = (end - start).days
    if total_days < 1:
        raise ValueError(
            "La maturité doit être au moins 2 jours après la pricing.")
    day_offsets = np.linspace(1, total_days, nb_points, dtype=int)
    day_offsets = np.unique(day_offsets)
    # --- Calcul des prix pour chaque date ---
    days_after, prices_switch, prices_trinom = _compute_prices_vs_exdiv(
        pricer_parameters, market_data, option_data, day_offsets, end)
    return price_no_div, days_after, prices_switch, prices_trinom


def price_vs_ex_div_date(
        pricer_parameters: PricerParameters,
        market_data: Market,
        option_data: Option,
        nb_points: int = 100,
        nb_steps_for_study: int = None) -> Tuple[plt.Figure, List[int],
                                                  List[float], List[float],
                                                  List[float]]:
    """Étudie le prix de l'option en fonction de la date ex-div."""
    original_steps = pricer_parameters.nb_steps
    original_mode = pricer_parameters.tree_mode
    try:
        if nb_steps_for_study is not None:
            pricer_parameters.nb_steps = int(nb_steps_for_study)
        # --- Calcul des données ---
        (price_no_div, days_after,
         prices_switch, prices_trinom) = _compute_exdiv_sweep_data(
            pricer_parameters, market_data, option_data, nb_points,
            nb_steps_for_study)
    finally:
        pricer_parameters.nb_steps = original_steps
        pricer_parameters.tree_mode = original_mode
    # --- Construction de la figure ---
    fig = _plot_price_vs_exdiv(days_after, prices_switch, prices_trinom,
                               price_no_div, market_data,
                               pricer_parameters, option_data)
    return fig, days_after, prices_switch, prices_trinom, price_no_div