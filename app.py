"""Interface Streamlit pour le pricing d'options sur arbre binomial/trinomial."""

import datetime as dt
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from option import Option
from market import Market
from parameters import PricerParameters
from black_scholes import Black_Scholes
from greeks import GreeksTree, GreeksBS
from timing import ExecutionTime
from tree import Tree
from studies import american_european_gap_vs_rate, price_vs_ex_div_date


# ---------------------------------------------------------------------------
# Valeurs par défaut et constantes
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    "s0": 100.0, "rate": 0.03, "vol": 0.2, "dividend": 0.0,
    "ex_div_date": dt.date(2026, 10, 15),
    "K": 101.0, "maturity": dt.date(2027, 9, 2),
    "pricing_date": dt.date(2026, 9, 15),
    "call_put": "call", "eur_am": "european",
    "nb_steps": 100,
    "pruning": True, "p_min": 1e-9,
    "method": "backward",
    "tree_mode": "binomial_with_div_switch",
    "debug": False,
}

WAITING_MD = "*⏳ Computing...*"


# ---------------------------------------------------------------------------
# Formatage
# ---------------------------------------------------------------------------
def _fmt_time(seconds: float) -> str:
    """Formate une durée en µs, ms ou s."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"

# ---------------------------------------------------------------------------
# Initialisation de la session Streamlit
# ---------------------------------------------------------------------------
def _init_state(*args: Any) -> None:
    """Initialise session_state avec les valeurs par défaut."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for key in ("results", "tree", "excel_bytes", "debug_tree"):
        if key not in st.session_state:
            st.session_state[key] = None


# ---------------------------------------------------------------------------
# Construction des objets métier (découpé en 4 sous-fonctions)
# ---------------------------------------------------------------------------
def _build_market_from_session() -> Market:
    """Construit le Market depuis session_state."""
    return Market(
        st.session_state.s0, st.session_state.rate, st.session_state.vol,
        st.session_state.dividend,
        dt.datetime.combine(st.session_state.ex_div_date,
                            dt.datetime.min.time()),
    )


def _build_option_from_session() -> Option:
    """Construit l'Option depuis session_state."""
    return Option(
        st.session_state.K,
        dt.datetime.combine(st.session_state.maturity,
                            dt.datetime.min.time()),
        st.session_state.call_put, st.session_state.eur_am,
    )


def _build_pricer_from_session() -> PricerParameters:
    """Construit les PricerParameters depuis session_state."""
    pricer = PricerParameters(
        dt.datetime.combine(st.session_state.pricing_date,
                            dt.datetime.min.time()),
        int(st.session_state.nb_steps),
        bool(st.session_state.pruning),
        p_min=1e-9,
    )
    pricer.tree_mode = st.session_state.get(
        "tree_mode", "binomial_with_div_switch")
    return pricer


def _build_objects(*args: Any) -> Tuple[PricerParameters, Market, Option]:
    """Construit pricer, market et option depuis session_state."""
    if st.session_state.maturity <= st.session_state.pricing_date:
        st.sidebar.error("⚠️ Maturity must be after pricing date.")
        st.stop()
    return (_build_pricer_from_session(),
            _build_market_from_session(),
            _build_option_from_session())


# ---------------------------------------------------------------------------
# Déduction de N depuis l'erreur max
# ---------------------------------------------------------------------------
def _nb_steps_from_error(max_error: float, s0: float, vol: float,
                         T: float) -> int:
    """Déduit N depuis l'erreur cible Gap(N) = C/N."""
    up: float = vol ** 2 * T
    down: float = np.log(
        1 + (8 * np.sqrt(2 * np.pi) * max_error
             * np.sqrt(np.exp(vol ** 2 * T) - 1)) / (3 * s0)
    )
    return max(1, int(round(up / down)))


# ---------------------------------------------------------------------------
# Colonnes du DataFrame Greeks
# ---------------------------------------------------------------------------
def _greeks_col_tree(results: dict, names: List[str]) -> List[str]:
    """Colonne empirique (tree) du tableau Greeks."""
    return [f"{results['greeks_tree'][g]:.6f}" for g in names]


def _greeks_col_bs(results: dict, names: List[str]) -> List[str]:
    """Colonne théorique (B&S) du tableau Greeks."""
    return [f"{results['greeks_bs'][g]:.6f}" for g in names]


def _greeks_col_diff(results: dict, names: List[str]) -> List[str]:
    """Colonne différence tree − B&S du tableau Greeks."""
    return [
        f"{results['greeks_tree'][g] - results['greeks_bs'][g]:.6f}"
        for g in names]


def _greeks_dataframe(results: dict) -> pd.DataFrame:
    """Construit le DataFrame Greeks (tree vs B&S)."""
    names = ["Delta", "Gamma", "Theta", "Rho", "Vega", "Vomma", "Vanna"]
    data: Dict[str, Any] = {"Greek": names}
    has_tree = "greeks_tree" in results
    has_bs = "greeks_bs" in results
    if has_tree:
        data["Empirical (tree)"] = _greeks_col_tree(results, names)
    if has_bs:
        data["Theoretical (B&S)"] = _greeks_col_bs(results, names)
    if has_tree and has_bs:
        data["Difference (Tree - B&S)"] = _greeks_col_diff(results, names)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Snapshot et récapitulatif du run
# ---------------------------------------------------------------------------
def _snapshot_params(pricer: PricerParameters, market: Market,
                     option: Option, method: str) -> Dict[str, Any]:
    """Capture les paramètres du run pour réaffichage ultérieur."""
    return {
        "s0": market.s0, "rate": market.rate, "vol": market.vol,
        "dividend": market.dividend, "ex_div_date": market.ex_div_date,
        "K": option.K, "maturity": option.maturity,
        "call_put": option.call_put, "eur_am": option.eur_am,
        "pricing_date": pricer.pricing_date, "nb_steps": int(pricer.nb_steps),
        "pruning": bool(pricer.pruning), "tree_mode": pricer.tree_mode,
        "method": method,
    }


def _recap_parts_base(snap: Dict[str, Any]) -> List[str]:
    """Construit les éléments de base du récapitulatif."""
    T = (snap["maturity"] - snap["pricing_date"]).days / 365
    return [
        f"**S₀** = {snap['s0']:.4f}", f"**K** = {snap['K']:.4f}",
        f"**r** = {snap['rate']:.4%}", f"**σ** = {snap['vol']:.4%}",
        f"**N** = {snap['nb_steps']}",
        f"**{snap['call_put'].capitalize()} · "
        f"{snap['eur_am'].capitalize()}**",
        f"**T** = {T:.4f} y",
    ]


def _recap_parts_dividend(snap: Dict[str, Any]) -> List[str]:
    """Construit les éléments de dividende du récapitulatif."""
    if snap["dividend"] > 0:
        return [
            f"**D** = {snap['dividend']:.4f}",
            f"**ex-div** = {snap['ex_div_date']:%Y-%m-%d}",
        ]
    return ["**D** = 0 (pas de dividende)"]


def _recap_parts_tail(snap: Dict[str, Any]) -> List[str]:
    """Construit les éléments de mode/pricing du récapitulatif."""
    if snap["tree_mode"] == "trinomial_everywhere":
        mode = "Trinomial partout"
    elif snap["dividend"] == 0:
        mode = "Binomial"
    else:
        mode = "Binomial + trinomial (pas ex-div)"
    return [
        f"**Mode** = {mode}",
        f"**Pricing** = {snap['method'].capitalize()}",
        f"**Pruning** = {'ON' if snap['pruning'] else 'OFF'}",
    ]


def _render_run_recap_from_snapshot(snap: Dict[str, Any]) -> None:
    """Affiche le récapitulatif des paramètres du run."""
    parts: List[str] = []
    parts.extend(_recap_parts_base(snap))
    parts.extend(_recap_parts_dividend(snap))
    parts.extend(_recap_parts_tail(snap))
    st.info("  ·  ".join(parts))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar(*args: Any) -> None:
    """Rend tous les widgets de la sidebar."""
    _render_sidebar_market()
    _render_sidebar_option()
    _render_sidebar_pricer()


def _render_sidebar_market(*args: Any) -> None:
    """Section Market data de la sidebar."""
    st.sidebar.header("Market data")
    st.sidebar.number_input("Underlying (S0)", key="s0", min_value=1e-6,
                            format="%.4f",
                            help="Must be strictly positive.")
    st.sidebar.number_input("Risk-free rate (r)", key="rate",
                            min_value=-1.0, max_value=1.0, format="%.4f",
                            help="Bounded at [-1, +1].")
    st.sidebar.number_input("Volatility (σ)", key="vol", min_value=1e-6,
                            format="%.6f",
                            help="Must be strictly positivee.")
    st.sidebar.number_input("Discrete dividend (€)", key="dividend",
                            min_value=0.0, format="%.4f")
    st.sidebar.date_input("Ex-dividend date", key="ex_div_date",
                          help="Ignored if dividend = 0.")


def _render_sidebar_option(*args: Any) -> None:
    """Section Option data de la sidebar."""
    st.sidebar.header("Option data")
    st.sidebar.number_input("Strike price (K)", key="K", min_value=0.01,
                            format="%.4f",
                            help="Must be strictly positive.")
    st.sidebar.date_input("Maturity date", key="maturity",
                          help="Must be strictly after the pricing date.")
    st.sidebar.radio("Option type", ["call", "put"], key="call_put",
                     horizontal=True, format_func=lambda x: x.capitalize())
    st.sidebar.radio("Exercise style", ["european", "american"],
                     key="eur_am", horizontal=True,
                     format_func=lambda x: x.capitalize())


def _render_sidebar_pricer(*args: Any) -> None:
    """Section Pricer parameters de la sidebar."""
    st.sidebar.header("Pricer parameters")
    st.sidebar.date_input("Pricing date", key="pricing_date")
    _render_sidebar_step_choice()
    st.sidebar.checkbox("Enable pruning", key="pruning",
                        help="p_min = 1e-9.")
    st.sidebar.radio("Pricing method", ["backward", "recursive"],
                     key="method", horizontal=True,
                     format_func=lambda x: x.capitalize(),
                     help="Recursive limited at ~900 steps "
                          "(RecursionError).")
    st.sidebar.radio("Tree mode",
                     ["binomial_with_div_switch", "trinomial_everywhere"],
                     key="tree_mode", format_func=_format_tree_mode)


def _format_tree_mode(x: str) -> str:
    """Formate le nom du mode d'arbre pour l'affichage."""
    if x == "binomial_with_div_switch":
        return "Binomial (switch to trinomial if div)"
    return "Trinomial only"


# ---------------------------------------------------------------------------
# Sidebar : choix de N
# ---------------------------------------------------------------------------
def _render_sidebar_step_choice(*args: Any) -> None:
    """Widget de choix du nombre de pas."""
    step_choice = st.sidebar.radio(
        "Choice of number of steps",
        ["Number of steps", "Maximum error"], key="step_choice")
    if step_choice == "Number of steps":
        _render_nb_steps_input()
    else:
        _render_max_error_input()


def _render_nb_steps_input(*args: Any) -> None:
    """Champ de saisie direct du nombre de pas."""
    st.sidebar.number_input("Number of steps (N)", key="nb_steps",
                            min_value=1, max_value=10000, step=1,
                            help="≤ 900 if method = recursive.")


def _render_max_error_input(*args: Any) -> None:
    """Champ de saisie de l'erreur max → déduction de N."""
    st.sidebar.number_input("Maximum error (€)", key="max_error",
                            min_value=1e-9, value=0.01, format="%.4f")
    T = ((st.session_state.maturity - st.session_state.pricing_date).days
         / 365)
    if not _valid_error_inputs(T):
        st.sidebar.error("Inputs invalides pour calculer N.")
        return
    computed_n = _nb_steps_from_error(st.session_state.max_error,
                                      st.session_state.s0,
                                      st.session_state.vol, T)
    st.session_state.nb_steps = computed_n
    st.sidebar.info(f"→ N = **{computed_n}**")


def _valid_error_inputs(T: float) -> bool:
    """Vérifie que tous les inputs sont valides pour le calcul de N."""
    return (T > 0 and st.session_state.vol > 0
            and st.session_state.s0 > 0
            and st.session_state.max_error > 0)


# ---------------------------------------------------------------------------
# Shifts adaptatifs pour les Greeks
# ---------------------------------------------------------------------------
def _compute_adaptive_shifts(s0: float, vol: float,
                              N: int) -> Tuple[float, float]:
    """Calcule les shifts adaptatifs pour delta/gamma/vega."""
    if N < 30:
        return 0.10 * s0, 0.25 * vol
    if N < 100:
        return 0.05 * s0, 0.25 * vol
    return 0.02 * s0, 0.20 * vol


def _render_shift_info(*args: Any) -> Tuple[float, float, float, float,
                                            float, float, float, int]:
    """Calcule et affiche les shifts finis adaptatifs pour les Greeks."""
    s0 = float(st.session_state.s0)
    vol = float(st.session_state.vol)
    N = int(st.session_state.nb_steps)
    h_s, h_vol = _compute_adaptive_shifts(s0, vol, N)
    shift_rho, shift_theta_days = 0.01, 1
    st.caption(
        f"ℹ️ **Adaptive finite-difference shifts** (N={N}): "
        f"Δ = {h_s:.4f} · Γ = {h_s:.4f} · "
        f"ν = {h_vol:.4f} · Vomma = {h_vol:.4f} · "
        f"Vanna = ({h_s:.4f}, {h_vol:.4f})"
    )
    return (h_s, h_s, h_vol, h_vol, h_s, h_vol, shift_rho,
            shift_theta_days)


# ---------------------------------------------------------------------------
# Checkboxes : pricing / standard / advanced
# ---------------------------------------------------------------------------
def _render_checkboxes(ctx: Dict[str, Any]) -> Dict[str, bool]:
    """Rend les checkbox et retourne le dict de flags."""
    st.subheader("Compute options")
    c1, c2, c3 = st.columns(3)
    flags: Dict[str, bool] = {}
    with c1:
        flags.update(_render_checkboxes_pricing(ctx))
    with c2:
        flags.update(_render_checkboxes_standard(ctx))
    with c3:
        flags.update(_render_checkboxes_advanced(ctx))
    _apply_flag_constraints(flags, ctx)
    _render_flag_warnings(flags, ctx)
    return flags


def _render_checkboxes_pricing(ctx: Dict[str, Any]) -> Dict[str, bool]:
    """Groupe Pricing des checkboxes."""
    st.markdown("**Pricing**")
    return {
        "tree_price": st.checkbox("Compute price with trinomial tree",
                                  value=True),
        "bs_price": st.checkbox("Compute price with Black & Scholes",
                                disabled=not ctx["bs_available"]),
        "greeks_tree": st.checkbox(
            "Compute empirical Greeks (with the tree)"),
        "greeks_bs": st.checkbox(
            "Compute theoretical Greeks (B&S)",
            disabled=not ctx["bs_available"]),
    }


def _render_checkboxes_standard_first(ctx: Dict[str, Any]) -> Dict[str, bool]:
    """3 premières checkbox Standard studies."""
    return {
        "strike_study": st.checkbox(
            "Study strike range (Tree vs B&S)",
            disabled=not ctx["bs_available"]),
        "error_curve": st.checkbox(
            "Show error vs. steps curve",
            disabled=not ctx["bs_available"] or ctx["N"] < 30,
            help="Disabled if N < 30 (so the convergence tunnel is "
                 "visible)."),
        "exec_time": st.checkbox(
            "Study execution time", disabled=ctx["disable_sweeps"],
            help="Disabled if N < 10 or N > 1000."),
    }


def _render_checkboxes_standard_second(ctx: Dict[str, Any]) -> Dict[str, bool]:
    """3 dernières checkbox Standard studies."""
    return {
        "pruning_compare": st.checkbox(
            "Compare pruning ON vs OFF", disabled=ctx["disable_sweeps"],
            help="Disabled if N < 10 or N > 1000."),
        "conv": st.checkbox(
            "Study convergence with B&S",
            disabled=not ctx["bs_available"] or ctx["disable_sweeps"],
            help="Disabled if N < 10 or N > 1000."),
        "compare_methods": st.checkbox(
            "Backward vs. recursive comparison",
            disabled=ctx["disable_sweeps"],
            help="Disabled if N < 10 or N > 1000."),
    }


def _render_checkboxes_standard(ctx: Dict[str, Any]) -> Dict[str, bool]:
    """Groupe Standard studies des checkboxes."""
    st.markdown("**Standard studies**")
    flags = _render_checkboxes_standard_first(ctx)
    flags.update(_render_checkboxes_standard_second(ctx))
    return flags


def _render_checkboxes_advanced(ctx: Dict[str, Any]) -> Dict[str, bool]:
    """Groupe Advanced studies des checkboxes."""
    st.markdown("**Advanced studies**")
    return {
        "exercise_prob": st.checkbox(
            "Exercise probability by dates",
            disabled=not ctx["is_american"]),
        "amer_vs_euro": st.checkbox("American vs. European vs. rate"),
        "avg_value": st.checkbox("Average option value by dates"),
        "price_vs_exdiv": st.checkbox("Price vs. ex-div date"),
        "debug": st.checkbox("Debug — tree construction", key="debug"),
        "show_tree": st.checkbox("Show tree"),
    }


def _apply_flag_constraints(flags: Dict[str, bool],
                            ctx: Dict[str, Any]) -> None:
    """Force à False les flags incompatibles avec la config courante."""
    if not ctx["bs_available"]:
        flags["bs_price"] = False
        flags["greeks_bs"] = False
        flags["strike_study"] = False
        flags["conv"] = False
        flags["error_curve"] = False
    if not ctx["is_american"]:
        flags["exercise_prob"] = False


def _render_flag_warnings(flags: Dict[str, bool],
                          ctx: Dict[str, Any]) -> None:
    """Affiche les warnings sur les options indisponibles."""
    if not ctx["bs_available"]:
        st.caption("⚠️ Black & Scholes is unavailable for American "
                   "options or when a discrete dividend is present.")
    if ctx["disable_sweeps"] and (
            flags["exec_time"] or flags["pruning_compare"]
            or flags["conv"] or flags["compare_methods"]):
        st.caption(f"⚠️ Sweep studies are disabled because N={ctx['N']} "
                   "is too small (<10) or too large (>1000).")


def _render_study_captions(flags: Dict[str, bool],
                           ctx: Dict[str, Any]) -> None:
    """Affiche les captions des paramètres dérivés des études."""
    if flags["strike_study"]:
        st.caption(f"ℹ️ Strike range : **{ctx['strike_min']} → "
                   f"{ctx['strike_max']}** (K ± 50%, "
                   f"déduit de K={ctx['K']:.2f})")
    if flags["error_curve"]:
        st.caption(f"ℹ️ Target error : **{ctx['target_error']:.4e} €** "
                   f"(déduit de N={ctx['N']} via Gap(N) = C/N)")
    if flags["amer_vs_euro"]:
        st.caption(f"ℹ️ Rate range : **{ctx['rate_min']:.2%} → "
                   f"{ctx['rate_max']:.2%}** "
                   f"(±5%, {ctx['rate_points']} points)")


# ---------------------------------------------------------------------------
# Contexte
# ---------------------------------------------------------------------------
def _compute_target_error(N: int, s0: float, r: float, sigma: float,
                          T_years: float) -> float:
    """Calcule l'erreur cible théorique déduite de N."""
    if T_years <= 0 or sigma <= 0:
        return 0.01
    C = (3 * s0 * np.exp(r * T_years)) / (8 * np.sqrt(2 * np.pi)) \
        * (sigma ** 2) / np.sqrt(np.exp(sigma ** 2 * T_years) - 1) * T_years
    return C / max(N, 1)


def _context_derived_bounds(K: float) -> Tuple[int, int]:
    """Déduit les bornes de strike depuis K."""
    strike_min: int = max(1, int(K * 0.5))
    strike_max: int = max(strike_min + 1, int(K * 1.5))
    return strike_min, strike_max


def _context_build_dict(ctx_parts: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble le dict de contexte final."""
    ctx: Dict[str, Any] = dict(ctx_parts)
    ctx["disable_sweeps"] = (ctx["N"] < 10 or ctx["N"] > 1000)
    ctx["max_display"] = min(ctx["N"], 10)
    return ctx


def _read_session_context() -> Dict[str, Any]:
    """Lit les paramètres de session_state."""
    return {
        "N": st.session_state.nb_steps,
        "K": st.session_state.K,
        "s0": st.session_state.s0,
        "r": st.session_state.rate,
        "sigma": st.session_state.vol,
        "T_years": ((st.session_state.maturity
                     - st.session_state.pricing_date).days / 365),
    }


def _context_objects_dict(pricer: PricerParameters, market: Market,
                          option: Option) -> Dict[str, Any]:
    """Sous-dict des objets métier."""
    return {"pricer": pricer, "market": market, "option": option}


def _context_flags_dict(market: Market, option: Option) -> Dict[str, Any]:
    """Sous-dict des flags booléens."""
    return {
        "bs_available": (not option.is_american())
                        and market.dividend == 0,
        "is_american": option.is_american(),
    }


def _context_study_dict(strike_min: int, strike_max: int,
                        target_error: float) -> Dict[str, Any]:
    """Sous-dict des paramètres d'études."""
    return {
        "strike_min": strike_min, "strike_max": strike_max,
        "rate_min": -0.05, "rate_max": 0.05, "rate_points": 15,
        "target_error": target_error,
    }


def _prepare_context(pricer: PricerParameters, market: Market,
                     option: Option) -> Dict[str, Any]:
    """Prépare le dict de contexte partagé entre toutes les fonctions."""
    sess = _read_session_context()
    strike_min, strike_max = _context_derived_bounds(sess["K"])
    target_error = _compute_target_error(
        sess["N"], sess["s0"], sess["r"],
        sess["sigma"], sess["T_years"])
    ctx: Dict[str, Any] = dict(sess)
    ctx.update(_context_objects_dict(pricer, market, option))
    ctx.update(_context_flags_dict(market, option))
    ctx.update(_context_study_dict(strike_min, strike_max, target_error))
    return _context_build_dict(ctx)


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------
def _make_waiting_placeholder(col: Any) -> Any:
    """Crée un placeholder vide avec le loader WAITING_MD."""
    ph = col.empty()
    ph.markdown(WAITING_MD)
    return ph


def _create_placeholders(flags: Dict[str, bool],
                         ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Crée tous les conteneurs vides dans l'ordre d'affichage."""
    ph: Dict[str, Any] = {}
    _create_placeholders_prices(flags, ph)
    _create_placeholders_standard(flags, ph)
    _create_placeholders_advanced(flags, ph)
    return ph


def _create_placeholders_prices(flags: Dict[str, bool],
                                ph: Dict[str, Any]) -> None:
    """Placeholders pour prix et Greeks."""
    if flags["tree_price"] or flags["bs_price"]:
        st.markdown("**Prices**")
        pc1, pc2 = st.columns(2)
        if flags["tree_price"]:
            ph["tree_price"] = _make_waiting_placeholder(pc1)
        if flags["bs_price"]:
            ph["bs_price"] = _make_waiting_placeholder(pc2)
    if flags["greeks_tree"] or flags["greeks_bs"]:
        st.markdown("**Greeks**")
        ph["greeks"] = _make_waiting_placeholder(st)


def _create_placeholders_standard(flags: Dict[str, bool],
                                  ph: Dict[str, Any]) -> None:
    """Placeholders pour les études standard."""
    standard = [
        ("strike_study", "strike", "Prices and slopes vs. strike"),
        ("error_curve", "error_curve", "Error vs. number of steps"),
        ("exec_time", "exec", "Execution time"),
        ("pruning_compare", "pruning_compare", "Pruning effect on runtime"),
        ("conv", "conv", "Convergence towards Black & Scholes"),
        ("compare_methods", "compare", "Backward vs. recursive"),
    ]
    for flag_key, ph_key, title in standard:
        if flags[flag_key]:
            st.markdown(f"**{title}**")
            ph[ph_key] = _make_waiting_placeholder(st)


def _create_placeholders_advanced(flags: Dict[str, bool],
                                  ph: Dict[str, Any]) -> None:
    """Placeholders pour les études avancées et le debug/tree."""
    advanced = [
        ("exercise_prob", "exercise_prob", "Exercise probability by dates"),
        ("amer_vs_euro", "amer_euro", "American vs. European vs. rate"),
        ("avg_value", "avg_value", "Average option value by dates"),
        ("price_vs_exdiv", "price_vs_exdiv", "Price vs. ex-div date"),
        ("debug", "debug", "Debug — tree construction"),
        ("show_tree", "tree", "Trinomial tree"),
    ]
    for flag_key, ph_key, title in advanced:
        if flags[flag_key]:
            st.markdown(f"**{title}**")
            ph[ph_key] = _make_waiting_placeholder(st)


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------
def _prepare_snapshot_and_recap(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Prépare le snapshot et affiche le récapitulatif."""
    snap = _snapshot_params(ctx["pricer"], ctx["market"], ctx["option"],
                            st.session_state.method)
    st.session_state["_current_snapshot"] = snap
    _render_run_recap_from_snapshot(snap)
    return snap


def _run_all_computations(flags: Dict[str, bool], ph: Dict[str, Any],
                          ctx: Dict[str, Any]
                          ) -> Tuple[Dict[str, Any], Dict[str, float],
                                     Optional[Tree]]:
    """Exécute tous les calculs et remplit les placeholders."""
    results: Dict[str, Any] = {}
    timings: Dict[str, float] = {}
    tree_obj, tree_price = _build_tree(flags, ctx)
    _fill_prices(flags, ph, results, timings, tree_obj, tree_price, ctx)
    _fill_greeks(flags, ph, results, timings, ctx)
    _fill_standard_studies(flags, ph, results, timings, ctx)
    _fill_advanced_studies(flags, ph, results, timings, tree_obj, ctx)
    _fill_tree_display(flags, ph, timings, tree_obj)
    return results, timings, tree_obj


def _persist_results(results: Dict[str, Any], timings: Dict[str, float],
                     snap: Dict[str, Any],
                     tree_obj: Optional[Tree]) -> None:
    """Persiste les résultats dans session_state."""
    results["_timings"] = timings
    results["_params_snapshot"] = snap
    st.session_state.results = results
    if tree_obj is not None:
        st.session_state.debug_tree = tree_obj


def _execute_all(flags: Dict[str, bool], ctx: Dict[str, Any]) -> None:
    """Exécute tous les calculs et remplit les placeholders."""
    st.session_state.excel_bytes = None
    st.session_state.debug_tree = None
    st.divider()
    st.subheader("Results")
    snap = _prepare_snapshot_and_recap(ctx)
    ph = _create_placeholders(flags, ctx)
    results, timings, tree_obj = _run_all_computations(flags, ph, ctx)
    _persist_results(results, timings, snap, tree_obj)


def _build_tree(flags: Dict[str, bool],
                ctx: Dict[str, Any]) -> Tuple[Optional[Tree],
                                              Optional[float]]:
    """Construit l'arbre si nécessaire, sinon retourne (None, None)."""
    need = (flags["tree_price"] or flags["show_tree"]
            or flags["exercise_prob"] or flags["avg_value"]
            or flags["debug"])
    if not need:
        return None, None
    t0: float = time.perf_counter()
    tree_obj = Tree(ctx["pricer"], ctx["market"], ctx["option"],
                    debug=bool(flags["debug"]))
    if st.session_state.method == "backward":
        price = tree_obj.backward_pricing()
    else:
        price = tree_obj.recursive_pricing()
    tree_obj.price_time = time.perf_counter() - t0
    return tree_obj, price


# ---------------------------------------------------------------------------
# Remplissage des prix
# ---------------------------------------------------------------------------
def _fill_prices(flags: Dict[str, bool], ph: Dict[str, Any],
                 results: Dict[str, Any], timings: Dict[str, float],
                 tree_obj: Optional[Tree], tree_price: Optional[float],
                 ctx: Dict[str, Any]) -> None:
    """Remplit les placeholders prix tree + B&S."""
    if flags["tree_price"] and tree_obj is not None:
        _fill_tree_price(ph, results, timings, tree_obj, tree_price)
    if flags["bs_price"]:
        _fill_bs_price(ph, results, timings, ctx)


def _fill_tree_price(ph: Dict[str, Any], results: Dict[str, Any],
                     timings: Dict[str, float], tree_obj: Tree,
                     price: Optional[float]) -> None:
    """Remplit le prix de l'arbre."""
    if price is None:
        st.error("Le pricing récursif a échoué (RecursionError). "
                 "Utilise Backward.")
        return
    results["tree_price"] = price
    timings["tree_price"] = tree_obj.price_time
    with ph["tree_price"].container():
        st.metric("Trinomial tree price", f"{price:.6f}")
        st.caption(f"⏱ {_fmt_time(tree_obj.price_time)}")


def _fill_bs_price(ph: Dict[str, Any], results: Dict[str, Any],
                   timings: Dict[str, float], ctx: Dict[str, Any]) -> None:
    """Remplit le prix Black & Scholes."""
    t0: float = time.perf_counter()
    results["bs_price"] = Black_Scholes(
        ctx["pricer"], ctx["market"], ctx["option"]).pricing_BS()
    timings["bs_price"] = time.perf_counter() - t0
    with ph["bs_price"].container():
        st.metric("Black & Scholes price", f"{results['bs_price']:.6f}")
        st.caption(f"⏱ {_fmt_time(timings['bs_price'])}")


# ---------------------------------------------------------------------------
# Remplissage des Greeks
# ---------------------------------------------------------------------------
def _fill_greeks(flags: Dict[str, bool], ph: Dict[str, Any],
                 results: Dict[str, Any], timings: Dict[str, float],
                 ctx: Dict[str, Any]) -> None:
    """Remplit le tableau des Greeks."""
    if not (flags["greeks_tree"] or flags["greeks_bs"]):
        return
    if flags["greeks_tree"]:
        _fill_greeks_tree(results, timings, ctx["shifts"], ctx)
    if flags["greeks_bs"]:
        _fill_greeks_bs(results, timings, ctx)
    _render_greeks_table(ph, results, timings)


def _fill_greeks_tree(results: Dict[str, Any], timings: Dict[str, float],
                      shifts: Tuple[float, float, float, float,
                                    float, float, float, int],
                      ctx: Dict[str, Any]) -> None:
    """Calcule les Greeks empiriques via l'arbre."""
    t0: float = time.perf_counter()
    gt = GreeksTree(ctx["pricer"], ctx["market"], ctx["option"])
    (sd, sg, sv, svom, svs, svv, sr, std) = shifts
    results["greeks_tree"] = {
        "Delta": gt.delta(float(sd)), "Gamma": gt.gamma(float(sg)),
        "Theta": gt.theta(int(std)), "Rho": gt.rho(float(sr)),
        "Vega": gt.vega(float(sv)), "Vomma": gt.vomma(float(svom)),
        "Vanna": gt.vanna(float(svs), float(svv)),
    }
    timings["greeks_tree"] = time.perf_counter() - t0


def _fill_greeks_bs(results: Dict[str, Any], timings: Dict[str, float],
                    ctx: Dict[str, Any]) -> None:
    """Calcule les Greeks théoriques B&S."""
    t0: float = time.perf_counter()
    gbs = GreeksBS(ctx["pricer"], ctx["market"], ctx["option"])
    results["greeks_bs"] = {
        "Delta": gbs.delta(), "Gamma": gbs.gamma(), "Theta": gbs.theta(),
        "Rho": gbs.rho(), "Vega": gbs.vega(),
        "Vomma": gbs.vomma(), "Vanna": gbs.vanna(),
    }
    timings["greeks_bs"] = time.perf_counter() - t0


def _render_greeks_table(ph: Dict[str, Any], results: Dict[str, Any],
                         timings: Dict[str, float]) -> None:
    """Affiche le tableau Greeks dans son placeholder."""
    parts = [f"{k}: {_fmt_time(timings[k])}"
             for k in ("greeks_tree", "greeks_bs") if k in timings]
    with ph["greeks"].container():
        st.dataframe(_greeks_dataframe(results), hide_index=True,
                     use_container_width=True)
        if parts:
            st.caption("⏱ " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Études standard
# ---------------------------------------------------------------------------
def _fill_standard_studies(flags: Dict[str, bool], ph: Dict[str, Any],
                           results: Dict[str, Any],
                           timings: Dict[str, float],
                           ctx: Dict[str, Any]) -> None:
    """Remplit toutes les études standard."""
    if flags["strike_study"]:
        _fill_strike_study(ph, results, timings, ctx)
    if flags["error_curve"]:
        _fill_error_curve(ph, results, timings, ctx)
    if flags["exec_time"]:
        _fill_exec_time(ph, results, timings, ctx)
    if flags["pruning_compare"]:
        _fill_pruning_compare(ph, results, timings, ctx)
    if flags["conv"]:
        _fill_conv(ph, results, timings, ctx)
    if flags["compare_methods"]:
        _fill_compare_methods(ph, results, timings, ctx)


def _fill_strike_study(ph: Dict[str, Any], results: Dict[str, Any],
                       timings: Dict[str, float],
                       ctx: Dict[str, Any]) -> None:
    """Étude prix vs strike."""
    t0: float = time.perf_counter()
    _, _, _, _, fig = Black_Scholes(
        ctx["pricer"], ctx["market"], ctx["option"]
    ).calculate_prices_and_slopes(int(ctx["strike_min"]),
                                  int(ctx["strike_max"]), nb_steps=ctx["N"])
    timings["strike"] = time.perf_counter() - t0
    results["strike_fig"] = fig
    plt.close("all")
    with ph["strike"].container():
        st.pyplot(fig)
        st.caption(f"⏱ {_fmt_time(timings['strike'])}")


def _fill_error_curve(ph: Dict[str, Any], results: Dict[str, Any],
                      timings: Dict[str, float],
                      ctx: Dict[str, Any]) -> None:
    """Courbe d'erreur théorique vs N."""
    t0: float = time.perf_counter()
    fig = Black_Scholes(
        ctx["pricer"], ctx["market"], ctx["option"]
    ).max_error_curve(nb_steps_max=max(10000, ctx["N"]),
                      target_error=float(ctx["target_error"]))
    timings["error_curve"] = time.perf_counter() - t0
    results["error_curve_fig"] = fig
    plt.close("all")
    with ph["error_curve"].container():
        st.pyplot(fig)
        st.caption(f"⏱ {_fmt_time(timings['error_curve'])} "
                   "— theoretical curve")


def _fill_exec_time(ph: Dict[str, Any], results: Dict[str, Any],
                    timings: Dict[str, float], ctx: Dict[str, Any]) -> None:
    """Étude du temps d'exécution."""
    t0: float = time.perf_counter()
    ctx["pricer"].nb_steps = ctx["N"]
    if st.session_state.method == "recursive":
        method_str = "recursive_pricing"
    else:
        method_str = "backward_pricing"
    ExecutionTime(ctx["pricer"], ctx["market"], ctx["option"]
                  ).compute_execution_time(method_str)
    timings["exec"] = time.perf_counter() - t0
    results["exec_fig"] = plt.gcf()
    plt.close("all")
    with ph["exec"].container():
        st.pyplot(results["exec_fig"])
        st.caption(f"⏱ {_fmt_time(timings['exec'])}")


def _fill_pruning_compare(ph: Dict[str, Any], results: Dict[str, Any],
                          timings: Dict[str, float],
                          ctx: Dict[str, Any]) -> None:
    """Comparaison temps d'exécution avec/sans pruning."""
    t0: float = time.perf_counter()
    try:
        fig, _, _, _ = ExecutionTime(
            ctx["pricer"], ctx["market"], ctx["option"]
        ).compare_pruning(max_steps=ctx["N"], n_repeats=2, log_spaced=True)
        results["pruning_compare_fig"] = fig
        plt.close("all")
    except Exception as e:
        st.error(f"Pruning comparison failed: {e}")
    timings["pruning_compare"] = time.perf_counter() - t0
    if "pruning_compare_fig" in results:
        with ph["pruning_compare"].container():
            st.pyplot(results["pruning_compare_fig"])
            st.caption(f"⏱ {_fmt_time(timings['pruning_compare'])}")


def _fill_conv(ph: Dict[str, Any], results: Dict[str, Any],
               timings: Dict[str, float], ctx: Dict[str, Any]) -> None:
    """Étude de convergence vers B&S."""
    t0: float = time.perf_counter()
    if st.session_state.method == "recursive":
        method_str = "recursive"
    else:
        method_str = "backward"
    Black_Scholes(ctx["pricer"], ctx["market"], ctx["option"]
                  ).convergence_BS(ctx["N"], method=method_str)
    timings["conv"] = time.perf_counter() - t0
    results["conv_fig"] = plt.gcf()
    plt.close("all")
    with ph["conv"].container():
        st.pyplot(results["conv_fig"])
        st.caption(f"⏱ {_fmt_time(timings['conv'])}")


def _fill_compare_methods(ph: Dict[str, Any], results: Dict[str, Any],
                          timings: Dict[str, float],
                          ctx: Dict[str, Any]) -> None:
    """Comparaison backward vs recursive."""
    t0: float = time.perf_counter()
    fig, _, _, _ = ExecutionTime(
        ctx["pricer"], ctx["market"], ctx["option"]
    ).compare_methods(ctx["N"])
    timings["compare"] = time.perf_counter() - t0
    results["compare_fig"] = fig
    plt.close("all")
    with ph["compare"].container():
        st.pyplot(fig)
        st.caption(f"⏱ {_fmt_time(timings['compare'])}")


# ---------------------------------------------------------------------------
# Études avancées
# ---------------------------------------------------------------------------
def _fill_advanced_studies(flags: Dict[str, bool], ph: Dict[str, Any],
                           results: Dict[str, Any],
                           timings: Dict[str, float],
                           tree_obj: Optional[Tree],
                           ctx: Dict[str, Any]) -> None:
    """Remplit les études avancées."""
    if flags["exercise_prob"] and tree_obj is not None:
        _fill_exercise_prob(ph, results, timings, tree_obj)
    if flags["amer_vs_euro"]:
        _fill_amer_vs_euro(ph, results, timings, ctx)
    if flags["avg_value"] and tree_obj is not None:
        _fill_avg_value(ph, results, timings, tree_obj)
    if flags["price_vs_exdiv"]:
        _fill_price_vs_exdiv(ph, results, timings, ctx)
    if flags["debug"] and tree_obj is not None:
        _fill_debug(ph, timings, tree_obj)


def _fill_exercise_prob(ph: Dict[str, Any], results: Dict[str, Any],
                        timings: Dict[str, float], tree_obj: Tree) -> None:
    """Probabilité d'exercice anticipé par date."""
    t0: float = time.perf_counter()
    fig, _, _ = tree_obj.plot_exercise_probability_by_date()
    timings["exercise_prob"] = time.perf_counter() - t0
    results["exercise_prob_fig"] = fig
    plt.close("all")
    with ph["exercise_prob"].container():
        st.pyplot(fig)
        st.caption(f"⏱ {_fmt_time(timings['exercise_prob'])}")


def _fill_amer_vs_euro(ph: Dict[str, Any], results: Dict[str, Any],
                       timings: Dict[str, float],
                       ctx: Dict[str, Any]) -> None:
    """Prime d'exercice anticipé vs taux."""
    t0: float = time.perf_counter()
    try:
        fig, _, _ = american_european_gap_vs_rate(
            ctx["pricer"], ctx["market"], ctx["option"],
            rate_min=ctx["rate_min"], rate_max=ctx["rate_max"],
            nb_points=ctx["rate_points"], nb_steps_for_study=ctx["N"])
        results["amer_euro_fig"] = fig
        plt.close("all")
    except Exception as e:
        st.error(f"American vs European study failed: {e}")
    timings["amer_euro"] = time.perf_counter() - t0
    if "amer_euro_fig" in results:
        with ph["amer_euro"].container():
            st.pyplot(results["amer_euro_fig"])
            st.caption(f"⏱ {_fmt_time(timings['amer_euro'])}")


def _fill_avg_value(ph: Dict[str, Any], results: Dict[str, Any],
                    timings: Dict[str, float], tree_obj: Tree) -> None:
    """Valeur moyenne de l'option par date."""
    t0: float = time.perf_counter()
    fig, _, _ = tree_obj.plot_average_option_value_by_date()
    timings["avg_value"] = time.perf_counter() - t0
    results["avg_value_fig"] = fig
    plt.close("all")
    with ph["avg_value"].container():
        st.pyplot(fig)
        st.caption(f"⏱ {_fmt_time(timings['avg_value'])}")


def _fill_price_vs_exdiv(ph: Dict[str, Any], results: Dict[str, Any],
                         timings: Dict[str, float],
                         ctx: Dict[str, Any]) -> None:
    """Prix de l'option vs date ex-div."""
    t0: float = time.perf_counter()
    try:
        fig, _, _, _, _ = price_vs_ex_div_date(
            ctx["pricer"], ctx["market"], ctx["option"],
            nb_points=100, nb_steps_for_study=ctx["N"])
        results["price_vs_exdiv_fig"] = fig
        plt.close("all")
    except Exception as e:
        st.error(f"Price vs. ex-div study failed: {e}")
    timings["price_vs_exdiv"] = time.perf_counter() - t0
    if "price_vs_exdiv_fig" in results:
        with ph["price_vs_exdiv"].container():
            st.pyplot(results["price_vs_exdiv_fig"])
            st.caption(f"⏱ {_fmt_time(timings['price_vs_exdiv'])}")


def _fill_debug(ph: Dict[str, Any], timings: Dict[str, float],
                tree_obj: Tree) -> None:
    """Panneau debug."""
    with ph["debug"].container():
        _render_debug_panel(tree_obj)
        st.caption(f"⏱ {_fmt_time(tree_obj.price_time)}")


def _fill_tree_display(flags: Dict[str, bool], ph: Dict[str, Any],
                       timings: Dict[str, float],
                       tree_obj: Optional[Tree]) -> None:
    """Affiche l'arbre et gère l'export Excel."""
    if not flags["show_tree"]:
        st.session_state.tree = None
        return
    if tree_obj is None:
        return
    t0: float = time.perf_counter()
    excel_bytes = tree_obj.export_to_excel()
    t_display: float = time.perf_counter() - t0
    st.session_state.tree = tree_obj
    st.session_state.excel_bytes = excel_bytes
    timings["tree_display"] = t_display
    max_display = st.session_state.get("max_display_steps", 10)
    with ph["tree"].container():
        _render_tree_content(tree_obj, excel_bytes, int(max_display),
                             elapsed=t_display, static_mode=False)


# ---------------------------------------------------------------------------
# Rendu arbre et bouton Excel
# ---------------------------------------------------------------------------
def _render_tree_content(tree_obj: Tree, excel_bytes: bytes,
                         max_display: int, elapsed: Optional[float] = None,
                         static_mode: bool = False) -> None:
    """Affiche l'arbre Plotly + bouton Excel."""
    fig = tree_obj.plot_tree(max_steps=int(max_display))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Export to Excel**")
    _render_excel_download(excel_bytes, static_mode)
    if elapsed is not None:
        st.caption(f"⏱ {_fmt_time(elapsed)}")


def _render_excel_download(excel_bytes: Optional[bytes],
                           static_mode: bool) -> None:
    """Affiche le bouton de téléchargement Excel."""
    if excel_bytes is None:
        st.caption("*Excel file not prepared.*")
        return
    if static_mode:
        excel_name = st.text_input("File name", value="tree.xlsx",
                                   key="excel_name")
    else:
        excel_name = "tree.xlsx"
        st.caption("File will download as `tree.xlsx`.")
    st.download_button(
        label="Download Excel file", data=excel_bytes,
        file_name=excel_name,
        mime="application/vnd.openxmlformats-officedocument"
             ".spreadsheetml.sheet",
        on_click="ignore", key="download_excel",
    )


# ---------------------------------------------------------------------------
# Panneau debug
# ---------------------------------------------------------------------------
def _render_debug_panel(tree_obj: Optional[Tree]) -> None:
    """Affiche le panneau de debug de l'arbre."""
    if tree_obj is None:
        st.info("Debug: aucun arbre construit lors de ce run.")
        return
    _render_debug_mode_header(tree_obj)
    _render_debug_metrics(tree_obj)
    _render_debug_growth(tree_obj)
    _render_debug_tech_params(tree_obj)
    st.divider()
    _render_debug_cross_check(tree_obj)
    st.divider()
    _render_debug_sanity(tree_obj)
    st.divider()
    _render_debug_steps_table(tree_obj)


def _render_debug_mode_header(tree_obj: Tree) -> None:
    """Affiche le mode de construction (binomial, trinomial, mixte)."""
    if tree_obj.tree_mode == "trinomial_everywhere":
        st.markdown("##### Mode : TRINOMIAL only")
    elif any(m == "TRI" for m in tree_obj.step_modes):
        n_tri = sum(1 for m in tree_obj.step_modes if m == "TRI")
        st.markdown(f"##### Mode : MIXED — binomial except {n_tri} "
                    "steps ex-div in trinomial")
    else:
        st.markdown("##### Mode : BINOMIAL (without dividend)")


def _render_debug_metrics(tree_obj: Tree) -> None:
    """Compteurs, temps et statut pruning."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Steps", tree_obj.nb_steps)
    col2.metric("Nodes created", f"{tree_obj.counters['nodes_created']:,}")
    col3.metric("Merges", f"{tree_obj.counters['merges']:,}")
    col4.metric("Clamped probas",
                f"{tree_obj.counters.get('clamped_probas', 0):,}")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Build time", _fmt_time(tree_obj.build_time))
    col6.metric("Backward time", _fmt_time(tree_obj.backward_time))
    total_t = tree_obj.build_time + tree_obj.backward_time
    col7.metric("Total", _fmt_time(total_t))
    col8.metric("Pruning",
                "ON" if tree_obj.pricer_parameters.pruning else "OFF")


def _render_debug_growth(tree_obj: Tree) -> None:
    """Ratios de croissance BIN max et global."""
    ratios = [tree_obj.step_nodes[i + 1] / max(tree_obj.step_nodes[i], 1)
              for i in range(len(tree_obj.step_nodes) - 1)]
    modes_after = tree_obj.step_modes[1:]
    bin_ratios = [ratios[i] for i, m in enumerate(modes_after)
                  if m == "BIN"]
    bin_growth = max(bin_ratios) if bin_ratios else 0.0
    global_growth = max(ratios) if ratios else 0.0
    _display_growth(bin_growth, global_growth)


def _display_growth(bin_growth: float, global_growth: float) -> None:
    """Affiche les deux ratios de croissance."""
    col9, col10 = st.columns(2)
    with col9:
        if bin_growth >= 2.5:
            st.error(f"**Growth ratio (BIN max) = {bin_growth:.4f}** "
                     "— explosion détectée")
        else:
            st.success(f"**Growth ratio (BIN max) = {bin_growth:.4f}** "
                       "— recombinaison OK")
    with col10:
        st.info(f"**Growth ratio (global) = {global_growth:.4f}**")


def _render_debug_tech_params(tree_obj: Tree) -> None:
    """Affiche les paramètres techniques (Δt, DF, α)."""
    with st.expander("Parameters (Δt, DF, α)", expanded=False):
        tech1, tech2, tech3, tech4 = st.columns(4)
        tech1.metric("Δt (years)", f"{tree_obj.delta_t:.6f}")
        tech2.metric("DF = e^(−r·Δt)", f"{tree_obj.df:.6f}")
        tech3.metric("√α_bin", f"{tree_obj.sqrt_alpha_bin:.6f}")
        tech4.metric("α_tri (HW)", f"{tree_obj.alpha_tri:.6f}")
        _render_debug_tri_summary(tree_obj)


def _render_tri_steps_info(tree_obj: Tree) -> None:
    """Affiche le nombre de pas trinomiaux."""
    n_tri = sum(1 for m in tree_obj.step_modes if m == "TRI")
    n_steps_total = len(tree_obj.step_modes) - 1
    if tree_obj.market_data.dividend > 0:
        if n_tri >= 1:
            st.success(f"TRI steps : {n_tri} / {n_steps_total} "
                       "— The ex-div step was correctly detected.")
        else:
            st.error(f"TRI steps : 0 / {n_steps_total} "
                     "— Dividend > 0 but no trinomial steps triggered.")
    else:
        st.caption(f"- TRI steps : {n_tri} / {n_steps_total} "
                   "— No dividend, so expected at 0.")


def _render_clamped_probas_info(tree_obj: Tree) -> None:
    """Affiche les probas clampées si nécessaire."""
    clamped = tree_obj.counters.get("clamped_probas", 0)
    if clamped > 0:
        pct = 100.0 * clamped / max(tree_obj.counters["nodes_created"], 1)
        st.info(f"- Renormalized probabilities (clamp) : {clamped} nœuds "
                f"({pct:.3f}% du total).")
    else:
        st.caption("- No negative probability detected.")


def _render_debug_tri_summary(tree_obj: Tree) -> None:
    """Résume les pas trinomiaux et les probabilités clampées."""
    _render_tri_steps_info(tree_obj)
    _render_clamped_probas_info(tree_obj)


def _compute_cross_check(tree_obj: Tree) -> Tuple[float, float, float]:
    """Calcule backward price, terminal price et leur écart."""
    if tree_obj.root.opt_price is None:
        tree_obj.backward_pricing()
    backward_price = tree_obj.root.opt_price
    terminal_price = tree_obj.european_price_by_terminal_sum()
    diff = abs(backward_price - terminal_price)
    return backward_price, terminal_price, diff


def _display_cross_check(backward_price: float, terminal_price: float,
                         diff: float) -> None:
    """Affiche les résultats du cross-check."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Backward pricing", f"{backward_price:.8f}")
    c2.metric("Terminal sum", f"{terminal_price:.8f}")
    c3.metric("Écart", f"{diff:.2e}")
    tol = max(1e-9, 1e-4 * abs(backward_price))
    if diff < tol:
        st.success(f"The two methods match "
                   f"(gap = {diff:.2e} < tolerance = {tol:.2e}).")
    else:
        st.error(f"Abnormal gap : {diff:.2e} > tolerance = {tol:.2e}.")


def _render_debug_cross_check(tree_obj: Tree) -> None:
    """Compare backward pricing vs. somme terminale (européen)."""
    if tree_obj.option_data.is_american():
        st.caption("Cross-check not available for American options.")
        return
    st.markdown("##### Cross-check (backward vs. terminal sum)")
    try:
        values = _compute_cross_check(tree_obj)
        _display_cross_check(*values)
    except Exception as e:
        st.warning(f"Cross-check unvailable : {e}")


def _display_sanity_check(c: Dict[str, str]) -> None:
    """Affiche un sanity check individuel."""
    icon = {"PASS": "", "FAIL": "", "N/A": "-"}.get(c["verdict"], "? ")
    msg = f"{icon} **{c['name']}** — `{c['value']}`"
    if c.get("detail"):
        msg += f"  _{c['detail']}_"
    if c["verdict"] == "PASS":
        st.success(msg)
    elif c["verdict"] == "FAIL":
        st.error(msg)
    else:
        st.caption(msg)


def _render_debug_sanity(tree_obj: Tree) -> None:
    """Affiche les sanity checks."""
    st.markdown("##### Sanity checks")
    try:
        for c in tree_obj.sanity_checks():
            _display_sanity_check(c)
    except Exception as e:
        st.error(f"Sanity checks unvailable : {e}")


def _render_debug_steps_table(tree_obj: Tree) -> None:
    """Affiche le DataFrame détaillé des steps."""
    df = tree_obj.debug_steps_dataframe()
    with st.expander(f"Détail by step ({len(df)} lines)", expanded=False):
        st.dataframe(df, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Réaffichage statique
# ---------------------------------------------------------------------------
def _redisplay_results(flags: Dict[str, bool],
                       ctx: Dict[str, Any]) -> None:
    """Réaffiche les résultats sauvegardés (sans recalcul)."""
    results = st.session_state.results
    timings = results.get("_timings", {})
    st.divider()
    st.subheader("Results")
    snap = results.get("_params_snapshot") or st.session_state.get(
        "_current_snapshot")
    if snap is not None:
        _render_run_recap_from_snapshot(snap)
    _redisplay_prices(results, timings)
    _redisplay_greeks(results, timings)
    _redisplay_standard_figs(results, timings)
    _redisplay_advanced_figs(results, timings)


def _redisplay_prices(results: Dict[str, Any],
                      timings: Dict[str, float]) -> None:
    """Réaffiche les prix."""
    if "tree_price" not in results and "bs_price" not in results:
        return
    st.markdown("**Prices**")
    pc1, pc2 = st.columns(2)
    for key, label, col in (("tree_price", "Trinomial tree price", pc1),
                            ("bs_price", "Black & Scholes price", pc2)):
        if key in results:
            with col.container():
                st.metric(label, f"{results[key]:.6f}")
                if key in timings:
                    st.caption(f"⏱ {_fmt_time(timings[key])}")


def _redisplay_greeks(results: Dict[str, Any],
                      timings: Dict[str, float]) -> None:
    """Réaffiche le tableau Greeks."""
    if "greeks_tree" not in results and "greeks_bs" not in results:
        return
    st.markdown("**Greeks**")
    st.dataframe(_greeks_dataframe(results), hide_index=True,
                 use_container_width=True)
    keys = [k for k in ("greeks_tree", "greeks_bs") if k in timings]
    if keys:
        st.caption("⏱ " + " | ".join(
            f"{k}: {_fmt_time(timings[k])}" for k in keys))


def _redisplay_standard_figs(results: Dict[str, Any],
                             timings: Dict[str, float]) -> None:
    """Réaffiche les figures des études standard."""
    sections = [
        ("strike_fig", "strike", "Prices and slopes vs. strike"),
        ("error_curve_fig", "error_curve", "Error vs. number of steps"),
        ("exec_fig", "exec", "Execution time"),
        ("pruning_compare_fig", "pruning_compare",
         "Pruning effect on runtime"),
        ("conv_fig", "conv", "Convergence towards Black & Scholes"),
        ("compare_fig", "compare", "Backward vs. recursive"),
    ]
    for fig_key, timing_key, title in sections:
        if fig_key in results:
            st.markdown(f"**{title}**")
            st.pyplot(results[fig_key])
            if timing_key in timings:
                st.caption(f"⏱ {_fmt_time(timings[timing_key])}")


def _redisplay_advanced_section(results: Dict[str, Any],
                                timings: Dict[str, float],
                                fig_key: str, timing_key: str,
                                title: str) -> None:
    """Réaffiche une figure d'étude avancée."""
    if fig_key not in results:
        return
    st.markdown(f"**{title}**")
    st.pyplot(results[fig_key])
    if timing_key in timings:
        st.caption(f"⏱ {_fmt_time(timings[timing_key])}")


def _redisplay_advanced_sections(results: Dict[str, Any],
                                 timings: Dict[str, float]) -> None:
    """Réaffiche les 4 figures des études avancées."""
    _redisplay_advanced_section(results, timings,
                                "exercise_prob_fig", "exercise_prob",
                                "Exercise probability by dates")
    _redisplay_advanced_section(results, timings,
                                "amer_euro_fig", "amer_euro",
                                "American vs. European vs. rate")
    _redisplay_advanced_section(results, timings,
                                "avg_value_fig", "avg_value",
                                "Average option value by dates")
    _redisplay_advanced_section(results, timings,
                                "price_vs_exdiv_fig", "price_vs_exdiv",
                                "Price vs. ex-div date")


def _redisplay_debug_and_tree(timings: Dict[str, float]) -> None:
    """Réaffiche le panneau debug et l'arbre (si actifs)."""
    if st.session_state.get("debug") and st.session_state.debug_tree:
        st.markdown("**Debug — tree construction**")
        _render_debug_panel(st.session_state.debug_tree)
    if st.session_state.tree is not None:
        st.markdown("**Trinomial tree**")
        max_display = st.session_state.get("max_display_steps", 10)
        _render_tree_content(st.session_state.tree,
                             st.session_state.excel_bytes,
                             int(max_display),
                             elapsed=timings.get("tree_display"),
                             static_mode=True)


def _redisplay_advanced_figs(results: Dict[str, Any],
                             timings: Dict[str, float]) -> None:
    """Réaffiche les figures avancées + debug + tree."""
    _redisplay_advanced_sections(results, timings)
    _redisplay_debug_and_tree(timings)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def _render_max_display_input(ctx: Dict[str, Any]) -> None:
    """Affiche l'input du nombre max de steps pour le plot."""
    st.number_input("Max steps to display", value=min(10, ctx["N"]),
                    min_value=1, max_value=min(50, ctx["N"]), step=1,
                    key="max_display_steps",
                    help="Capped at 50 for readability.")


def _render_main_widgets(ctx: Dict[str, Any]) -> Dict[str, bool]:
    """Rend les checkboxes et captions du panneau principal."""
    flags = _render_checkboxes(ctx)
    # --- Shift info AVANT les captions d'études (comme dans l'original) ---
    if flags["greeks_tree"]:
        ctx["shifts"] = _render_shift_info()
    _render_study_captions(flags, ctx)
    if flags["show_tree"]:
        _render_max_display_input(ctx)
    return flags


def _dispatch_run(flags: Dict[str, bool], ctx: Dict[str, Any]) -> None:
    """Dispatch entre run et redisplay."""
    run = st.button("Run calculations", type="primary")
    if run:
        _execute_all(flags, ctx)
    elif st.session_state.results is not None:
        _redisplay_results(flags, ctx)


def main(*args: Any) -> None:
    """Point d'entrée principal de l'application."""
    st.set_page_config(page_title="Option Pricing: Trinomial Tree",
                       layout="wide")
    st.title("Option Pricing with a Binomial/Trinomial Tree")
    _init_state()
    _render_sidebar()
    pricer, market, option = _build_objects()
    ctx = _prepare_context(pricer, market, option)
    flags = _render_main_widgets(ctx)
    _dispatch_run(flags, ctx)


if __name__ == "__main__":
    main()
