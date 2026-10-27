import matplotlib
matplotlib.use('Agg')

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import time
import numpy as np

from option import Option
from market import Market
from parameters import PricerParameters
from black_scholes import Black_Scholes
from greeks import GreeksTree, GreeksBS
from timing import ExecutionTime
from tree import Tree
from studies import american_european_gap_vs_rate, price_vs_ex_div_date, greeks_profiles


DEFAULTS = {
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


def fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for key in ("results", "tree", "excel_bytes", "debug_tree"):
        if key not in st.session_state:
            st.session_state[key] = None


def build_objects():
    market = Market(
        st.session_state.s0, st.session_state.rate, st.session_state.vol,
        st.session_state.dividend,
        dt.datetime.combine(st.session_state.ex_div_date, dt.datetime.min.time()),
    )
    option = Option(
        st.session_state.K,
        dt.datetime.combine(st.session_state.maturity, dt.datetime.min.time()),
        st.session_state.call_put, st.session_state.eur_am,
    )
    pricer = PricerParameters(
        dt.datetime.combine(st.session_state.pricing_date, dt.datetime.min.time()),
        int(st.session_state.nb_steps),
        bool(st.session_state.pruning),
        p_min=1e-9,
    )
    pricer.tree_mode = st.session_state.get("tree_mode", "binomial_with_div_switch")
    if st.session_state.maturity <= st.session_state.pricing_date:
        st.sidebar.error("⚠️ Maturity must be after pricing date.")
        st.stop()
    return pricer, market, option


def nb_steps_from_error(max_error: float, s0: float, vol: float, T: float) -> int:
    up = vol ** 2 * T
    down = np.log(1 + (8 * np.sqrt(2 * np.pi) * max_error * np.sqrt(np.exp(vol ** 2 * T) - 1)) / (3 * s0))
    return max(1, int(round(up / down)))


def greeks_dataframe(results: dict) -> pd.DataFrame:
    greek_names = ["Delta", "Gamma", "Theta", "Rho", "Vega", "Vomma", "Vanna"]
    data = {"Greek": greek_names}
    
    has_tree = "greeks_tree" in results
    has_bs = "greeks_bs" in results
    
    if has_tree:
        data["Empirical (tree)"] = [f"{results['greeks_tree'][g]:.6f}" for g in greek_names]
    if has_bs:
        data["Theoretical (B&S)"] = [f"{results['greeks_bs'][g]:.6f}" for g in greek_names]
    if has_tree and has_bs:
        data["Difference (Tree - B&S)"] = [
            f"{results['greeks_tree'][g] - results['greeks_bs'][g]:.6f}" 
            for g in greek_names
        ]
    return pd.DataFrame(data)


def snapshot_params(pricer, market, option, method: str) -> dict:
    return {
        "s0": market.s0, "K": option.K, "rate": market.rate,
        "vol": market.vol, "dividend": market.dividend,
        "ex_div_date": market.ex_div_date, "maturity": option.maturity,
        "pricing_date": pricer.pricing_date, "call_put": option.call_put,
        "eur_am": option.eur_am, "nb_steps": int(pricer.nb_steps),
        "pruning": bool(pricer.pruning), "tree_mode": pricer.tree_mode,
        "method": method,
    }


def render_run_recap_from_snapshot(snap: dict) -> None:
    N = snap["nb_steps"]
    if snap["tree_mode"] == "trinomial_everywhere":
        mode = "Trinomial partout"
    elif snap["dividend"] == 0:
        mode = "Binomial"
    else:
        mode = "Binomial + trinomial (pas ex-div)"

    T = (snap["maturity"] - snap["pricing_date"]).days / 365
    parts = [
        f"**S₀** = {snap['s0']:.4f}", f"**K** = {snap['K']:.4f}",
        f"**r** = {snap['rate']:.4%}", f"**σ** = {snap['vol']:.4%}",
        f"**N** = {N}", f"**{snap['call_put'].capitalize()} · {snap['eur_am'].capitalize()}**",
        f"**T** = {T:.4f} y",
    ]
    if snap["dividend"] > 0:
        parts.append(f"**D** = {snap['dividend']:.4f}")
        parts.append(f"**ex-div** = {snap['ex_div_date']:%Y-%m-%d}")
    else:
        parts.append("**D** = 0 (pas de dividende)")
    parts.append(f"**Mode** = {mode}")
    parts.append(f"**Pricing** = {snap['method'].capitalize()}")
    parts.append(f"**Pruning** = {'ON' if snap['pruning'] else 'OFF'}")
    st.info("  ·  ".join(parts))


def render_tree_content(tree_obj: Tree, excel_bytes: bytes, max_display: int,
                        elapsed: float = None, static_mode: bool = False) -> None:
    fig = tree_obj.plot_tree(max_steps=int(max_display))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Export to Excel**")
    if excel_bytes is None:
        st.caption("*Excel file not prepared.*")
    else:
        if static_mode:
            excel_name = st.text_input("File name", value="tree.xlsx", key="excel_name")
        else:
            excel_name = "tree.xlsx"
            st.caption("File will download as `tree.xlsx`.")
        st.download_button(
            label="Download Excel file", data=excel_bytes, file_name=excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click="ignore", key="download_excel",
        )
    if elapsed is not None:
        st.caption(f"⏱ {fmt_time(elapsed)}")


def render_debug_panel(tree_obj: Tree) -> None:
    if tree_obj is None:
        st.info("Debug: aucun arbre construit lors de ce run.")
        return

    if tree_obj.tree_mode == "trinomial_everywhere":
        st.markdown("##### Mode : TRINOMIAL only")
    elif any(m == "TRI" for m in tree_obj.step_modes):
        n_tri = sum(1 for m in tree_obj.step_modes if m == "TRI")
        st.markdown(f"##### Mode : MIXED — binomial except {n_tri} steps ex-div in trinomial")
    else:
        st.markdown("##### Mode : BINOMIAL (without dividend)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Steps", tree_obj.nb_steps)
    col2.metric("Nodes created", f"{tree_obj.counters['nodes_created']:,}")
    col3.metric("Merges", f"{tree_obj.counters['merges']:,}")
    col4.metric("Clamped probas", f"{tree_obj.counters.get('clamped_probas', 0):,}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Build time", fmt_time(tree_obj.build_time))
    col6.metric("Backward time", fmt_time(tree_obj.backward_time))
    total_t = tree_obj.build_time + tree_obj.backward_time
    col7.metric("Total", fmt_time(total_t))
    col8.metric("Pruning", "ON" if tree_obj.pricer_parameters.pruning else "OFF")

    ratios = [tree_obj.step_nodes[i+1] / max(tree_obj.step_nodes[i], 1)
              for i in range(len(tree_obj.step_nodes) - 1)]
    modes_after = tree_obj.step_modes[1:]
    bin_ratios = [ratios[i] for i, m in enumerate(modes_after) if m == "BIN"]
    bin_growth = max(bin_ratios) if bin_ratios else 0.0
    global_growth = max(ratios) if ratios else 0.0

    col9, col10 = st.columns(2)
    with col9:
        if bin_growth >= 2.5:
            st.error(f"**Growth ratio (BIN max) = {bin_growth:.4f}** — explosion détectée")
        else:
            st.success(f"**Growth ratio (BIN max) = {bin_growth:.4f}** — recombinaison OK")
    with col10:
        st.info(f"**Growth ratio (global) = {global_growth:.4f}**")

    with st.expander("Parameters (Δt, DF, α)", expanded=False):
        tech1, tech2, tech3, tech4 = st.columns(4)
        tech1.metric("Δt (years)", f"{tree_obj.delta_t:.6f}")
        tech2.metric("DF = e^(−r·Δt)", f"{tree_obj.df:.6f}")
        tech3.metric("√α_bin", f"{tree_obj.sqrt_alpha_bin:.6f}")
        tech4.metric("α_tri (HW)", f"{tree_obj.alpha_tri:.6f}")
        n_tri = sum(1 for m in tree_obj.step_modes if m == "TRI")
        n_steps_total = len(tree_obj.step_modes) - 1
        if tree_obj.market_data.dividend > 0:
            if n_tri >= 1:
                st.success(f"TRI steps : {n_tri} / {n_steps_total} — The ex-div step was correctly detected.")
            else:
                st.error(f"TRI steps : 0 / {n_steps_total} — Dividend > 0 but no trinomial steps has been triggered.")
        else:
            st.caption(f"- TRI steps : {n_tri} / {n_steps_total} — No dividend, so expected at 0.")
        clamped = tree_obj.counters.get("clamped_probas", 0)
        if clamped > 0:
            pct = 100.0 * clamped / max(tree_obj.counters["nodes_created"], 1)
            st.info(f"- Renormalized probabilities (clamp) : {clamped} nœuds ({pct:.3f}% du total).")
        else:
            st.caption("- No negative probability detected.")

    st.divider()

    if not tree_obj.option_data.is_american():
        st.markdown("##### Cross-check (backward vs. terminal sum)")
        try:
            if tree_obj.root.opt_price is None:
                tree_obj.backward_pricing()
            backward_price = tree_obj.root.opt_price
            terminal_price = tree_obj.european_price_by_terminal_sum()
            diff = abs(backward_price - terminal_price)
            c1, c2, c3 = st.columns(3)
            c1.metric("Backward pricing", f"{backward_price:.8f}")
            c2.metric("Terminal sum",     f"{terminal_price:.8f}")
            c3.metric("Écart",            f"{diff:.2e}")
            tol = max(1e-9, 1e-4 * abs(backward_price))
            if diff < tol:
                st.success(f"The two methods match (gap = {diff:.2e} < tolerance = {tol:.2e}).")
            else:
                st.error(f"Abnormal gap : {diff:.2e} > tolerance = {tol:.2e}.")
        except Exception as e:
            st.warning(f"Cross-check unvailable : {e}")
    else:
        st.caption("Cross-check not available for American options.")

    st.divider()

    st.markdown("##### Sanity checks")
    try:
        checks = tree_obj.sanity_checks()
        for c in checks:
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
    except Exception as e:
        st.error(f"Sanity checks unvailable : {e}")

    st.divider()
    df = tree_obj.debug_steps_dataframe()
    with st.expander(f"Détail by step ({len(df)} lines)", expanded=False):
        st.dataframe(df, hide_index=True, use_container_width=True)


def render_sidebar() -> None:
    st.sidebar.header("Market data")
    st.sidebar.number_input("Underlying (S0)", key="s0", min_value=1e-6, format="%.4f",
                            help="Must be strictly positive.")
    st.sidebar.number_input("Risk-free rate (r)", key="rate", min_value=-1.0, max_value=1.0, format="%.4f",
                            help="Bounded at [-1, +1].")
    st.sidebar.number_input("Volatility (σ)", key="vol", min_value=1e-6, format="%.6f",
                            help="Must be strictly positivee.")
    st.sidebar.number_input("Discrete dividend (€)", key="dividend", min_value=0.0, format="%.4f")
    st.sidebar.date_input("Ex-dividend date", key="ex_div_date",
                          help="Ignored if dividend = 0.")

    st.sidebar.header("Option data")
    st.sidebar.number_input("Strike price (K)", key="K", min_value=0.01, format="%.4f",
                            help="Must be strictly positive.")
    st.sidebar.date_input("Maturity date", key="maturity",
                          help="Must be strictly after the pricing date.")
    st.sidebar.radio("Option type", ["call", "put"], key="call_put", horizontal=True, format_func=lambda x: x.capitalize())
    st.sidebar.radio("Exercise style", ["european", "american"], key="eur_am", horizontal=True, format_func=lambda x: x.capitalize())

    st.sidebar.header("Pricer parameters")
    st.sidebar.date_input("Pricing date", key="pricing_date")

    step_choice = st.sidebar.radio("Choice of number of steps", ["Number of steps", "Maximum error"], key="step_choice")
    if step_choice == "Number of steps":
        st.sidebar.number_input("Number of steps (N)", key="nb_steps", min_value=1, max_value=10000, step=1,
                                help="≤ 900 if method = recursive.")
    else:
        st.sidebar.number_input("Maximum error (€)", key="max_error", min_value=1e-9, value=0.01, format="%.4f")
        T = (st.session_state.maturity - st.session_state.pricing_date).days / 365
        if (T > 0 and st.session_state.vol > 0 and st.session_state.s0 > 0 and st.session_state.max_error > 0):
            computed_n = nb_steps_from_error(st.session_state.max_error, st.session_state.s0, st.session_state.vol, T)
            st.session_state.nb_steps = computed_n
            st.sidebar.info(f"→ N = **{computed_n}**")
        else:
            st.sidebar.error("Inputs invalides pour calculer N.")

    st.sidebar.checkbox("Enable pruning", key="pruning",
                        help="p_min = 1e-9.")

    st.sidebar.radio("Pricing method", ["backward", "recursive"], key="method", horizontal=True, format_func=lambda x: x.capitalize(),
                    help="Recursive limited at ~900 steps (RecursionError).")

    st.sidebar.radio("Tree mode", ["binomial_with_div_switch", "trinomial_everywhere"], key="tree_mode",
                     format_func=lambda x: ("Binomial (switch to trinomial if div)" if x == "binomial_with_div_switch" else "Trinomial only"))


def render_shift_info():
    s0 = float(st.session_state.s0)
    vol = float(st.session_state.vol)
    N = int(st.session_state.nb_steps)
    
    # À petit N, il faut de plus grands shifts pour lisser le bruit
    # À grand N, on peut se permettre des shifts plus petits
    if N < 30:
        # Pour N très petit, on augmente les shifts
        h_S = 0.10 * s0     # 10% de S0
        h_vol = 0.25 * vol  # 25% relatif de la vol (ex: 0.05 pour vol=20%)
    elif N < 100:
        h_S = 0.05 * s0     # 5% de S0
        h_vol = 0.25 * vol  
    else:
        h_S = 0.02 * s0     # 2% de S0
        h_vol = 0.20 * vol  # 20% relatif
    
    shift_delta = h_S
    shift_gamma = h_S
    shift_vega = h_vol
    shift_vomma = h_vol
    shift_vanna_s = h_S
    shift_vanna_vol = h_vol
    shift_rho = 0.01
    shift_theta_days = 365.0 / 252.0
    
    st.caption(
        f"ℹ️ **Adaptive finite-difference shifts** (N={N}): "
        f"Δ = {shift_delta:.4f} · Γ = {shift_gamma:.4f} · "
        f"ν = {shift_vega:.4f} · Vomma = {shift_vomma:.4f} · "
        f"Vanna = ({shift_vanna_s:.4f}, {shift_vanna_vol:.4f})"
    )
    return (shift_delta, shift_gamma, shift_vega, shift_vomma, 
            shift_vanna_s, shift_vanna_vol, shift_rho, shift_theta_days)


def main() -> None:
    st.set_page_config(page_title="Option Pricing: Trinomial Tree", layout="wide")
    st.title("Option Pricing with a Binomial/Trinomial Tree")
    init_state()
    render_sidebar()

    pricer, market, option = build_objects()
    bs_available = (not option.is_american()) and market.dividend == 0
    is_american = option.is_american()
    N = st.session_state.nb_steps

    K = st.session_state.K
    s0 = st.session_state.s0
    r = st.session_state.rate
    sigma = st.session_state.vol
    T_years = (st.session_state.maturity - st.session_state.pricing_date).days / 365

    strike_min = max(1, int(K * 0.5))
    strike_max = max(strike_min + 1, int(K * 1.5))
    rate_min = -0.05
    rate_max = 0.05
    rate_points = 15

    if T_years > 0 and sigma > 0:
        C = (3 * s0 * np.exp(r * T_years)) / (8 * np.sqrt(2 * np.pi)) \
            * (sigma ** 2) / np.sqrt(np.exp(sigma ** 2 * T_years) - 1) * T_years
        target_error = C / max(N, 1)
    else:
        target_error = 0.01

    max_display = min(N, 10)
    disable_sweeps = (N < 10 or N > 1000)

    st.subheader("Compute options")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Pricing**")
        compute_tree_price = st.checkbox("Compute price with trinomial tree", value=True)
        compute_bs_price = st.checkbox("Compute price with Black & Scholes", disabled=not bs_available)
        compute_greeks_tree = st.checkbox("Compute empirical Greeks (with the tree)")
        compute_greeks_bs = st.checkbox("Compute theoretical Greeks (B&S)", disabled=not bs_available)
    with c2:
        st.markdown("**Standard studies**")
        compute_strike_study = st.checkbox("Study strike range (Tree vs B&S)", disabled=not bs_available)
        compute_error_curve = st.checkbox("Show error vs. steps curve", disabled=not bs_available or N < 30,
                                          help="Disabled if N < 30 (so the convergence tunnel is visible).")
        compute_exec_time = st.checkbox("Study execution time", disabled=disable_sweeps,
                                        help="Disabled if N < 10 or N > 1000.")
        compute_pruning_compare = st.checkbox("Compare pruning ON vs OFF", disabled=disable_sweeps,
                                              help="Disabled if N < 10 or N > 1000.")
        compute_conv = st.checkbox("Study convergence with B&S", disabled=not bs_available or disable_sweeps,
                                   help="Disabled if N < 10 or N > 1000.")
        compute_compare_methods = st.checkbox("Backward vs. recursive comparison", disabled=disable_sweeps,
                                              help="Disabled if N < 10 or N > 1000.")
    with c3:
        st.markdown("**Advanced studies**")
        compute_exercise_prob = st.checkbox("Exercise probability by dates", disabled=not is_american)
        compute_amer_vs_euro = st.checkbox("American vs. European vs. rate")
        compute_avg_value = st.checkbox("Average option value by dates")
        compute_price_vs_exdiv = st.checkbox("Price vs. ex-div date")
        compute_greeks_profile = st.checkbox("Greeks Profiles (Tree vs BS)", disabled=not bs_available)
        compute_debug = st.checkbox("Debug — tree construction", key="debug")
        show_tree = st.checkbox("Show tree")

    if not bs_available:
        compute_bs_price = False
        compute_greeks_bs = False
        compute_strike_study = False
        compute_conv = False
        compute_error_curve = False
    if not is_american:
        compute_exercise_prob = False

    if not bs_available:
        st.caption("⚠️ Black & Scholes is unavailable for American options or when a discrete dividend is present.")
    if disable_sweeps and (compute_exec_time or compute_pruning_compare or compute_conv or compute_compare_methods):
        st.caption(f"⚠️ Sweep studies are disabled because N={N} is too small (<10) or too large (>1000).")

    shift_delta = shift_gamma = shift_vega = shift_rho = shift_theta_days = None
    if compute_greeks_tree:
        (shift_delta, shift_gamma, shift_vega, shift_vomma, 
         shift_vanna_s, shift_vanna_vol, shift_rho, shift_theta_days) = render_shift_info()

    # --- Captions pour les paramètres dérivés ---
    if compute_strike_study:
        st.caption(f"ℹ️ Strike range : **{strike_min} → {strike_max}** (K ± 50%, déduit de K={K:.2f})")
    if compute_error_curve:
        st.caption(f"ℹ️ Target error : **{target_error:.4e} €** (déduit de N={N} via Gap(N) = C/N)")
    if compute_amer_vs_euro:
        st.caption(f"ℹ️ Rate range : **{rate_min:.2%} → {rate_max:.2%}** (±5%, {rate_points} points)")

    # --- Input max_display (borné par N) ---
    if show_tree:
        max_display = st.number_input("Max steps to display", value=min(10, N), min_value=1, max_value=min(50, N), step=1, key="max_display_steps",
                                      help="Capped at 50 for readability.")

    run = st.button("Run calculations", type="primary")

    # ================= RUN =================
    if run:
        st.session_state.excel_bytes = None
        st.session_state.debug_tree = None

        st.divider()
        st.subheader("Results")
        snap = snapshot_params(pricer, market, option, st.session_state.method)
        st.session_state["_current_snapshot"] = snap
        render_run_recap_from_snapshot(snap)

        price_tree_ph = bs_price_ph = None
        if compute_tree_price or compute_bs_price:
            st.markdown("**Prices**")
            pc1, pc2 = st.columns(2)
            if compute_tree_price:
                price_tree_ph = pc1.empty(); price_tree_ph.markdown(WAITING_MD)
            if compute_bs_price:
                bs_price_ph = pc2.empty(); bs_price_ph.markdown(WAITING_MD)

        greeks_ph = None
        if compute_greeks_tree or compute_greeks_bs:
            st.markdown("**Greeks**")
            greeks_ph = st.empty(); greeks_ph.markdown(WAITING_MD)

        strike_ph = None
        if compute_strike_study:
            st.markdown("**Prices and slopes vs. strike**")
            strike_ph = st.empty(); strike_ph.markdown(WAITING_MD)

        error_curve_ph = None
        if compute_error_curve:
            st.markdown("**Error vs. number of steps**")
            error_curve_ph = st.empty(); error_curve_ph.markdown(WAITING_MD)

        exec_ph = None
        if compute_exec_time:
            st.markdown("**Execution time**")
            exec_ph = st.empty(); exec_ph.markdown(WAITING_MD)

        pruning_compare_ph = None
        if compute_pruning_compare:
            st.markdown("**Pruning effect on runtime**")
            pruning_compare_ph = st.empty(); pruning_compare_ph.markdown(WAITING_MD)

        conv_ph = None
        if compute_conv:
            st.markdown("**Convergence towards Black & Scholes**")
            conv_ph = st.empty(); conv_ph.markdown(WAITING_MD)

        compare_ph = None
        if compute_compare_methods:
            st.markdown("**Backward vs. recursive**")
            compare_ph = st.empty(); compare_ph.markdown(WAITING_MD)

        exercise_prob_ph = None
        if compute_exercise_prob:
            st.markdown("**Exercise probability by dates**")
            exercise_prob_ph = st.empty(); exercise_prob_ph.markdown(WAITING_MD)

        amer_euro_ph = None
        if compute_amer_vs_euro:
            st.markdown("**American vs. European vs. rate**")
            amer_euro_ph = st.empty(); amer_euro_ph.markdown(WAITING_MD)

        avg_value_ph = None
        if compute_avg_value:
            st.markdown("**Average option value by dates**")
            avg_value_ph = st.empty(); avg_value_ph.markdown(WAITING_MD)

        price_vs_exdiv_ph = None
        if compute_price_vs_exdiv:
            st.markdown("**Price vs. ex-div date**")
            price_vs_exdiv_ph = st.empty(); price_vs_exdiv_ph.markdown(WAITING_MD)

        greeks_profile_ph = None
        if compute_greeks_profile:
            st.markdown("**Greeks Profiles**")
            greeks_profile_ph = st.empty(); greeks_profile_ph.markdown(WAITING_MD)

        debug_ph = None
        if compute_debug:
            st.markdown("**Debug — tree construction**")
            debug_ph = st.empty(); debug_ph.markdown(WAITING_MD)

        tree_ph = None
        if show_tree:
            st.markdown("**Trinomial tree**")
            tree_ph = st.empty(); tree_ph.markdown(WAITING_MD)

        results = {}
        timings = {}
        tree_obj = None
        tree_price = None

        need_tree = (compute_tree_price or show_tree or compute_exercise_prob or compute_avg_value or compute_debug)
        if need_tree:
            t0 = time.perf_counter()
            tree_obj = Tree(pricer, market, option, debug=bool(compute_debug))
            if st.session_state.method == "backward":
                tree_price = tree_obj.backward_pricing()
            else:
                tree_price = tree_obj.recursive_pricing()
            t_tree = time.perf_counter() - t0

            if compute_tree_price:
                if tree_price is None:
                    st.error("Le pricing récursif a échoué (RecursionError). Utilise Backward.")
                else:
                    results["tree_price"] = tree_price
                    timings["tree_price"] = t_tree
                    with price_tree_ph.container():
                        st.metric("Trinomial tree price", f"{tree_price:.6f}")
                        st.caption(f"⏱ {fmt_time(t_tree)}")

            if compute_debug:
                with debug_ph.container():
                    render_debug_panel(tree_obj)
                    st.caption(f"⏱ {fmt_time(t_tree)}")

            if show_tree:
                t0 = time.perf_counter()
                excel_bytes = tree_obj.export_to_excel()
                t_tree_display = time.perf_counter() - t0
                st.session_state.tree = tree_obj
                st.session_state.excel_bytes = excel_bytes
                timings["tree_display"] = t_tree_display
                tree_ph.empty()
                with tree_ph.container():
                    render_tree_content(tree_obj, excel_bytes, int(max_display), elapsed=t_tree_display, static_mode=False)

        if not show_tree:
            st.session_state.tree = None

        if compute_bs_price:
            t0 = time.perf_counter()
            results["bs_price"] = Black_Scholes(pricer, market, option).pricing_BS()
            timings["bs_price"] = time.perf_counter() - t0
            with bs_price_ph.container():
                st.metric("Black & Scholes price", f"{results['bs_price']:.6f}")
                st.caption(f"⏱ {fmt_time(timings['bs_price'])}")

        if compute_greeks_tree or compute_greeks_bs:
            t_g_tree = t_g_bs = None
            if compute_greeks_tree:
                t0 = time.perf_counter()
                gt = GreeksTree(pricer, market, option)
                results["greeks_tree"] = {
                    "Delta": gt.delta(float(shift_delta)), "Gamma": gt.gamma(float(shift_gamma)),
                    "Theta": gt.theta(int(shift_theta_days)), "Rho": gt.rho(float(shift_rho)),
                    "Vega": gt.vega(float(shift_vega)), "Vomma": gt.vomma(float(shift_vega)),
                    "Vomma": gt.vomma(float(shift_vomma)), 
                    "Vanna": gt.vanna(float(shift_vanna_s), float(shift_vanna_vol)),
                }
                t_g_tree = time.perf_counter() - t0
            if compute_greeks_bs:
                t0 = time.perf_counter()
                gbs = GreeksBS(pricer, market, option)
                results["greeks_bs"] = {
                    "Delta": gbs.delta(), "Gamma": gbs.gamma(), "Theta": gbs.theta(),
                    "Rho": gbs.rho(), "Vega": gbs.vega(), "Vomma": gbs.vomma(), "Vanna": gbs.vanna(),
                }
                t_g_bs = time.perf_counter() - t0
            parts = []
            if t_g_tree is not None: parts.append(f"tree: {fmt_time(t_g_tree)}")
            if t_g_bs is not None: parts.append(f"B&S: {fmt_time(t_g_bs)}")
            with greeks_ph.container():
                st.dataframe(greeks_dataframe(results), hide_index=True, use_container_width=True)
                st.caption(f"⏱ {' | '.join(parts)}")
            if t_g_tree is not None: timings["greeks_tree"] = t_g_tree
            if t_g_bs is not None: timings["greeks_bs"] = t_g_bs

        if compute_strike_study:
            t0 = time.perf_counter()
            _, _, _, _, fig = Black_Scholes(pricer, market, option).calculate_prices_and_slopes(int(strike_min), int(strike_max), nb_steps=N)
            timings["strike"] = time.perf_counter() - t0
            results["strike_fig"] = fig; plt.close("all")
            with strike_ph.container():
                st.pyplot(results["strike_fig"])
                st.caption(f"⏱ {fmt_time(timings['strike'])}")

        if compute_error_curve:
            t0 = time.perf_counter()
            fig = Black_Scholes(pricer, market, option).max_error_curve(nb_steps_max=max(10000, N), target_error=float(target_error))
            timings["error_curve"] = time.perf_counter() - t0
            results["error_curve_fig"] = fig; plt.close("all")
            with error_curve_ph.container():
                st.pyplot(fig)
                st.caption(f"⏱ {fmt_time(timings['error_curve'])} — theoretical curve")

        if compute_exec_time:
            t0 = time.perf_counter()
            setattr(pricer, "nb_steps", N)
            method_str = "recursive_pricing" if st.session_state.method == "recursive" else "backward_pricing"
            ExecutionTime(pricer, market, option).compute_execution_time(method_str)
            timings["exec"] = time.perf_counter() - t0
            results["exec_fig"] = plt.gcf(); plt.close("all")
            with exec_ph.container():
                st.pyplot(results["exec_fig"])
                st.caption(f"⏱ {fmt_time(timings['exec'])}")

        if compute_pruning_compare:
            t0 = time.perf_counter()
            try:
                fig, _, _, _ = ExecutionTime(pricer, market, option).compare_pruning(max_steps=N, n_repeats=2, log_spaced=True)
                results["pruning_compare_fig"] = fig; plt.close("all")
            except Exception as e:
                st.error(f"Pruning comparison failed: {e}")
            timings["pruning_compare"] = time.perf_counter() - t0
            if "pruning_compare_fig" in results:
                with pruning_compare_ph.container():
                    st.pyplot(results["pruning_compare_fig"])
                    st.caption(f"⏱ {fmt_time(timings['pruning_compare'])}")

        if compute_conv:
            t0 = time.perf_counter()
            method_str = "recursive" if st.session_state.method == "recursive" else "backward"
            Black_Scholes(pricer, market, option).convergence_BS(N, method=method_str)
            timings["conv"] = time.perf_counter() - t0
            results["conv_fig"] = plt.gcf(); plt.close("all")
            with conv_ph.container():
                st.pyplot(results["conv_fig"])
                st.caption(f"⏱ {fmt_time(timings['conv'])}")

        if compute_compare_methods:
            t0 = time.perf_counter()
            fig, _, _, _ = ExecutionTime(pricer, market, option).compare_methods(N)
            timings["compare"] = time.perf_counter() - t0
            results["compare_fig"] = fig; plt.close("all")
            with compare_ph.container():
                st.pyplot(fig)
                st.caption(f"⏱ {fmt_time(timings['compare'])}")

        if compute_exercise_prob and tree_obj is not None:
            t0 = time.perf_counter()
            fig, _, _ = tree_obj.plot_exercise_probability_by_date()
            timings["exercise_prob"] = time.perf_counter() - t0
            results["exercise_prob_fig"] = fig; plt.close("all")
            with exercise_prob_ph.container():
                st.pyplot(fig)
                st.caption(f"⏱ {fmt_time(timings['exercise_prob'])}")

        if compute_amer_vs_euro:
            t0 = time.perf_counter()
            try:
                fig, _, _ = american_european_gap_vs_rate(pricer, market, option, rate_min=rate_min, rate_max=rate_max, nb_points=rate_points, nb_steps_for_study=N)
                results["amer_euro_fig"] = fig; plt.close("all")
            except Exception as e:
                st.error(f"American vs European study failed: {e}")
            timings["amer_euro"] = time.perf_counter() - t0
            if "amer_euro_fig" in results:
                with amer_euro_ph.container():
                    st.pyplot(results["amer_euro_fig"])
                    st.caption(f"⏱ {fmt_time(timings['amer_euro'])}")

        if compute_avg_value and tree_obj is not None:
            t0 = time.perf_counter()
            fig, _, _ = tree_obj.plot_average_option_value_by_date()
            timings["avg_value"] = time.perf_counter() - t0
            results["avg_value_fig"] = fig; plt.close("all")
            with avg_value_ph.container():
                st.pyplot(fig)
                st.caption(f"⏱ {fmt_time(timings['avg_value'])}")

        if compute_price_vs_exdiv:
            t0 = time.perf_counter()
            try:
                fig, _, _, _, _ = price_vs_ex_div_date(pricer, market, option, nb_points=100, nb_steps_for_study=N)
                results["price_vs_exdiv_fig"] = fig; plt.close("all")
            except Exception as e:
                st.error(f"Price vs. ex-div study failed: {e}")
            timings["price_vs_exdiv"] = time.perf_counter() - t0
            if "price_vs_exdiv_fig" in results:
                with price_vs_exdiv_ph.container():
                    st.pyplot(results["price_vs_exdiv_fig"])
                    st.caption(f"⏱ {fmt_time(timings['price_vs_exdiv'])}")

        if compute_greeks_profile:
            t0 = time.perf_counter()
            try:
                fig = greeks_profiles(pricer, market, option, nb_points=15)
                results["greeks_profile_fig"] = fig; plt.close("all")
            except Exception as e:
                st.error(f"Greeks profiles failed: {e}")
            timings["greeks_profile"] = time.perf_counter() - t0
            if "greeks_profile_fig" in results:
                with greeks_profile_ph.container():
                    st.pyplot(results["greeks_profile_fig"])
                    st.caption(f"⏱ {fmt_time(timings['greeks_profile'])}")

        results["_timings"] = timings
        results["_params_snapshot"] = snap
        st.session_state.results = results
        if tree_obj is not None:
            st.session_state.debug_tree = tree_obj

    # ================= STATIC redisplay =================
    if not run and st.session_state.results is not None:
        results = st.session_state.results
        timings = results.get("_timings", {})
        st.divider()
        st.subheader("Results")
        snap = results.get("_params_snapshot")
        if snap is None:
            snap = st.session_state.get("_current_snapshot")
        if snap is not None:
            render_run_recap_from_snapshot(snap)

        if "tree_price" in results or "bs_price" in results:
            st.markdown("**Prices**")
            pc1, pc2 = st.columns(2)
            if "tree_price" in results:
                with pc1.container():
                    st.metric("Trinomial tree price", f"{results['tree_price']:.6f}")
                    if "tree_price" in timings: st.caption(f"⏱ {fmt_time(timings['tree_price'])}")
            if "bs_price" in results:
                with pc2.container():
                    st.metric("Black & Scholes price", f"{results['bs_price']:.6f}")
                    if "bs_price" in timings: st.caption(f"⏱ {fmt_time(timings['bs_price'])}")

        if "greeks_tree" in results or "greeks_bs" in results:
            st.markdown("**Greeks**")
            st.dataframe(greeks_dataframe(results), hide_index=True, use_container_width=True)
            gk = [k for k in ("greeks_tree", "greeks_bs") if k in timings]
            if gk: st.caption("⏱ " + " | ".join(f"{k}: {fmt_time(timings[k])}" for k in gk))

        if "strike_fig" in results:
            st.markdown("**Prices and slopes vs. strike**")
            st.pyplot(results["strike_fig"])
            if "strike" in timings: st.caption(f"⏱ {fmt_time(timings['strike'])}")

        if "error_curve_fig" in results:
            st.markdown("**Error vs. number of steps**")
            st.pyplot(results["error_curve_fig"])
            if "error_curve" in timings: st.caption(f"⏱ {fmt_time(timings['error_curve'])}")

        if "exec_fig" in results:
            st.markdown("**Execution time**")
            st.pyplot(results["exec_fig"])
            if "exec" in timings: st.caption(f"⏱ {fmt_time(timings['exec'])}")

        if "pruning_compare_fig" in results:
            st.markdown("**Pruning effect on runtime**")
            st.pyplot(results["pruning_compare_fig"])
            if "pruning_compare" in timings: st.caption(f"⏱ {fmt_time(timings['pruning_compare'])}")

        if "conv_fig" in results:
            st.markdown("**Convergence towards Black & Scholes**")
            st.pyplot(results["conv_fig"])
            if "conv" in timings: st.caption(f"⏱ {fmt_time(timings['conv'])}")

        if "compare_fig" in results:
            st.markdown("**Backward vs. recursive**")
            st.pyplot(results["compare_fig"])
            if "compare" in timings: st.caption(f"⏱ {fmt_time(timings['compare'])}")

        if "exercise_prob_fig" in results:
            st.markdown("**Exercise probability by dates**")
            st.pyplot(results["exercise_prob_fig"])
            if "exercise_prob" in timings: st.caption(f"⏱ {fmt_time(timings['exercise_prob'])}")

        if "amer_euro_fig" in results:
            st.markdown("**American vs. European vs. rate**")
            st.pyplot(results["amer_euro_fig"])
            if "amer_euro" in timings: st.caption(f"⏱ {fmt_time(timings['amer_euro'])}")

        if "avg_value_fig" in results:
            st.markdown("**Average option value by dates**")
            st.pyplot(results["avg_value_fig"])
            if "avg_value" in timings: st.caption(f"⏱ {fmt_time(timings['avg_value'])}")

        if "price_vs_exdiv_fig" in results:
            st.markdown("**Price vs. ex-div date**")
            st.pyplot(results["price_vs_exdiv_fig"])
            if "price_vs_exdiv" in timings: st.caption(f"⏱ {fmt_time(timings['price_vs_exdiv'])}")

        if "greeks_profile_fig" in results:
            st.markdown("**Greeks Profiles**")
            st.pyplot(results["greeks_profile_fig"])
            if "greeks_profile" in timings: st.caption(f"⏱ {fmt_time(timings['greeks_profile'])}")

        if st.session_state.get("debug") and st.session_state.debug_tree is not None:
            st.markdown("**Debug — tree construction**")
            render_debug_panel(st.session_state.debug_tree)

        if st.session_state.tree is not None:
            st.markdown("**Trinomial tree**")
            render_tree_content(st.session_state.tree, st.session_state.excel_bytes, int(max_display), elapsed=timings.get("tree_display"), static_mode=True)


if __name__ == "__main__":
    main()
