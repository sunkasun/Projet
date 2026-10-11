import time
import pandas as pd
from Market import Market
from OptionParameters import OptionParameters
from Tree import Tree
from Pricer import Pricer
from Extension import Extension
from Node import Node


def run_calculations(input_data: dict) -> dict:
    mkt = Market(input_data)
    params = OptionParameters(input_data)
    arbre = Tree()
    pricer = Pricer(input_data)
    Ext = Extension(mkt, params, pricer)

    # --- Pricing de base ---
    if input_data["BSCondition"]:
        t0 = time.time()
        Ext.black_sholes()
        pricer.BS_Time = time.time() - t0

    if input_data["MCCondition"]:
        t0 = time.time()
        Ext.monte_carlo_price(100000)
        pricer.MC_Time = time.time() - t0

    t0 = time.time()
    arbre.init(mkt, params, pricer, False)
    time_no_greeks = time.time() - t0
    results = dict(arbre.price_results)
    results["TimeNoGreeks"] = time_no_greeks

    if input_data["Convergence"]:
        results["ConvergenceFig"] = arbre.Convergence_fig
    if input_data["StrikeStudy"]:
        results["StrikeStudyFig"] = arbre.compute_StrikeStudy(10, mkt, pricer, params, Ext)
    if input_data["TreeBol"] and arbre.df is not None:
        results["TreeDf"] = arbre.df

    # --- Debug info ---
    if pricer.debug_mode:
        results["DebugInfo"] = arbre.debug_stats()

    # --- Greeks ---
    if input_data["ComputeGreeks"]:
        t0 = time.time()
        base_price = results.get("TreePrice", 0.0)
        tree_g = arbre.compute_tree_greeks(mkt, params, pricer, base_price)
        bs_g = Ext.bs_greeks() if input_data["BSCondition"] else None
        mc_g = Ext.monte_carlo_greeks(50000) if input_data["MCCondition"] else None

        greek_names = ["Delta", "Gamma", "Vega", "Vomma", "Vanna", "Theta"]
        cols = {"Tree": tree_g}
        if bs_g:
            cols["Black-Scholes"] = bs_g
        if mc_g:
            cols["Monte-Carlo"] = mc_g
        greeks_df = pd.DataFrame(
            {name: [cols[c][name] for c in cols] for name in greek_names},
            index=list(cols.keys())).T
        results["GreeksTable"] = greeks_df
        results["TimeGreeks"] = time.time() - t0

    return results