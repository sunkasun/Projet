import math
import time
import numpy as np
import pandas as pd
from Node import Node
from Market import Market
from OptionParameters import OptionParameters
from Pricer import Pricer
from Extension import Extension


class Tree:
    def __init__(self):
        self.dt = 0.0
        self.Alpha = 1.0
        self.sqrt_alpha = 1.0
        self.Root_Node = None
        self.Market = None
        self.candidateMid = None
        self.Final_Node = None
        self.tronc = None
        self.ModifExMid = False
        self.isTrunc = True
        self.Convergence_results = None
        self.price_results = {}
        self.nnext = self.nUp = self.nDown = None
        self.df = None
        self.Convergence_fig = None
        self.time_calculation_tree = 0.0
        self.last_tree_type = "?"
        self.binom_band = None
        self.binom_nodes = None

    # ============================================================
    #                         INIT
    # ============================================================
    def init(self, mkt, params, pricer, IsVega):
        self.Market = mkt
        if pricer.Convergence or pricer.StrikeStudy:
            _ext = Extension(mkt, params, pricer)
            _ext.black_sholes()

        if pricer.Convergence:
            for T in range(1, pricer.timeSteps + 1):
                print(f"Calcul du prix pour {T} Time Steps")
                self.build_tree(mkt, params, pricer, T, pricer.BS_price, True, IsVega)
            self.Convergence_fig = pricer.create_plot_convergence()
        else:
            if pricer.gap != 0:
                pricer.newTs = pricer.compute_time_steps(mkt, params)
                self.build_tree(mkt, params, pricer, pricer.newTs, pricer.BS_price, False, IsVega)
            else:
                self.build_tree(mkt, params, pricer, pricer.timeSteps, pricer.BS_price, False, IsVega)

    # ============================================================
    #                       DISPATCHER
    # ============================================================
    def build_tree(self, mkt, params, pricer, T, BS_Price, condition_Convergence, isVega):
        tree_type = getattr(pricer, "TreeType", "Auto")
        ttm = (params.DateMaturity - pricer.PricingDate).days / 365
        has_div = (mkt.dividend > 0) and (0 < mkt.div_date < ttm)

        if tree_type == "Binomial" and has_div:
            print("Warning: Binomial sélectionné mais dividende discret → Trinomial forcé")
            tree_type = "Trinomial"
        elif tree_type.startswith("Auto"):
            tree_type = "Trinomial" if has_div else "Binomial"

        self.last_tree_type = tree_type

        if tree_type == "Binomial":
            return self._build_binomial_tree(mkt, params, pricer, T, BS_Price,
                                             condition_Convergence, isVega)
        return self._build_trinomial_tree(mkt, params, pricer, T, BS_Price,
                                          condition_Convergence, isVega)

    # ============================================================
    #                       HELPERS
    # ============================================================
    def make_node(self, market_values: float) -> Node:
        new_node = Node()
        new_node.init_node(market_values)
        return new_node

    def display_results(self, pricer, time_calculation, price, time_calculation_tree, newTs):
        self.price_results = {"TreePrice": price, "TimePricing": time_calculation}
        if newTs != 0:
            self.price_results["New TS"] = newTs
        if pricer.display_tree_bool:
            self.price_results["Tree Display Time"] = time_calculation_tree
        if pricer.BS_condition:
            self.price_results['BS Price'] = pricer.BS_price
            self.price_results['BS_Time'] = pricer.BS_Time
            self.price_results['TreeGapBS'] = price - pricer.BS_price
            self.price_results['TreeGapBSTime'] = time_calculation - pricer.BS_Time
        if pricer.MC_condition:
            self.price_results['MC Price'] = pricer.MC_price
            self.price_results['MCTime'] = pricer.MC_Time
            self.price_results['MCGapTree'] = pricer.MC_price - price
            self.price_results['MCGapTreeTime'] = time_calculation - pricer.MC_Time

    # ============================================================
    #                    TRINOMIAL
    # ============================================================
    def _build_trinomial_tree(self, mkt, params, pricer, T, BS_Price,
                              condition_Convergence, isVega):
        start_time = time.time()
        self.init_root_node_and_parameters(mkt, params, pricer, T)

        for i in range(1, T + 1):
            self.build_columns(self.candidateMid, mkt, params, i, pricer)

        if pricer.display_tree_bool:
            if T <= 500:
                self.time_calculation_tree = self.display_tree(pricer)
            else:
                print(f"⚠ Affichage arbre trinomial désactivé pour T={T} (> 500)")

        self.candidateMid.price(self.candidateMid, params, mkt, self)
        price = self.Root_Node.OptPrice
        time_calculation = time.time() - start_time

        if pricer.compute_DeltaAndGamma_tree_var:
            pricer.DeltaTree = self.Compute_DeltaTree(
                self.Root_Node.UpNode.OptPrice, self.Root_Node.DownNode.OptPrice, mkt.SpotPrice)
            pricer.GammaTree = self.Compute_GammaTree(
                price, self.Root_Node.UpNode.OptPrice,
                self.Root_Node.DownNode.OptPrice, mkt.SpotPrice)
            # Détache les nœuds fake après extraction
            self.Root_Node.UpNode = None
            self.Root_Node.DownNode = None

        if condition_Convergence:
            mc_p = pricer.MC_price if pricer.MC_condition else None
            pricer.compute_perf(T, price, time_calculation, BS_Price, mc_p)
        if not isVega:
            self.display_results(pricer, time_calculation, price,
                                 self.time_calculation_tree, pricer.newTs)
        return self.Root_Node

    def init_root_node_and_parameters(self, mkt, params, pricer, T):
        self.dt = ((params.DateMaturity - pricer.PricingDate).days / T) / 365
        self.Alpha = math.exp(mkt.volatility * math.sqrt(3 * self.dt))
        self.sqrt_alpha = math.sqrt(self.Alpha)
        self.Root_Node = self.make_node(mkt.SpotPrice)

        if pricer.compute_DeltaAndGamma_tree_var:
            NodeCalcul = self.make_node(mkt.SpotPrice)
            NodeCalcul.Next_Mid = self.make_node(NodeCalcul.forward(mkt, self.dt, 0))
            self.compute_proba(NodeCalcul, NodeCalcul.Next_Mid, mkt, 0)
            self.Root_Node.UpNode = self.make_node(mkt.SpotPrice * self.Alpha)
            self.Root_Node.DownNode = self.make_node(mkt.SpotPrice / self.Alpha)
            self.Root_Node.Cum_Proba = NodeCalcul.Proba_Mid
            self.Root_Node.UpNode.Cum_Proba = NodeCalcul.Proba_Up
            self.Root_Node.DownNode.Cum_Proba = NodeCalcul.Proba_Down
        else:
            self.Root_Node.Cum_Proba = 1
        self.candidateMid = self.Root_Node

    def build_triple(self, candidateMid, mkt, div):
        prix_forward = candidateMid.forward(mkt, self.dt, div)
        self.build_mid(candidateMid, prix_forward, mkt, div)
        self.link_and_new_node(prix_forward * self.Alpha, candidateMid, self.nnext, True)
        self.link_and_new_node(prix_forward / self.Alpha, candidateMid, self.nnext, False)

    def build_mid(self, candidateMid, prix_forward, mkt, div):
        self.nnext = self.make_node(prix_forward)
        candidateMid.Next_Mid = self.nnext
        self.nnext.nprec = candidateMid
        self.compute_proba(candidateMid, self.nnext, mkt, div)
        candidateMid.Next_Mid.Cum_Proba = candidateMid.Cum_Proba * candidateMid.Proba_Mid

    def link_and_new_node(self, prix, candidateMid, nnext, upper):
        newnode = self.make_node(prix)
        if upper:
            newnode.Cum_Proba = candidateMid.Cum_Proba * candidateMid.Proba_Up
            newnode.DownNode = nnext
            nnext.UpNode = newnode
            candidateMid.Next_Up = newnode
            self.nUp = newnode
        else:
            newnode.Cum_Proba = candidateMid.Cum_Proba * candidateMid.Proba_Down
            newnode.UpNode = nnext
            nnext.DownNode = newnode
            candidateMid.Next_Down = newnode
            self.nDown = newnode

    def build_columns(self, candidateMid, mkt, params, i, pricer):
        div = mkt.compute_dividend(i, self.dt)
        self.isTrunc = True
        self.build_triple(candidateMid, mkt, div)
        self.tronc = candidateMid
        self.build_up_and_down(candidateMid, mkt, params, pricer, div, "up")
        if not self.isTrunc:
            self.nUp = candidateMid.Next_Up
            self.nnext = candidateMid.Next_Mid
            self.nDown = candidateMid.Next_Down
            self.build_up_and_down(candidateMid, mkt, params, pricer, div, "down")
        candidateMid = self.tronc
        self.candidateMid = candidateMid.Next_Mid

    def build_up_and_down(self, candidateMid, mkt, params, pricer, div, direction):
        if direction == "up":
            while candidateMid.UpNode is not None:
                self.isTrunc = False
                self.next_forward(candidateMid.UpNode, mkt, params, pricer, direction, div)
                candidateMid = candidateMid.UpNode
        elif direction == "down":
            while candidateMid.DownNode is not None:
                self.next_forward(candidateMid.DownNode, mkt, params, pricer, direction, div)
                candidateMid = candidateMid.DownNode

    def next_forward(self, candidateMid, mkt, params, pricer, direction, div):
        isUp = (direction == "up")
        if candidateMid.Cum_Proba > pricer.PruningTreshold:
            self.ModifExMid = False
            self.chek_and_link(candidateMid, mkt, direction, div, isUp, False)
            self.compute_and_allocating_probas(candidateMid, mkt, div, isUp)
            self.new_and_link(candidateMid, direction, False)
        else:
            self.chek_and_link(candidateMid, mkt, direction, div, isUp, True)

    def chek_and_link(self, candidateMid, mkt, direction, div, isUp, isprunne):
        if isUp:
            if div != 0:
                self.controle_next_mid(candidateMid, self.nUp, direction, mkt, div)
            if not self.ModifExMid and not isprunne:
                candidateMid.Next_Mid = self.nUp
                candidateMid.Next_Down = self.nnext
            elif not self.ModifExMid and isprunne:
                candidateMid.Next_Mid = self.nUp
                candidateMid.Proba_Mid = 1
        else:
            if div != 0:
                self.controle_next_mid(candidateMid, self.nDown, direction, mkt, div)
            if not self.ModifExMid and not isprunne:
                candidateMid.Next_Mid = self.nDown
                candidateMid.Next_Up = self.nnext
            elif not self.ModifExMid and isprunne:
                candidateMid.Next_Mid = self.nDown
                candidateMid.Proba_Mid = 1

    def compute_and_allocating_probas(self, candidateMid, mkt, div, isUp):
        self.compute_proba(candidateMid, candidateMid.Next_Mid, mkt, div)
        candidateMid.Next_Mid.Cum_Proba += candidateMid.Cum_Proba * candidateMid.Proba_Mid
        if isUp:
            candidateMid.Next_Down.Cum_Proba += candidateMid.Cum_Proba * candidateMid.Proba_Down
        else:
            candidateMid.Next_Up.Cum_Proba += candidateMid.Cum_Proba * candidateMid.Proba_Up

    def compute_transition_up_node(self, newnode, ModifExMid, IsDiveTime):
        if IsDiveTime:
            self.nnext = newnode
            self.nDown = self.nUp
            self.nUp = self.nnext
        elif not ModifExMid:
            self.nDown = self.nnext
            self.nnext = self.nUp
            self.nUp = newnode
        else:
            self.nUp = newnode

    def compute_transition_down_node(self, newnode, ModifExMid, IsDiveTime):
        if IsDiveTime:
            self.nnext = newnode
            self.nUp = self.nDown
            self.nDown = self.nnext
        elif not ModifExMid:
            self.nUp = self.nnext
            self.nnext = self.nDown
            self.nDown = newnode
        else:
            self.nDown = newnode

    def build_up(self, candidate_mid, is_dive_time):
        newnode = self.make_node(self.nUp.UndPrice * self.Alpha)
        if is_dive_time:
            candidate_mid.Next_Mid = newnode
            self.nUp.UpNode = newnode
            candidate_mid.Next_Down = self.nUp
            candidate_mid.Next_Down.UpNode = newnode
        else:
            candidate_mid.Next_Up = newnode
            candidate_mid.Next_Mid.UpNode = newnode
            candidate_mid.Next_Up.DownNode = self.nUp
        self.compute_transition_up_node(newnode, self.ModifExMid, is_dive_time)

    def build_down(self, candidate_mid, is_dive_time):
        newnode = self.make_node(self.nDown.UndPrice / self.Alpha)
        if is_dive_time:
            candidate_mid.Next_Mid = newnode
            candidate_mid.Next_Mid.UpNode = self.nDown
            candidate_mid.Next_Up = self.nDown
            candidate_mid.Next_Up.DownNode = newnode
        else:
            candidate_mid.Next_Down = newnode
            candidate_mid.Next_Mid.DownNode = newnode
            candidate_mid.Next_Down.UpNode = self.nDown
        self.compute_transition_down_node(newnode, self.ModifExMid, is_dive_time)

    def new_and_link(self, candidate_mid, direction, is_dive_time):
        is_up = (direction == "up")
        if is_up:
            self.build_up(candidate_mid, is_dive_time)
        else:
            self.build_down(candidate_mid, is_dive_time)
        if not is_dive_time:
            if is_up:
                candidate_mid.Next_Up.Cum_Proba = candidate_mid.Cum_Proba * candidate_mid.Proba_Up
            else:
                candidate_mid.Next_Down.Cum_Proba = candidate_mid.Cum_Proba * candidate_mid.Proba_Down

    def controle_next_mid(self, candidate_mid, potential_mid, direction, mkt, div):
        fwd = candidate_mid.forward(mkt, self.dt, div)
        if (fwd < (potential_mid.UndPrice * (1 + (1 / self.Alpha))) / 2
                or fwd > (potential_mid.UndPrice * (1 + self.Alpha)) / 2):
            self.new_and_link(candidate_mid, direction, True)
            self.ModifExMid = True

    def compute_proba(self, candidateMid, nnext, mkt, div):
        esperance = candidateMid.UndPrice * np.exp(mkt.RiskFree * self.dt) - div
        variance = (candidateMid.UndPrice ** 2) * np.exp(2 * mkt.RiskFree * self.dt) * (
                np.exp(mkt.volatility ** 2 * self.dt) - 1)

        candidateMid.Proba_Down = ((nnext.UndPrice ** -2 * (variance + esperance ** 2) - 1 -
                                   (self.Alpha + 1) * (nnext.UndPrice ** -1 * esperance - 1)) /
                                   ((1 - self.Alpha) * (self.Alpha ** -2 - 1)))
        candidateMid.Proba_Up = (nnext.UndPrice ** -1 * esperance - 1 - (self.Alpha ** -1 - 1) *
                                 candidateMid.Proba_Down) / (self.Alpha - 1)
        candidateMid.Proba_Mid = 1 - candidateMid.Proba_Down - candidateMid.Proba_Up

        if candidateMid.Proba_Down < -1e-12 or candidateMid.Proba_Up < -1e-12 \
                or candidateMid.Proba_Mid < -1e-12:
            print(f"⚠ Probabilités négatives : up={candidateMid.Proba_Up:.6f} "
                  f"mid={candidateMid.Proba_Mid:.6f} down={candidateMid.Proba_Down:.6f}")

    def Compute_DeltaTree(self, P1, P_1, S0):
        return (P1 - P_1) / ((self.Alpha * S0) - (S0 / self.Alpha))

    def Compute_GammaTree(self, P, P1, P_1, S0):
        return ((((P1 - P) / ((self.Alpha * S0) - S0)) - ((P - P_1) / (S0 - (S0 / self.Alpha))))
                / (((self.Alpha * S0) - (S0 / self.Alpha)) / 2))

    # ============================================================
    #                   BINOMIAL (band + rolling)
    # ============================================================
    def _build_binomial_tree(self, mkt, params, pricer, T, BS_Price,
                             condition_Convergence, isVega):
        start_time = time.time()

        self.dt = ((params.DateMaturity - pricer.PricingDate).days / T) / 365
        exp_s2dt = math.exp(mkt.volatility ** 2 * self.dt)
        half = (1 + exp_s2dt) / 2
        self.sqrt_alpha = half + math.sqrt(max(half ** 2 - 1.0, 0.0))
        self.Alpha = self.sqrt_alpha

        p_up = 1.0 / (self.sqrt_alpha + 1.0)
        p_down = self.sqrt_alpha / (self.sqrt_alpha + 1.0)
        exp_r_dt = math.exp(mkt.RiskFree * self.dt)
        disc = math.exp(-mkt.RiskFree * self.dt)
        S0 = mkt.SpotPrice
        threshold = max(getattr(pricer, "PruningTreshold", 0.0), 0.0)

        band = [[] for _ in range(T + 1)]
        cum = {(0, 0): 1.0}
        band[0] = [0]

        for i in range(1, T + 1):
            for k in range(i + 1):
                c = 0.0
                if (i - 1, k - 1) in cum:
                    c += cum[(i - 1, k - 1)] * p_up
                if (i - 1, k) in cum:
                    c += cum[(i - 1, k)] * p_down
                if i < T and c < threshold:
                    continue
                band[i].append(k)
                cum[(i, k)] = c
            if i >= 2:
                for key in [key for key in cum if key[0] <= i - 2]:
                    del cum[key]

        def price_at(i, k):
            return (S0 * (self.sqrt_alpha * exp_r_dt) ** k
                    * (exp_r_dt / self.sqrt_alpha) ** (i - k))

        V_prev = {k: params.Vi(price_at(T, k)) for k in band[T]}

        for i in range(T - 1, -1, -1):
            V_curr = {}
            for k in band[i]:
                has_up = (k + 1) in V_prev
                has_down = k in V_prev
                if has_up and has_down:
                    val = disc * (p_up * V_prev[k + 1] + p_down * V_prev[k])
                elif has_up:
                    val = disc * V_prev[k + 1]
                elif has_down:
                    val = disc * V_prev[k]
                else:
                    val = 0.0
                if params.Exercice == "American":
                    val = max(val, params.Vi(price_at(i, k)))
                V_curr[k] = val
            V_prev = V_curr

        price = V_prev.get(0, 0.0)
        time_calculation = time.time() - start_time

        need_nodes = pricer.compute_DeltaAndGamma_tree_var or pricer.display_tree_bool
        can_materialize = need_nodes and T <= 2000
        if can_materialize:
            self._materialize_binomial(band, price_at, p_up, p_down, params, mkt)
        elif need_nodes:
            print(f"⚠ Matérialisation binomiale désactivée pour T={T} (> 2000)")

        if pricer.display_tree_bool and T <= 2000:
            self.time_calculation_tree = self.display_tree(pricer)

        if pricer.compute_DeltaAndGamma_tree_var and self.Root_Node is not None \
                and self.Root_Node.Next_Up is not None:
            P_up = self.Root_Node.Next_Up.OptPrice
            P_down = self.Root_Node.Next_Down.OptPrice
            S_up = self.Root_Node.Next_Up.UndPrice
            S_down = self.Root_Node.Next_Down.UndPrice
            pricer.DeltaTree = (P_up - P_down) / (S_up - S_down)
            pricer.GammaTree = 2 * ((P_up - price) / (S_up - S0)
                                    - (price - P_down) / (S0 - S_down)) / (S_up - S_down)

        if condition_Convergence:
            mc_p = pricer.MC_price if pricer.MC_condition else None
            pricer.compute_perf(T, price, time_calculation, BS_Price, mc_p)
        if not isVega:
            self.display_results(pricer, time_calculation, price,
                                 self.time_calculation_tree, pricer.newTs)

        self.binom_band = band
        if self.Root_Node is None:
            self.Root_Node = self.make_node(price)
        self.candidateMid = self.Root_Node
        return self.Root_Node

    def _materialize_binomial(self, band, price_at, p_up, p_down, params, mkt):
        T = len(band) - 1
        nodes = {}
        root = self.make_node(price_at(0, 0))
        root.Proba_Up, root.Proba_Down, root.Proba_Mid = p_up, p_down, 0.0
        root.Cum_Proba = 1.0
        nodes[(0, 0)] = root
        self.Root_Node = root

        for i in range(1, T + 1):
            for k in band[i]:
                n = self.make_node(price_at(i, k))
                n.Proba_Up, n.Proba_Down, n.Proba_Mid = p_up, p_down, 0.0
                n.Cum_Proba = 0.0
                nodes[(i, k)] = n
                if (i - 1, k - 1) in nodes:
                    nodes[(i - 1, k - 1)].Next_Up = n
                    n.Cum_Proba += nodes[(i - 1, k - 1)].Cum_Proba * p_up
                if (i - 1, k) in nodes:
                    nodes[(i - 1, k)].Next_Down = n
                    n.Cum_Proba += nodes[(i - 1, k)].Cum_Proba * p_down

        disc = math.exp(-mkt.RiskFree * self.dt)
        for k in band[T]:
            nodes[(T, k)].OptPrice = params.Vi(nodes[(T, k)].UndPrice)
        for i in range(T - 1, -1, -1):
            for k in band[i]:
                n = nodes[(i, k)]
                up, down = n.Next_Up, n.Next_Down
                if up and down:
                    v = disc * (p_up * up.OptPrice + p_down * down.OptPrice)
                elif up:
                    v = disc * up.OptPrice
                elif down:
                    v = disc * down.OptPrice
                else:
                    v = 0.0
                if params.Exercice == "American":
                    v = max(v, params.Vi(n.UndPrice))
                n.OptPrice = v
        self.binom_nodes = nodes

    # ============================================================
    #                 TREE GREEKS + CUT TREE
    # ============================================================
    def compute_tree_greeks(self, mkt, params, pricer, base_price):
        from datetime import timedelta
        S0, sigma = mkt.SpotPrice, mkt.volatility
        M0 = params.DateMaturity

        saved = (pricer.Convergence, pricer.display_tree_bool, pricer.BS_condition,
                 pricer.MC_condition, pricer.compute_DeltaAndGamma_tree_var,
                 pricer.compute_Greeks)
        pricer.Convergence = False
        pricer.display_tree_bool = False
        pricer.BS_condition = False
        pricer.MC_condition = False
        pricer.compute_DeltaAndGamma_tree_var = False
        pricer.compute_Greeks = False

        def price_with(S=S0, sig=sigma, mat=M0):
            mkt.SpotPrice, mkt.volatility, params.DateMaturity = S, sig, mat
            self.build_tree(mkt, params, pricer, pricer.timeSteps, 0.0, False, True)
            return self.Root_Node.OptPrice if self.Root_Node else 0.0

        P0 = base_price
        h_S, h_sig = S0 * 0.01, 0.01
        P_Su = price_with(S=S0 + h_S); P_Sd = price_with(S=S0 - h_S)
        P_su = price_with(sig=sigma + h_sig); P_sd = price_with(sig=sigma - h_sig)
        P_uu = price_with(S=S0 + h_S, sig=sigma + h_sig)
        P_ud = price_with(S=S0 + h_S, sig=sigma - h_sig)
        P_du = price_with(S=S0 - h_S, sig=sigma + h_sig)
        P_dd = price_with(S=S0 - h_S, sig=sigma - h_sig)
        P_sh = price_with(mat=M0 - timedelta(days=1))

        mkt.SpotPrice, mkt.volatility, params.DateMaturity = S0, sigma, M0
        (pricer.Convergence, pricer.display_tree_bool, pricer.BS_condition,
         pricer.MC_condition, pricer.compute_DeltaAndGamma_tree_var,
         pricer.compute_Greeks) = saved

        return {
            "Delta": (P_Su - P_Sd) / (2 * h_S),
            "Gamma": (P_Su - 2 * P0 + P_Sd) / (h_S ** 2),
            "Vega": (P_su - P_sd) / (2 * h_sig) / 100,
            "Vomma": (P_su - 2 * P0 + P_sd) / (h_sig ** 2) / 10000,
            "Vanna": (P_uu - P_ud - P_du + P_dd) / (4 * h_S * h_sig) / 100,
            "Theta": P_sh - P0,
        }

    def cut_tree(self):
        seen = set()

        def _cut(n):
            if n is None or id(n) in seen:
                return
            seen.add(id(n))
            for attr in ("Next_Up", "Next_Down", "Next_Mid",
                         "UpNode", "DownNode", "MidNode", "nprec"):
                child = getattr(n, attr, None)
                setattr(n, attr, None)
                _cut(child)

        _cut(getattr(self, "Root_Node", None))
        self.binom_nodes = None
        self.Root_Node = None
        self.candidateMid = None

    def debug_stats(self):
        stats = {
            "tree_type": self.last_tree_type,
            "dt_years": self.dt,
            "sqrt_alpha": getattr(self, "sqrt_alpha", None),
        }
        if Node.COUNTING:
            stats["nb_alloc_total"] = Node.nb_alloc
            stats["nb_nodes_alive_now"] = Node.live_count()
        if self.Root_Node is not None:
            p = [self.Root_Node.Proba_Up,
                 self.Root_Node.Proba_Mid,
                 self.Root_Node.Proba_Down]
            stats["p_root_up_mid_down"] = p
            stats["p_root_sum"] = float(np.sum(p))
            stats["p_root_min"] = float(np.min(p))
        return stats

    # ============================================================
    #                       VEGA + STRIKE STUDY
    # ============================================================
    def compute_vega_tree(self, params, pricer, mkt):
        greeks = self.compute_tree_greeks(
            mkt, params, pricer,
            self.Root_Node.OptPrice if self.Root_Node else 0.0)
        return greeks["Vega"]

    def compute_StrikeStudy(self, StrikeSteps, mkt, pricer, params, Extension):
        df_strike = pd.DataFrame(
            np.nan, index=range(StrikeSteps),
            columns=["Strike", "Tree Price", "BS Price", "MC Price",
                     "Tree-BS", "Tree-MC", "Slope BS", "Slope Tree", "Slope MC"])

        saved_conv = pricer.Convergence
        saved_disp = pricer.display_tree_bool
        saved_strike = params.strike
        pricer.Convergence = False
        pricer.display_tree_bool = False

        params.strike = saved_strike - 5
        T = StrikeSteps
        for i in range(1, StrikeSteps + 1):
            params.strike += 1
            self.build_tree(mkt, params, pricer, pricer.timeSteps,
                            pricer.BS_price, True, False)
            Tree_price = self.Root_Node.OptPrice if self.Root_Node else 0.0
            BS_Price = Extension.black_sholes()["BS Price"]

            MC_Price = np.nan
            if pricer.MC_condition:
                MC_Price = Extension.monte_carlo_price(50000)

            df_strike.at[i, "Strike"] = params.strike
            df_strike.at[i, "Tree Price"] = Tree_price
            df_strike.at[i, "BS Price"] = BS_Price
            df_strike.at[i, "MC Price"] = MC_Price
            df_strike.at[i, "Tree-BS"] = Tree_price - BS_Price
            if not np.isnan(MC_Price):
                df_strike.at[i, "Tree-MC"] = Tree_price - MC_Price

        pricer.Convergence = saved_conv
        pricer.display_tree_bool = saved_disp
        params.strike = saved_strike

        df_strike = self.compute_slope(df_strike, StrikeSteps,
                                       with_mc=pricer.MC_condition)
        return pricer.create_plot_StrikeStudy(
            df_strike, df_strike.at[2, "Strike"], df_strike.at[T, "Strike"],
            with_mc=pricer.MC_condition)

    def compute_slope(self, df_strike, StrikeSteps, with_mc=False):
        for T in range(2, StrikeSteps):
            dS = df_strike.at[T + 1, "Strike"] - df_strike.at[T - 1, "Strike"]
            df_strike.at[T, "Slope BS"] = (
                (df_strike.at[T + 1, "BS Price"] - df_strike.at[T - 1, "BS Price"]) / dS)
            df_strike.at[T, "Slope Tree"] = (
                (df_strike.at[T + 1, "Tree Price"] - df_strike.at[T - 1, "Tree Price"]) / dS)
            if with_mc:
                df_strike.at[T, "Slope MC"] = (
                    (df_strike.at[T + 1, "MC Price"] - df_strike.at[T - 1, "MC Price"]) / dS)
        return df_strike

    # ============================================================
    #                         DISPLAY
    # ============================================================
    def display_tree(self, pricer):
        """Parcours BFS par colonne en utilisant UNIQUEMENT les liens
        Next_Up / Next_Down / Next_Mid (inter-colonnes). On n'utilise
        PLUS UpNode/DownNode (intra-colonne) qui peuvent être bricolés
        par la logique de dive du tronc.

        Résultat garanti : colonne i contient exactement les nœuds
        atteignables depuis la racine en i sauts, sans doublons.
        """
        if getattr(self, "binom_nodes", None) is not None and self.last_tree_type == "Binomial":
            return self._display_binomial_tree(pricer)

        start_time_tree = time.time()
        max_depth = pricer.timeSteps

        # BFS : colonne i = ensemble des nœuds atteignables en i pas
        columns = {0: [self.Root_Node]}
        current = [self.Root_Node]

        for step in range(1, max_depth + 1):
            nxt = []
            seen = set()
            for n in current:
                if n is None:
                    continue
                for child in (getattr(n, "Next_Up", None),
                              getattr(n, "Next_Mid", None),
                              getattr(n, "Next_Down", None)):
                    if child is not None and id(child) not in seen:
                        seen.add(id(child))
                        nxt.append(child)
            if not nxt:
                break
            columns[step] = nxt
            current = nxt

        # ---------- Tri vertical pour un rendu lisible ----------
        # Pour chaque colonne, on trie par UndPrice décroissant.
        # Row 0 = plus haut. On centre sur le tronc (le mid).
        max_cols = max(columns.keys())
        rows_data = {}  # (row, col) -> UndPrice
        for step, nodes in columns.items():
            # On trie par prix décroissant
            sorted_nodes = sorted(nodes, key=lambda x: -x.UndPrice)
            n = len(sorted_nodes)
            # La row 0 est le prix le plus haut, on descend
            for i, node in enumerate(sorted_nodes):
                rows_data[(i, step)] = node.UndPrice

        if not rows_data:
            self.df = pd.DataFrame()
            return time.time() - start_time_tree

        max_row = max(r for (r, c) in rows_data)
        self.df = pd.DataFrame(index=range(max_row + 1),
                               columns=range(max_cols + 1),
                               dtype=object)
        for (r, c), v in rows_data.items():
            self.df.iloc[r, c] = v

        return time.time() - start_time_tree

    def _display_binomial_tree(self, pricer):
        start = time.time()
        nodes = self.binom_nodes
        T = max(i for (i, k) in nodes.keys())
        self.df = pd.DataFrame(index=range(T + 1), columns=range(T + 1), dtype=object)
        for (i, k), n in nodes.items():
            self.df.iloc[T - k, i] = n.UndPrice
        return time.time() - start