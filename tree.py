from __future__ import annotations

import bisect
import datetime as dt
import io
import time
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
import plotly.graph_objects as go
from typing import List, Tuple

from node import Node, TruncNode
from parameters import PricerParameters
from market import Market
from option import Option


def _thermal_color(t: float) -> str:
    """Échelle thermique bleu → cyan → vert → jaune → rouge. t ∈ [0, 1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        r, g, b = 0, int(255 * (t / 0.25)), 255
    elif t < 0.5:
        r, g, b = 0, 255, int(255 * (1 - (t - 0.25) / 0.25))
    elif t < 0.75:
        r, g, b = int(255 * ((t - 0.5) / 0.25)), 255, 0
    else:
        r, g, b = 255, int(255 * (1 - (t - 0.75) / 0.25)), 0
    return f"rgb({r},{g},{b})"


class Tree:
    def __init__(self, pricer_parameters: PricerParameters,
                 market_data: Market, option_data: Option,
                 debug: bool = False) -> None:
        self.pricer_parameters = pricer_parameters
        self.market_data = market_data
        self.option_data = option_data

        self.nb_steps = int(pricer_parameters.nb_steps)
        self.T = (option_data.maturity - pricer_parameters.pricing_date).days / 365
        self.delta_t = self.T / self.nb_steps if self.nb_steps > 0 else 0.0
        self.df = np.exp(-market_data.rate * self.delta_t)

        x = np.exp(market_data.vol ** 2 * self.delta_t) - 1.0
        self.sqrt_alpha_bin = 1.0 + x / 2.0 + np.sqrt(max(x * (1.0 + x / 4.0), 0.0))
        self.alpha_bin = self.sqrt_alpha_bin ** 2
        self.alpha_tri = np.exp(market_data.vol * np.sqrt(3.0 * self.delta_t))

        self.tree_mode = pricer_parameters.tree_mode
        self.alpha = self.alpha_tri if self.tree_mode == "trinomial_everywhere" \
                     else self.alpha_bin

        self.nodes_by_date: List[List[Node]] = []
        self.root: Node = None
        self.last: Node = None

        self.debug = debug
        self.counters = {"nodes_created": 0, "merges": 0, "clamped_probas": 0}
        self.step_times: List[float] = []
        self.step_nodes: List[int] = []
        self.step_candidates: List[int] = []
        self.step_modes: List[str] = []
        self.build_time = 0.0
        self.backward_time = 0.0
        self.price_time = 0.0
        self.recursion_failed = False    # <-- NOUVEAU

    def _forward_from(self, date: dt.datetime, S_start: float) -> float:
        """
        Calcule le forward du tronc partant de S_start à `date` sur un pas
        Δt, en intégrant le dividende s'il tombe dans (date, date+Δt].
        """
        r = self.market_data.rate
        D = self.market_data.dividend
        ex_div = self.market_data.ex_div_date
        dt_ = self.delta_t

        E_pur = S_start * np.exp(r * dt_)
        if D > 0 and (date < ex_div <= date + dt.timedelta(days=dt_ * 365)):
            tau_days = (ex_div - date).total_seconds() / 86400
            tau_ratio = tau_days / (dt_ * 365)
            tau_ratio = max(0.0, min(1.0, tau_ratio))
            return E_pur - D * np.exp(r * dt_ * (1.0 - tau_ratio))
        return E_pur

    # ==================================================================
    def _forward_from(self, date: dt.datetime, S_start: float) -> float:
        r = self.market_data.rate
        D = self.market_data.dividend
        ex_div = self.market_data.ex_div_date
        dt_ = self.delta_t
        E_pur = S_start * np.exp(r * dt_)
        if D > 0 and (date < ex_div <= date + dt.timedelta(days=dt_ * 365)):
            tau_days = (ex_div - date).total_seconds() / 86400
            tau_ratio = tau_days / (dt_ * 365) if dt_ * 365 > 0 else 1.0
            tau_ratio = max(0.0, min(1.0, tau_ratio))
            return E_pur - D * np.exp(r * dt_ * (1.0 - tau_ratio))
        return E_pur

    def _nearest_grid_position(self, expected: float, forward: float, alpha: float) -> int:
        if expected <= 0 or forward <= 0:
            return 0
        log_alpha = np.log(alpha) if abs(alpha - 1.0) > 1e-10 else 0.0
        if abs(log_alpha) < 1e-12:
            return 0
        return int(round(np.log(expected / forward) / log_alpha))

    def _find_or_create_child(self, price: float, current_ref: list,
                              next_date: dt.datetime) -> Node:
        # Cas 1 : aucun enfant créé à ce niveau
        if current_ref[0] is None:
            child = Node(price, self, next_date)
            self.counters["nodes_created"] += 1
            current_ref[0] = child
            return child

        current = current_ref[0]
        tol = 1e-9 * max(abs(price), 1.0)

        # Cas 2 : on avance dans la bonne direction depuis le dernier nœud
        if current.und_price < price:
            # Remonter
            while (current.up_node is not None
                   and current.up_node.und_price <= price + tol):
                current = current.up_node
            if abs(current.und_price - price) < tol:
                current_ref[0] = current
                self.counters["merges"] = self.counters.get("merges", 0) + 1
                return current
            lower = current
            upper = current.up_node
        else:
            # Descendre
            while (current.down_node is not None
                   and current.down_node.und_price >= price - tol):
                current = current.down_node
            if abs(current.und_price - price) < tol:
                current_ref[0] = current
                self.counters["merges"] = self.counters.get("merges", 0) + 1
                return current
            lower = current.down_node
            upper = current

        # Cas 3 : insérer entre lower et upper
        child = Node(price, self, next_date)
        self.counters["nodes_created"] += 1
        child.down_node = lower
        child.up_node = upper
        if lower is not None:
            lower.up_node = child
        if upper is not None:
            upper.down_node = child
        current_ref[0] = child
        return child

    def _branch_parent(self, parent: Node, E_trunk: float, alpha: float,
                       is_tri: bool, current_ref: list,
                       next_date: dt.datetime, ignore_pruning: bool):
        E, _ = parent.forward_variance()
        k_i = self._nearest_grid_position(E, E_trunk, alpha)

        parent.next_up = None
        parent.next_mid = None
        parent.next_down = None

        # Pruning
        if (self.pricer_parameters.pruning
                and parent.p_node < self.pricer_parameters.p_min
                and not ignore_pruning):
            if is_tri:
                parent.next_mid = self._find_or_create_child(
                    E_trunk * (alpha ** k_i), current_ref, next_date)
                parent.p_up, parent.p_mid, parent.p_down = 0.0, 1.0, 0.0
            else:
                if k_i > 0:
                    parent.next_down = self._find_or_create_child(
                        E_trunk * (alpha ** (k_i - 1)), current_ref, next_date)
                    parent.p_up, parent.p_mid, parent.p_down = 0.0, 0.0, 1.0
                else:
                    parent.next_up = self._find_or_create_child(
                        E_trunk * (alpha ** (k_i + 1)), current_ref, next_date)
                    parent.p_up, parent.p_mid, parent.p_down = 1.0, 0.0, 0.0
            return

        # Insertions croissantes : down, mid, up
        if is_tri:
            parent.next_down = self._find_or_create_child(
                E_trunk * (alpha ** (k_i - 1)), current_ref, next_date)
            parent.next_mid = self._find_or_create_child(
                E_trunk * (alpha ** k_i), current_ref, next_date)
            parent.next_up = self._find_or_create_child(
                E_trunk * (alpha ** (k_i + 1)), current_ref, next_date)
        else:
            parent.next_down = self._find_or_create_child(
                E_trunk * (alpha ** (k_i - 1)), current_ref, next_date)
            parent.next_up = self._find_or_create_child(
                E_trunk * (alpha ** (k_i + 1)), current_ref, next_date)

    def make_tree(self) -> None:
        t0_total = time.perf_counter()
        self.root = TruncNode(self.market_data.s0, self,
                              self.pricer_parameters.pricing_date, None)
        self.root.p_node = 1.0
        self.counters["nodes_created"] = 1
        self.nodes_by_date = [[self.root]]
        self.step_nodes = [1]
        self.step_candidates = [1]
        self.step_times = [0.0]
        self.step_modes = ["ROOT"]

        current_parents = [self.root]
        trunk_S = self.market_data.s0
        current_date = self.pricer_parameters.pricing_date

        for step in range(self.nb_steps):
            t0 = time.perf_counter()
            is_tri = current_parents[0].is_trinomial_step()
            mode_str = "TRI" if is_tri else "BIN"
            next_date = current_date + dt.timedelta(days=self.delta_t * 365)
            alpha = (self.alpha_tri if self.tree_mode == "trinomial_everywhere"
                     else self.alpha_bin) if is_tri else self.sqrt_alpha_bin

            E_trunk = self._forward_from(current_date, trunk_S)

            # Cas dégénéré (vol = 0)
            if abs(alpha - 1.0) < 1e-10:
                child = Node(E_trunk, self, next_date)
                self.counters["nodes_created"] += 1
                self.root.next_up = self.root.next_mid = self.root.next_down = child
                self.nodes_by_date.append([child])
                self.step_nodes.append(1)
                self.step_times.append(time.perf_counter() - t0)
                self.step_modes.append(mode_str)
                current_parents = [child]
                trunk_S = E_trunk
                current_date = next_date
                continue

            current_ref = [None]

            # 1. Identifier le tronc (le plus proche de E_trunk)
            trunk_parent = min(current_parents,
                               key=lambda n: abs(n.forward_variance()[0] - E_trunk))

            # 2. Collecter les parents DU PLUS BAS AU PLUS HAUT
            parents_below = []
            parent = trunk_parent.down_node
            while parent is not None:
                parents_below.append(parent)
                parent = parent.down_node
            parents_below.reverse()   # du plus bas au moins bas

            parents_above = []
            parent = trunk_parent.up_node
            while parent is not None:
                parents_above.append(parent)
                parent = parent.up_node
            parents_above.reverse()   # du moins haut au plus haut

            # 3. Brancher dans l'ordre croissant (bas → tronc → haut)
            for p in parents_below:
                self._branch_parent(p, E_trunk, alpha, is_tri,
                                    current_ref, next_date, ignore_pruning=False)
            self._branch_parent(trunk_parent, E_trunk, alpha, is_tri,
                                current_ref, next_date, ignore_pruning=True)
            for p in parents_above:
                self._branch_parent(p, E_trunk, alpha, is_tri,
                                    current_ref, next_date, ignore_pruning=False)

            # 4. Reconstruire la liste des nœuds du prochain niveau
            next_nodes = []
            curr = current_ref[0]
            if curr is not None:
                while curr.down_node is not None:
                    curr = curr.down_node
                while curr is not None:
                    next_nodes.append(curr)
                    curr = curr.up_node

            # 5. Probabilités
            for node in current_parents:
                node.probas()
                node.next_nodes_probas()

            self.step_candidates.append(len(next_nodes))
            self.nodes_by_date.append(next_nodes)
            self.step_nodes.append(len(next_nodes))
            self.step_times.append(time.perf_counter() - t0)
            self.step_modes.append(mode_str)

            current_parents = next_nodes
            trunk_S = E_trunk
            current_date = next_date

        self.last = self.nodes_by_date[-1][0] if self.nodes_by_date[-1] else None
        self.build_time = time.perf_counter() - t0_total

    # ==================================================================
    def backward_pricing(self) -> float:
        t0 = time.perf_counter()
        self.make_tree()
        t1 = time.perf_counter()
        for step in range(len(self.nodes_by_date) - 1, -1, -1):
            for node in self.nodes_by_date[step]:
                node.price()
        t2 = time.perf_counter()
        self.build_time = t1 - t0
        self.backward_time = t2 - t1
        self.price_time = t2 - t0
        return self.root.opt_price

    def recursive_pricing(self) -> float:
        t0 = time.perf_counter()
        try:
            self.make_tree()
            t1 = time.perf_counter()
            r = self.root.price()
            t2 = time.perf_counter()
            self.build_time = t1 - t0
            self.backward_time = t2 - t1
            self.price_time = t2 - t0
            return r
        except RecursionError:
            self.recursion_failed = True    # <-- NOUVEAU
            self.price_time = time.perf_counter() - t0
            return None

    # ==================================================================
    def european_price_by_terminal_sum(self) -> float:
        if self.option_data.is_american():
            raise ValueError("european_price_by_terminal_sum est réservé "
                             "aux options européennes.")
        if not self.nodes_by_date:
            self.make_tree()
        df_T = np.exp(-self.market_data.rate * self.T)
        terminal_nodes = self.nodes_by_date[-1]
        total = sum(node.p_node * self.option_data.payoff(node.und_price)
                    for node in terminal_nodes)
        return total * df_T

    # ==================================================================
    def iterate_columns(self):
        for nodes in self.nodes_by_date:
            if nodes:
                yield nodes[0].date, nodes

    # ==================================================================
    def plot_exercise_probability_by_date(self):
        if self.root.opt_price is None:
            self.backward_pricing()
        dates, probs = [], []
        for date, nodes in self.iterate_columns():
            probs.append(sum(n.p_node for n in nodes if n.is_exerced))
            dates.append(date)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(dates)), probs, marker='o', color='steelblue',
                linewidth=2, markersize=4)
        ax.set_xlabel("Step index"); ax.set_ylabel("Exercise probability")
        ax.set_title(f"Exercise probability by date "
                     f"({self.option_data.eur_am.capitalize()} "
                     f"{self.option_data.call_put}, K={self.option_data.K}, "
                     f"{self.nb_steps} steps)")
        ax.grid(True, alpha=0.4); ax.set_ylim(bottom=0)
        plt.tight_layout()
        return fig, dates, probs

    def plot_average_option_value_by_date(self):
        if self.root.opt_price is None:
            self.backward_pricing()
        dates, values = [], []
        for date, nodes in self.iterate_columns():
            elapsed_seconds = (date - self.pricer_parameters.pricing_date).total_seconds()
            t_years = elapsed_seconds / (365.0 * 86400.0)
            df = np.exp(-self.market_data.rate * t_years)
            values.append(sum(n.p_node * (n.opt_price or 0.0) * df for n in nodes))
            dates.append(date)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(dates)), values, marker='o', color='steelblue',
                linewidth=2, markersize=4, label='Average value (today money)')
        ax.axhline(y=values[0], color='red', linestyle='--', linewidth=1,
                   label=f'Root value = {values[0]:.4f}')
        ax.set_xlabel("Step index"); ax.set_ylabel("Option value (discounted)")
        ax.set_title(f"Average option value by date "
                     f"({self.option_data.eur_am.capitalize()} "
                     f"{self.option_data.call_put}, K={self.option_data.K}, "
                     f"{self.nb_steps} steps)")
        ax.grid(True, alpha=0.4); ax.legend()
        root_val = values[0]
        margin = max(0.01 * abs(root_val), 1e-3)
        ax.set_ylim(root_val - margin, root_val + margin)
        plt.tight_layout()
        return fig, dates, values

    def plot_tree(self, max_steps: int = 10):
        """
        Plot Plotly de l'arbre :
        - Taille des nœuds ∝ √p_node
        - Couleur graduée selon opt_price
        - Rouge vif pour les nœuds d'exercice anticipé
        - Positionnement par index dans la colonne (pas de lignes croisées)
        - Axe Y standard : prix haut en haut, prix bas en bas.
        """
        import plotly.graph_objects as go

        node_x, node_y = [], []
        node_size, node_color, node_labels = [], [], []
        node_text = []
        line_x, line_y = [], []

        pos_map = {}

        # 1) Calculer la position de chaque nœud (centré)
        # Les nœuds sont triés par prix croissant dans nodes_by_date.
        # i=0 -> prix bas, i=max -> prix haut.
        # On veut que i=max soit en haut (y positif).
        for step, nodes in enumerate(self.nodes_by_date):
            if step > max_steps:
                break
            mid = (len(nodes) - 1) / 2
            for i, node in enumerate(nodes):
                # i - mid sera négatif pour les prix bas (en bas)
                # et positif pour les prix hauts (en haut).
                pos_map[(step, id(node))] = (i - mid)

        prices_all = [n.opt_price for nodes in self.nodes_by_date[:max_steps+1] 
                      for n in nodes if n.opt_price is not None]
        p_min_color = min(prices_all) if prices_all else 0.0
        p_max_color = max(prices_all) if prices_all else 1.0
        if p_max_color == p_min_color:
            p_max_color += 1e-9

        def _thermal_color(t: float) -> str:
            t = max(0.0, min(1.0, t))
            if t < 0.25:
                r, g, b = 0, int(255 * (t / 0.25)), 255
            elif t < 0.5:
                r, g, b = 0, 255, int(255 * (1 - (t - 0.25) / 0.25))
            elif t < 0.75:
                r, g, b = int(255 * ((t - 0.5) / 0.25)), 255, 0
            else:
                r, g, b = 255, int(255 * (1 - (t - 0.75) / 0.25)), 0
            return f"rgb({r},{g},{b})"

        for step, nodes in enumerate(self.nodes_by_date):
            if step > max_steps:
                break
            
            for node in nodes:
                y = pos_map[(step, id(node))]
                display_val = node.opt_price if node.opt_price is not None else node.und_price
                
                # Skip les nœuds non atteints (pas de parent)
                if node.p_node <= 0.0 and step > 0:
                    continue
                    
                size = 8 + 30 * float(np.sqrt(max(node.p_node, 0.0)))

                if node.is_exerced:
                    color = "red"
                elif node.opt_price is not None:
                    norm = (node.opt_price - p_min_color) / (p_max_color - p_min_color)
                    color = _thermal_color(norm)
                else:
                    color = "#aaaaaa"

                node_x.append(step)
                node_y.append(y)
                node_size.append(size)
                node_color.append(color)
                node_labels.append(f"{display_val:.2f}")

                hover = [f"<b>Step</b> {step}", f"<b>S</b> = {node.und_price:.4f}"]
                if node.opt_price is not None:
                    hover.append(f"<b>Option</b> = {node.opt_price:.4f}")
                hover.append(f"<b>p_node</b> = {node.p_node:.4e}")
                if node.is_exerced:
                    hover.append("<b>⚡ Early exercise</b>")
                node_text.append("<br>".join(hover))

                if step < max_steps:
                    for child in (node.next_up, node.next_mid, node.next_down):
                        if child is not None and child.p_node > 0.0:
                            child_y = pos_map.get((step + 1, id(child)))
                            if child_y is not None:
                                line_x += [step, step + 1, None]
                                line_y += [y, child_y, None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=line_x, y=line_y, mode='lines',
            line=dict(color='rgba(180,180,180,0.6)', width=1),
            hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            marker=dict(size=node_size, color=node_color,
                        line=dict(color='rgb(50,50,50)', width=1)),
            text=node_labels, textposition='middle center',
            textfont=dict(color='white', size=9),
            hovertext=node_text, hoverinfo='text', showlegend=False))

        max_width = max(len(nodes) for nodes in self.nodes_by_date[:max_steps+1])
        fig.update_layout(
            xaxis=dict(showline=False, zeroline=False, showgrid=False,
                       showticklabels=True, title="Step",
                       range=[-1, max_steps + 2]),
            yaxis=dict(showline=False, zeroline=False, showgrid=False,
                       showticklabels=False,
                       range=[-max_width / 2 - 1, max_width / 2 + 1]),
            margin=dict(l=40, r=40, b=60, t=60), hovermode='closest',
            plot_bgcolor='rgb(248,248,248)', height=600)
        return fig

    # ==================================================================
    def export_to_excel(self, elements: List[str] = None) -> bytes:
        wb = Workbook()

        ws0 = wb.create_sheet("Metadata")
        mkt = self.market_data
        opt = self.option_data
        pr = self.pricer_parameters
        metadata = [
            ("Pricing date", pr.pricing_date.strftime("%Y-%m-%d")),
            ("Maturity date", opt.maturity.strftime("%Y-%m-%d")),
            ("T (years)", round(self.T, 6)),
            ("Underlying S0", mkt.s0),
            ("Strike K", opt.K),
            ("Risk-free rate r", mkt.rate),
            ("Volatility σ", mkt.vol),
            ("Dividend D", mkt.dividend),
            ("Ex-div date", (mkt.ex_div_date.strftime("%Y-%m-%d")
                             if mkt.dividend > 0 else "— (pas de dividende)")),
            ("Option type", opt.call_put),
            ("Exercise style", opt.eur_am),
            ("Tree mode", pr.tree_mode),
            ("Nb steps", self.nb_steps),
            ("Δt (years)", round(self.delta_t, 8)),
            ("Discount factor DF", round(self.df, 8)),
            ("√α_bin", round(self.sqrt_alpha_bin, 8)),
            ("α_tri (HW)", round(self.alpha_tri, 8)),
            ("Pruning enabled", pr.pruning),
            ("p_min", pr.p_min if pr.pruning else "—"),
            ("Nodes created", self.counters["nodes_created"]),
            ("Merges", self.counters["merges"]),
            ("Clamped probas", self.counters.get("clamped_probas", 0)),
            ("Build time (ms)", round(self.build_time * 1000, 3)),
            ("Backward time (ms)", round(self.backward_time * 1000, 3)),
            ("Total time (ms)", round(self.price_time * 1000, 3)),
        ]
        ws0.cell(row=1, column=1, value="Parameter")
        ws0.cell(row=1, column=2, value="Value")
        for i, (k, v) in enumerate(metadata, start=2):
            ws0.cell(row=i, column=1, value=k)
            ws0.cell(row=i, column=2, value=v)

        ws = wb.create_sheet("Summary")
        headers = ["step", "date", "mode", "n_nodes", "sum_p_node",
                   "sum_p_exercised", "min_und", "max_und", "weighted_opt_price"]
        for j, h in enumerate(headers, start=1):
            ws.cell(row=1, column=j, value=h)

        for step_idx, nodes in enumerate(self.nodes_by_date):
            if not nodes:
                continue
            mode = self.step_modes[step_idx] if step_idx < len(self.step_modes) else ""
            p_sum = sum(n.p_node for n in nodes)
            p_ex = sum(n.p_node for n in nodes if n.is_exerced)
            unds = [n.und_price for n in nodes]
            opt_w = sum(n.p_node * (n.opt_price or 0.0) for n in nodes)
            row = [step_idx, nodes[0].date.strftime("%Y-%m-%d"), mode,
                   len(nodes), round(p_sum, 10), round(p_ex, 10),
                   round(min(unds), 6), round(max(unds), 6), round(opt_w, 6)]
            for j, v in enumerate(row, start=1):
                ws.cell(row=step_idx + 2, column=j, value=v)

        ws2 = wb.create_sheet("Nodes")
        headers2 = ["step", "date", "und_price", "opt_price",
                    "p_node", "p_up", "p_mid", "p_down",
                    "is_trinomial_step", "is_exerced",
                    "next_up_und", "next_mid_und", "next_down_und"]
        for j, h in enumerate(headers2, start=1):
            ws2.cell(row=1, column=j, value=h)

        r = 2
        for step_idx, nodes in enumerate(self.nodes_by_date):
            is_tri_step = (step_idx < len(self.step_modes) - 1
                           and self.step_modes[step_idx + 1] == "TRI")
            for node in nodes:
                row = [
                    step_idx, node.date.strftime("%Y-%m-%d"),
                    round(node.und_price, 6),
                    round(node.opt_price, 6) if node.opt_price is not None else None,
                    round(node.p_node, 10),
                    round(node.p_up, 10), round(node.p_mid, 10), round(node.p_down, 10),
                    bool(is_tri_step), bool(node.is_exerced),
                    round(node.next_up.und_price, 6) if node.next_up else None,
                    round(node.next_mid.und_price, 6) if node.next_mid else None,
                    round(node.next_down.und_price, 6) if node.next_down else None,
                ]
                for j, v in enumerate(row, start=1):
                    ws2.cell(row=r, column=j, value=v)
                r += 1

        self._write_diamond(wb.create_sheet("Diamond_und_price"), "und_price")
        self._write_diamond(wb.create_sheet("Diamond_opt_price"), "opt_price")
        self._write_diamond(wb.create_sheet("Diamond_p_node"),    "p_node")

        if "Sheet" in wb.sheetnames:
            del wb["Sheet"]

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf.getvalue()

    def _write_diamond(self, ws, field: str) -> None:
        max_col = max((len(c) for c in self.nodes_by_date), default=1)
        center_row = max_col + 2

        for col_idx, nodes in enumerate(self.nodes_by_date):
            if not nodes:
                continue
            sorted_nodes = sorted(nodes, key=lambda n: n.und_price, reverse=True)
            n = len(sorted_nodes)
            start_row = center_row - (n - 1) // 2
            for i, node in enumerate(sorted_nodes):
                row = start_row + i
                if row < 1:
                    row = 1
                val = getattr(node, field, None)
                if isinstance(val, float):
                    val = round(val, 8)
                ws.cell(row=row, column=col_idx + 1, value=val)

    # ==================================================================
    def _has_div_step(self) -> bool:
        return any(m == "TRI" for m in self.step_modes)

    def _overall_mode(self) -> str:
        if self.tree_mode == "trinomial_everywhere":
            return "TRINOMIAL everywhere"
        if self._has_div_step():
            return "MIXED (binomial + trinomial au pas ex-div)"
        return "BINOMIAL (pas de dividende)"

    def debug_report(self) -> str:
        lines = ["=== Debug report ==="]
        lines.append(f"  mode           = {self._overall_mode()}")
        lines.append(f"  tree_mode param= {self.tree_mode}")
        lines.append(f"  steps          = {self.nb_steps}")
        lines.append(f"  α_bin          = {self.alpha_bin:.8f}")
        lines.append(f"  √α_bin         = {self.sqrt_alpha_bin:.8f}")
        lines.append(f"  α_tri (HW)     = {self.alpha_tri:.8f}")
        lines.append(f"  dividend       = {self.market_data.dividend}")
        if self.market_data.dividend > 0:
            lines.append(f"  ex-div date    = {self.market_data.ex_div_date:%Y-%m-%d}")
        lines.append(f"  pruning        = {self.pricer_parameters.pruning}"
                     + (f" (p_min={self.pricer_parameters.p_min:.2e})"
                        if self.pricer_parameters.pruning else ""))
        lines.append(f"  nodes_created  = {self.counters['nodes_created']}")
        lines.append(f"  merges         = {self.counters['merges']}")
        lines.append(f"  clamped_probas = {self.counters.get('clamped_probas', 0)}")
        lines.append(f"  Taille max col = {max(self.step_nodes)}")
        lines.append(f"  Taille finale  = {self.step_nodes[-1]}")
        n_tri = sum(1 for m in self.step_modes if m == "TRI")
        lines.append(f"  Nb steps TRI   = {n_tri}  / {len(self.step_modes) - 1}")

        ratios = [self.step_nodes[i+1] / max(self.step_nodes[i], 1)
                  for i in range(len(self.step_nodes) - 1)]
        growth = max(ratios) if ratios else 0.0
        bin_ratios = [ratios[i] for i, m in enumerate(self.step_modes[1:]) if m == "BIN"]
        bin_growth = max(bin_ratios) if bin_ratios else 0.0
        verdict = "OK" if bin_growth < 2.5 else "⚠️ EXPLOSION"
        lines.append(f"  Growth ratio (BIN max) = {bin_growth:.4f}  → {verdict}")
        lines.append(f"  Growth ratio (global)  = {growth:.4f}")
        lines.append(f"  build_time     = {self.build_time*1000:.3f} ms")
        lines.append(f"  backward_time  = {self.backward_time*1000:.3f} ms")
        lines.append(f"  total          = {self.price_time*1000:.3f} ms")
        if self.recursion_failed:
            lines.append(f"  recursion      = ⚠️ FAILED (RecursionError)")
        return "\n".join(lines)

    def debug_steps_dataframe(self):
        import pandas as pd
        lengths = [len(self.step_modes), len(self.step_nodes),
                   len(self.step_candidates), len(self.step_times)]
        n = min(lengths)
        modes = list(self.step_modes[:n])
        nodes = list(self.step_nodes[:n])
        cands = list(self.step_candidates[:n])
        times = list(self.step_times[:n])
        parents = [None] + nodes[:-1]
        return pd.DataFrame({
            "step": list(range(n)),
            "mode": modes,
            "parents": parents,
            "candidates": cands,
            "nodes": nodes,
            "time (ms)": [float(t) * 1000.0 for t in times],
        })

    # ==================================================================
    def sanity_checks(self) -> list:
        from black_scholes import Black_Scholes
        from option import Option as _Option
        from market import Market as _Market

        checks = []
        opt = self.option_data
        mkt = self.market_data
        pr = self.pricer_parameters
        N = max(self.nb_steps, 1)

        def _tol_invariant(scale: float, floor: float = 1e-4) -> float:
            return max(floor, scale * 1e-4, scale / N * 0.1)

        def _tol_convergence(scale: float, floor: float = 1e-3) -> float:
            return max(floor, scale * 2.0 / N)

        def price_variant(call_put=None, eur_am=None, div=None):
            new_opt = _Option(opt.K, opt.maturity,
                              call_put or opt.call_put, eur_am or opt.eur_am)
            new_mkt = _Market(mkt.s0, mkt.rate, mkt.vol,
                              div if div is not None else mkt.dividend,
                              mkt.ex_div_date)
            t = Tree(pr, new_mkt, new_opt)
            return t.backward_pricing()

        base_price = self.root.opt_price

        # --- Check 1 : Euro sans div ≈ BS ---
        if (not opt.is_american()) and mkt.dividend == 0 and mkt.vol > 1e-8:
            try:
                bs = Black_Scholes(pr, mkt, opt).pricing_BS()
                diff = abs(base_price - bs)
                tol = _tol_convergence(abs(bs))
                if self.nb_steps < 30:
                    verdict = "N/A"
                    detail = f"too few steps ({self.nb_steps}) to judge"
                else:
                    verdict = "PASS" if diff < tol else "FAIL"
                    detail = ""  # <-- CORRECTION ICI
                checks.append({
                    "name": "European without div ≈ Black-Scholes",
                    "value": f"tree={base_price:.6f}  BS={bs:.6f}  diff={diff:.2e}  tolerance={tol:.2e}",
                    "verdict": verdict, "detail": detail})
            except Exception as e:
                checks.append({"name": "European without div ≈ BS",
                               "value": str(e), "verdict": "N/A", "detail": ""})
        else:
            checks.append({"name": "European without div ≈ BS",
                           "value": "only for european options without dividend",
                           "verdict": "N/A", "detail": ""})

        # --- Check 2 : Put-call parity ---
        if (not opt.is_american()) and mkt.dividend == 0 and mkt.vol > 1e-8:
            try:
                c = price_variant(call_put="call")
                p = price_variant(call_put="put")
                lhs = c - p
                rhs = mkt.s0 - opt.K * np.exp(-mkt.rate * self.T)
                diff = abs(lhs - rhs)
                scale = max(abs(lhs), abs(rhs), 1.0)
                tol = _tol_invariant(scale)
                verdict = "PASS" if diff < tol else "FAIL"
                checks.append({
                    "name": "Put-call parity : C − P = S₀ − K·e^(−rT)",
                    "value": f"C−P={lhs:.6f}  S−Ke^(−rT)={rhs:.6f}  diff={diff:.2e}  tolerance={tol:.2e}",
                    "verdict": verdict})
            except Exception as e:
                checks.append({"name": "Put-call parity",
                               "value": str(e), "verdict": "N/A", "detail": ""})
        else:
            checks.append({"name": "Put-call parity",
                           "value": "only for european options without dividend",
                           "verdict": "N/A", "detail": ""})

        # --- Check 3 : A_call = E_call (sans div, r>0) ---
        if mkt.dividend == 0 and mkt.rate > 0 and opt.call_put == "call":
            try:
                a = price_variant(eur_am="american")
                e = price_variant(eur_am="european")
                diff = a - e
                tol = _tol_invariant(max(abs(e), 1.0))
                verdict = "PASS" if abs(diff) < tol else "FAIL"
                checks.append({
                    "name": "A_call = E_call (without div, r>0)",
                    "value": f"A={a:.6f}  E={e:.6f}  diff={diff:.2e}  tolerance={tol:.2e}",
                    "verdict": verdict})
            except Exception as ex:
                checks.append({"name": "A_call = E_call", "value": str(ex),
                               "verdict": "N/A", "detail": ""})
        else:
            checks.append({"name": "A_call = E_call (without div, r>0)",
                           "value": "only for call without dividend and with r>0",
                           "verdict": "N/A", "detail": ""})

        # --- Check 4 : A_put ≥ E_put ---
        if opt.call_put == "put":
            try:
                a = price_variant(eur_am="american")
                e = price_variant(eur_am="european")
                diff = a - e
                verdict = "PASS" if diff >= -1e-8 else "FAIL"
                checks.append({
                    "name": "A_put ≥ E_put",
                    "value": f"A={a:.6f}  E={e:.6f}  prime={diff:.2e}",
                    "verdict": verdict,
                    "detail": "early exercise premium ≥ 0"})
            except Exception as ex:
                checks.append({"name": "A_put ≥ E_put", "value": str(ex),
                               "verdict": "N/A", "detail": ""})
        else:
            checks.append({"name": "A_put ≥ E_put",
                           "value": "only for put",
                           "verdict": "N/A", "detail": ""})

        # --- Check 5 : A_call ≥ E_call (div > 0) ---
        if mkt.dividend > 0 and opt.call_put == "call":
            try:
                a = price_variant(eur_am="american")
                e = price_variant(eur_am="european")
                diff = a - e
                verdict = "PASS" if diff >= -1e-8 else "FAIL"
                detail = "exercise just before ex-div"
                if abs(diff) < 1e-8:
                    detail += " — premium = 0 (call OTM everywhere before ex-div ?)"
                checks.append({
                    "name": "A_call ≥ E_call (div > 0)",
                    "value": f"A={a:.6f}  E={e:.6f}  prime={diff:.2e}",
                    "verdict": verdict, "detail": detail})
            except Exception as ex:
                checks.append({"name": "A_call ≥ E_call (div>0)",
                               "value": str(ex), "verdict": "N/A", "detail": ""})
        else:
            checks.append({"name": "A_call ≥ E_call (div>0)",
                           "value": "only for call with dividend",
                           "verdict": "N/A", "detail": ""})

        return checks
    # ==================================================================
    def gap(self, given_gap: float = None) -> Tuple[float, float, float]:
        T = self.T
        sd = self.market_data.s0 * np.exp(self.market_data.rate * T) \
             * np.sqrt(np.exp(self.market_data.vol ** 2 * T) - 1)
        gap = ((3 * self.market_data.s0) / (8 * np.sqrt(2 * np.pi))) \
              * (np.exp(self.market_data.vol ** 2 * self.delta_t) - 1) \
              * np.exp(2 * self.market_data.rate * self.delta_t) \
              / np.sqrt(np.exp(self.market_data.vol ** 2 * T) - 1)
        calc = None
        if given_gap is not None and given_gap > 0:
            up = self.market_data.vol ** 2 * T
            down = np.log(1 + (8 * np.sqrt(2 * np.pi) * given_gap
                               * np.sqrt(np.exp(self.market_data.vol ** 2 * T) - 1))
                          / (3 * self.market_data.s0))
            if abs(down) > 1e-15:
                calc = round(up / down)
        return sd, gap, calc
