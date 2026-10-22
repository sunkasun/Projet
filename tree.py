"""Arbre binomial/trinomial avec switch sur le pas ex-div."""

from __future__ import annotations

import datetime as dt
import io
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet
import plotly.graph_objects as go
import pandas as pd

from node import Node, TruncNode
from parameters import PricerParameters
from market import Market
from option import Option


# ---------------------------------------------------------------------------
# Fonctions utilitaires du module
# ---------------------------------------------------------------------------
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


def _tol_invariant(scale: float, nb_steps: int,
                   floor: float = 1e-4) -> float:
    """Tolérance pour les invariants (parité put-call, etc.)."""
    return max(floor, scale * 1e-4, scale / nb_steps * 0.1)


def _tol_convergence(scale: float, nb_steps: int,
                     floor: float = 1e-3) -> float:
    """Tolérance pour la convergence vers Black-Scholes."""
    return max(floor, scale * 2.0 / nb_steps)


class Tree:
    """Arbre binomial/trinomial avec switch sur le pas ex-div."""

    def __init__(self, pricer_parameters: PricerParameters,
                 market_data: Market, option_data: Option,
                 debug: bool = False) -> None:
        """Initialise les paramètres de l'arbre."""
        # --- Objets métier ---
        self.pricer_parameters: PricerParameters = pricer_parameters
        self.market_data: Market = market_data
        self.option_data: Option = option_data
        self.debug: bool = debug
        # --- Initialisation en sous-étapes ---
        self._compute_tree_parameters()
        self._init_containers()

    # -------------------------------------------------------------------
    # Initialisation : paramètres dérivés
    # -------------------------------------------------------------------
    def _compute_tree_parameters(self) -> None:
        """Calcule Δt, DF puis les facteurs α."""
        self.nb_steps: int = int(self.pricer_parameters.nb_steps)
        self.T: float = ((self.option_data.maturity
                          - self.pricer_parameters.pricing_date).days / 365)
        if self.nb_steps > 0:
            self.delta_t: float = self.T / self.nb_steps
        else:
            self.delta_t: float = 0.0
        self.df: float = np.exp(-self.market_data.rate * self.delta_t)
        self._compute_alpha_factors()

    def _compute_alpha_factors(self) -> None:
        """Calcule les facteurs de saut binomiaux et trinomiaux."""
        x: float = np.exp(self.market_data.vol ** 2 * self.delta_t) - 1.0
        self.sqrt_alpha_bin: float = (
            1.0 + x / 2.0 + np.sqrt(max(x * (1.0 + x / 4.0), 0.0)))
        self.alpha_bin: float = self.sqrt_alpha_bin ** 2
        self.alpha_tri: float = np.exp(
            self.market_data.vol * np.sqrt(3.0 * self.delta_t))
        self.tree_mode: str = self.pricer_parameters.tree_mode
        if self.tree_mode == "trinomial_everywhere":
            self.alpha: float = self.alpha_tri
        else:
            self.alpha = self.alpha_bin

    def _init_containers(self) -> None:
        """Initialise les structures vides et compteurs."""
        # --- Structure de l'arbre ---
        self.nodes_by_date: List[List[Node]] = []
        self.root: Optional[Node] = None
        self.last: Optional[Node] = None
        # --- Compteurs et timings ---
        self.counters: Dict[str, int] = {
            "nodes_created": 0, "merges": 0, "clamped_probas": 0}
        self.step_times: List[float] = []
        self.step_nodes: List[int] = []
        self.step_candidates: List[int] = []
        self.step_modes: List[str] = []
        self.build_time: float = 0.0
        self.backward_time: float = 0.0
        self.price_time: float = 0.0
        self.recursion_failed: bool = False

    # -------------------------------------------------------------------
    # Forward et construction
    # -------------------------------------------------------------------
    def _forward_from(self, date: dt.datetime, S_start: float) -> float:
        """Forward du tronc depuis S_start sur un pas Δt (avec dividende)."""
        r: float = self.market_data.rate
        D: float = self.market_data.dividend
        ex_div: dt.datetime = self.market_data.ex_div_date
        dt_: float = self.delta_t
        E_pur: float = S_start * np.exp(r * dt_)
        # --- Détection de la date ex-div dans le pas ---
        step_end: dt.datetime = date + dt.timedelta(days=dt_ * 365)
        if D > 0 and (date < ex_div <= step_end):
            tau_days: float = (ex_div - date).total_seconds() / 86400
            tau_ratio: float = tau_days / (dt_ * 365) if dt_ > 1e-15 else 1.0
            tau_ratio = max(0.0, min(1.0, tau_ratio))
            return E_pur - D * np.exp(r * dt_ * (1.0 - tau_ratio))
        return E_pur

    def make_tree(self) -> None:
        """Construit l'arbre pas par pas."""
        t0_total: float = time.perf_counter()
        self._init_tree_state()
        state: Dict[str, Any] = {
            "current": [self.root],
            "trunk_S": self.market_data.s0,
            "date": self.pricer_parameters.pricing_date,
        }
        # --- Boucle principale sur les pas ---
        for step in range(self.nb_steps):
            self._process_step(step, state)
        self._finalize_tree(t0_total)

    def _init_tree_state(self) -> None:
        """Initialise root et les listes de stats."""
        self.root = TruncNode(self.market_data.s0, self,
                              self.pricer_parameters.pricing_date, None)
        self.root.p_node = 1.0
        self.counters["nodes_created"] += 1
        self.nodes_by_date = [[self.root]]
        self.step_nodes = [1]
        self.step_candidates = [1]
        self.step_times = [0.0]
        self.step_modes = ["ROOT"]

    def _finalize_tree(self, t0_total: float) -> None:
        """Finalise last et build_time."""
        if self.nodes_by_date[-1]:
            self.last = self.nodes_by_date[-1][0]
        else:
            self.last = None
        self.build_time = time.perf_counter() - t0_total

    def _process_step(self, step: int, state: Dict[str, Any]) -> None:
        """Traite un pas complet en deux sous-étapes."""
        ctx: Dict[str, Any] = self._prepare_step_context(step, state)
        self._execute_step(ctx, state)

    def _prepare_step_context(self, step: int,
                              state: Dict[str, Any]) -> Dict[str, Any]:
        """Prépare toutes les valeurs du pas courant."""
        t0: float = time.perf_counter()
        is_tri: bool = state["current"][0].is_trinomial_step()
        mode_str: str = "TRI" if is_tri else "BIN"
        next_date: dt.datetime = state["date"] + dt.timedelta(
            days=self.delta_t * 365)
        a: float = self._step_alpha(is_tri)
        E_trunk: float = self._forward_from(state["date"], state["trunk_S"])
        log_a: float = np.log(a) if abs(a - 1.0) > 1e-10 else 0.0
        parent_k: List[int] = self._compute_parent_k(
            state["current"], E_trunk, log_a)
        # --- p_min selon le mode pruning ---
        if self.pricer_parameters.pruning:
            p_min: float = self.pricer_parameters.p_min
        else:
            p_min = 0.0
        return {"t0": t0, "is_tri": is_tri, "mode_str": mode_str,
                "next_date": next_date, "a": a, "E_trunk": E_trunk,
                "log_a": log_a, "parent_k": parent_k, "p_min": p_min,
                "step": step}

    def _execute_step(self, ctx: Dict[str, Any],
                      state: Dict[str, Any]) -> None:
        """Crée les nœuds, lie les parents et propage les probas."""
        reached_k = self._compute_reached_k(
            state["current"], ctx["parent_k"], ctx["is_tri"],
            ctx["p_min"], ctx["step"])
        self.step_candidates.append(len(reached_k))
        sorted_k = sorted(reached_k)
        k_to_node, next_nodes = self._create_next_nodes(
            ctx["E_trunk"], ctx["a"], sorted_k, ctx["next_date"])
        self._link_parents(state["current"], ctx["parent_k"], k_to_node,
                           ctx["is_tri"], ctx["p_min"])
        self._link_siblings(next_nodes)
        self._propagate_probas(state["current"], ctx["p_min"])
        self._append_step(next_nodes, ctx["t0"], ctx["mode_str"])
        # --- Mise à jour du state pour le pas suivant ---
        state["current"] = next_nodes
        state["trunk_S"] = ctx["E_trunk"]
        state["date"] = ctx["next_date"]

    def _step_alpha(self, is_tri: bool) -> float:
        """Facteur de saut a selon le mode du pas."""
        if is_tri:
            if self.tree_mode == "trinomial_everywhere":
                return self.alpha_tri
            return self.alpha_bin
        return self.sqrt_alpha_bin

    # -------------------------------------------------------------------
    # Calcul des indices k
    # -------------------------------------------------------------------
    def _compute_parent_k(self, current: List[Node], E_trunk: float,
                          log_a: float) -> List[int]:
        """Indice k de chaque parent (position relative au tronc)."""
        parent_k: List[int] = []
        for node in current:
            E_i, _ = node.forward_variance()
            if E_i <= 0:
                E_i = 1e-12
            # --- Calcul du logarithme (évite division par 0) ---
            if abs(log_a) < 1e-12:
                k_i: int = 0
            else:
                k_i = int(round(np.log(E_i / E_trunk) / log_a))
            parent_k.append(k_i)
        return parent_k

    def _compute_reached_k(self, current: List[Node], parent_k: List[int],
                           is_tri: bool, p_min: float, step: int) -> set:
        """Indices k atteignables à l'étape suivante."""
        reached_k: set = set()
        for idx, node in enumerate(current):
            k_i: int = parent_k[idx]
            if node.p_node < p_min:
                reached_k.add(k_i)
            elif is_tri:
                reached_k.update([k_i - 1, k_i, k_i + 1])
            else:
                reached_k.update([k_i - 1, k_i + 1])
        # --- Filtrage par écart-type si pruning actif ---
        if self.pricer_parameters.pruning and reached_k:
            reached_k = self._filter_by_std(reached_k, is_tri, step)
        return reached_k

    def _filter_by_std(self, reached_k: set, is_tri: bool,
                       step: int) -> set:
        """Limite le nombre de k atteints via un critère d'écarts-types."""
        if is_tri:
            half_w: int = int(6 * np.sqrt(step + 1) / np.sqrt(3))
        else:
            half_w = int(6 * np.sqrt(step + 1))
        half_w = max(half_w, 5)
        return {k for k in reached_k if abs(k) <= half_w}

    # -------------------------------------------------------------------
    # Création et liaison des nœuds
    # -------------------------------------------------------------------
    def _create_next_nodes(self, E_trunk: float, a: float,
                           sorted_k: List[int],
                           next_date: dt.datetime
                           ) -> Tuple[Dict[int, Node], List[Node]]:
        """Crée les nœuds de l'étape suivante, indexés par k."""
        k_to_node: Dict[int, Node] = {}
        next_nodes: List[Node] = []
        for k in sorted_k:
            price: float = E_trunk * (a ** k)
            node = Node(price, self, next_date)
            k_to_node[k] = node
            next_nodes.append(node)
            self.counters["nodes_created"] += 1
        return k_to_node, next_nodes

    def _link_parents(self, current: List[Node], parent_k: List[int],
                      k_to_node: Dict[int, Node], is_tri: bool,
                      p_min: float) -> None:
        """Lie chaque parent à ses enfants et fixe ses probas."""
        for idx, parent in enumerate(current):
            k_i: int = parent_k[idx]
            if parent.p_node < p_min:
                self._link_pruned_parent(parent, k_i, k_to_node, is_tri)
            else:
                self._link_active_parent(parent, k_i, k_to_node, is_tri)

    def _link_pruned_parent(self, parent: Node, k_i: int,
                            k_to_node: Dict[int, Node],
                            is_tri: bool) -> None:
        """Lie un parent dont p_node < p_min (mode pruning)."""
        if is_tri:
            parent.next_mid = k_to_node.get(k_i)
            parent.next_up = parent.next_down = None
            parent.p_up = parent.p_down = 0.0
            parent.p_mid = 1.0
        elif k_i > 0:
            parent.next_down = k_to_node.get(k_i - 1)
            parent.next_up = parent.next_mid = None
            parent.p_up = parent.p_mid = 0.0
            parent.p_down = 1.0
        else:
            parent.next_up = k_to_node.get(k_i + 1)
            parent.next_mid = parent.next_down = None
            parent.p_mid = parent.p_down = 0.0
            parent.p_up = 1.0

    def _link_active_parent(self, parent: Node, k_i: int,
                            k_to_node: Dict[int, Node],
                            is_tri: bool) -> None:
        """Lie un parent actif (p_node ≥ p_min) à ses enfants."""
        if is_tri:
            parent.next_mid = k_to_node.get(k_i)
            parent.next_up = k_to_node.get(k_i + 1)
            parent.next_down = k_to_node.get(k_i - 1)
        else:
            parent.next_mid = None
            parent.next_up = k_to_node.get(k_i + 1)
            parent.next_down = k_to_node.get(k_i - 1)

    def _link_siblings(self, next_nodes: List[Node]) -> None:
        """Lie chaque nœud à son frère du dessus/dessous."""
        for i in range(1, len(next_nodes)):
            next_nodes[i].down_node = next_nodes[i - 1]
            next_nodes[i - 1].up_node = next_nodes[i]

    def _propagate_probas(self, current: List[Node], p_min: float) -> None:
        """Calcule les probas et propage p_node aux enfants."""
        for node in current:
            if node.p_node >= p_min:
                node.probas()
            node.next_nodes_probas()

    def _append_step(self, next_nodes: List[Node], t0: float,
                     mode_str: str) -> None:
        """Enregistre les stats du pas courant."""
        self.nodes_by_date.append(next_nodes)
        self.step_nodes.append(len(next_nodes))
        self.step_times.append(time.perf_counter() - t0)
        self.step_modes.append(mode_str)

    # -------------------------------------------------------------------
    # Pricing
    # -------------------------------------------------------------------
    def backward_pricing(self) -> float:
        """Pricing itératif (bottom-up puis backward)."""
        t0: float = time.perf_counter()
        self.make_tree()
        t1: float = time.perf_counter()
        # --- Parcours backward de toutes les colonnes ---
        for step in range(len(self.nodes_by_date) - 1, -1, -1):
            for node in self.nodes_by_date[step]:
                node.price()
        t2: float = time.perf_counter()
        self.build_time = t1 - t0
        self.backward_time = t2 - t1
        self.price_time = t2 - t0
        return self.root.opt_price

    def recursive_pricing(self) -> Optional[float]:
        """Pricing récursif (RecursionError au-delà de ~900 pas)."""
        t0: float = time.perf_counter()
        try:
            self.make_tree()
            t1: float = time.perf_counter()
            result: float = self.root.price()
            t2: float = time.perf_counter()
            self.build_time = t1 - t0
            self.backward_time = t2 - t1
            self.price_time = t2 - t0
            return result
        except RecursionError:
            self.recursion_failed = True
            self.price_time = time.perf_counter() - t0
            return None

    def european_price_by_terminal_sum(self) -> float:
        """Prix européen par somme terminale (contrôle backward)."""
        if self.option_data.is_american():
            raise ValueError("european_price_by_terminal_sum est réservé "
                             "aux options européennes.")
        if not self.nodes_by_date:
            self.make_tree()
        df_T: float = np.exp(-self.market_data.rate * self.T)
        total: float = sum(
            node.p_node * self.option_data.payoff(node.und_price)
            for node in self.nodes_by_date[-1])
        return total * df_T

    # -------------------------------------------------------------------
    # Itération et plots simples
    # -------------------------------------------------------------------
    def iterate_columns(self) -> Iterator[Tuple[dt.datetime, List[Node]]]:
        """Générateur (date, nodes) par colonne."""
        for nodes in self.nodes_by_date:
            if nodes:
                yield nodes[0].date, nodes

    def plot_exercise_probability_by_date(
            self) -> Tuple[plt.Figure, List[dt.datetime], List[float]]:
        """Probabilité d'exercice anticipé par date."""
        if self.root.opt_price is None:
            self.backward_pricing()
        dates: List[dt.datetime] = []
        probs: List[float] = []
        for date, nodes in self.iterate_columns():
            probs.append(sum(n.p_node for n in nodes if n.is_exerced))
            dates.append(date)
        title = (f"Exercise probability by date "
                 f"({self.option_data.eur_am.capitalize()} "
                 f"{self.option_data.call_put}, K={self.option_data.K}, "
                 f"{self.nb_steps} steps)")
        fig, _ = self._plot_series(probs, dates, 'steelblue',
                                    "Exercise probability", title)
        return fig, dates, probs

    def _plot_series(self, values: List[float], dates: List[dt.datetime],
                     color: str, ylabel: str,
                     title: str) -> Tuple[plt.Figure, Any]:
        """Graphe générique ligne (séries temporelles)."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(dates)), values, marker='o', color=color,
                linewidth=2, markersize=4)
        ax.set_xlabel("Step index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.4)
        return fig, ax

    def plot_average_option_value_by_date(
            self) -> Tuple[plt.Figure, List[dt.datetime], List[float]]:
        """Valeur moyenne de l'option par date (actualisée)."""
        if self.root.opt_price is None:
            self.backward_pricing()
        dates: List[dt.datetime] = []
        values: List[float] = []
        # --- Boucle sur les colonnes avec actualisation ---
        for date, nodes in self.iterate_columns():
            elapsed_s: float = (
                date - self.pricer_parameters.pricing_date).total_seconds()
            t_years: float = elapsed_s / (365.0 * 86400.0)
            df: float = np.exp(-self.market_data.rate * t_years)
            values.append(sum(n.p_node * (n.opt_price or 0.0) * df
                              for n in nodes))
            dates.append(date)
        fig = self._plot_avg_value(dates, values)
        return fig, dates, values

    def _plot_avg_value(self, dates: List[dt.datetime],
                        values: List[float]) -> plt.Figure:
        """Trace la valeur moyenne par date."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(dates)), values, marker='o', color='steelblue',
                linewidth=2, markersize=4, label='Average value')
        ax.axhline(y=values[0], color='red', linestyle='--', linewidth=1,
                   label=f'Root value = {values[0]:.4f}')
        ax.set_xlabel("Step index")
        ax.set_ylabel("Option value (discounted)")
        ax.set_title(f"Average option value by date "
                     f"({self.option_data.eur_am.capitalize()} "
                     f"{self.option_data.call_put}, K={self.option_data.K}, "
                     f"{self.nb_steps} steps)")
        ax.grid(True, alpha=0.4)
        ax.legend()
        root_val: float = values[0]
        margin: float = max(0.01 * abs(root_val), 1e-3)
        ax.set_ylim(root_val - margin, root_val + margin)
        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------
    # Plot Plotly de l'arbre
    # -------------------------------------------------------------------
    def plot_tree(self, max_steps: int = 10) -> go.Figure:
        """Plot Plotly de l'arbre (nodes + edges)."""
        node_xy: Dict[Tuple[int, int], float] = \
            self._compute_node_positions(max_steps)
        p_min_c, p_max_c = self._compute_color_range(max_steps)
        traces: Dict[str, list] = self._build_tree_traces(
            max_steps, node_xy, p_min_c, p_max_c)
        return self._make_plotly_figure(traces, max_steps)

    def _compute_node_positions(self, max_steps: int
                                ) -> Dict[Tuple[int, int], float]:
        """Dict (step, id(node)) -> y centré."""
        pos_map: Dict[Tuple[int, int], float] = {}
        for step, nodes in enumerate(self.nodes_by_date):
            if step > max_steps:
                break
            mid: float = (len(nodes) - 1) / 2
            for i, node in enumerate(nodes):
                pos_map[(step, id(node))] = i - mid
        return pos_map

    def _compute_color_range(self, max_steps: int) -> Tuple[float, float]:
        """(min, max) des prix d'option pour le gradient."""
        prices_all: List[float] = [
            n.opt_price for nodes in self.nodes_by_date[:max_steps + 1]
            for n in nodes if n.opt_price is not None]
        p_min_c: float = min(prices_all) if prices_all else 0.0
        p_max_c: float = max(prices_all) if prices_all else 1.0
        if p_max_c == p_min_c:
            p_max_c += 1e-9
        return p_min_c, p_max_c

    def _build_tree_traces(self, max_steps: int,
                           node_xy: Dict[Tuple[int, int], float],
                           p_min_c: float,
                           p_max_c: float) -> Dict[str, list]:
        """Construit les listes pour les traces Plotly."""
        traces: Dict[str, list] = {
            "line_x": [], "line_y": [], "node_x": [], "node_y": [],
            "size": [], "color": [], "labels": [], "text": []}
        for step, nodes in enumerate(self.nodes_by_date):
            if step > max_steps:
                break
            for node in nodes:
                if node.p_node <= 0.0 and step > 0:
                    continue
                self._append_node_trace(traces, step, node, node_xy,
                                        p_min_c, p_max_c)
                if step < max_steps:
                    self._append_edges(traces, step, node, node_xy)
        return traces

    def _append_node_trace(self, traces: Dict[str, list], step: int,
                           node: Node,
                           node_xy: Dict[Tuple[int, int], float],
                           p_min_c: float, p_max_c: float) -> None:
        """Ajoute un nœud à la trace courante."""
        y: float = node_xy[(step, id(node))]
        if node.opt_price is not None:
            display_val: float = node.opt_price
        else:
            display_val = node.und_price
        traces["node_x"].append(step)
        traces["node_y"].append(y)
        traces["size"].append(
            8 + 30 * float(np.sqrt(max(node.p_node, 0.0))))
        traces["color"].append(self._node_color(node, p_min_c, p_max_c))
        traces["labels"].append(f"{display_val:.2f}")
        traces["text"].append(self._node_hover(step, node, display_val))

    def _append_edges(self, traces: Dict[str, list], step: int, node: Node,
                      node_xy: Dict[Tuple[int, int], float]) -> None:
        """Ajoute les arêtes sortantes d'un nœud."""
        y: float = node_xy[(step, id(node))]
        for child in (node.next_up, node.next_mid, node.next_down):
            if child is None or child.p_node <= 0.0:
                continue
            child_y = node_xy.get((step + 1, id(child)))
            if child_y is None:
                continue
            traces["line_x"] += [step, step + 1, None]
            traces["line_y"] += [y, child_y, None]

    def _node_color(self, node: Node, p_min_c: float,
                    p_max_c: float) -> str:
        """Couleur d'un nœud (rouge si exercé, gradient sinon)."""
        if node.is_exerced:
            return "red"
        if node.opt_price is None:
            return "#aaaaaa"
        norm: float = (node.opt_price - p_min_c) / (p_max_c - p_min_c)
        return _thermal_color(norm)

    def _node_hover(self, step: int, node: Node, display_val: float) -> str:
        """Texte hover d'un nœud."""
        hover: List[str] = [f"<b>Step</b> {step}",
                             f"<b>S</b> = {node.und_price:.4f}"]
        if node.opt_price is not None:
            hover.append(f"<b>Option</b> = {node.opt_price:.4f}")
        hover.append(f"<b>p_node</b> = {node.p_node:.4e}")
        if node.is_exerced:
            hover.append("<b>⚡ Early exercise</b>")
        return "<br>".join(hover)

    def _make_plotly_figure(self, traces: Dict[str, list],
                            max_steps: int) -> go.Figure:
        """Assemble la figure Plotly finale."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=traces["line_x"], y=traces["line_y"], mode='lines',
            line=dict(color='rgba(180,180,180,0.6)', width=1),
            hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(
            x=traces["node_x"], y=traces["node_y"], mode='markers+text',
            marker=dict(size=traces["size"], color=traces["color"],
                        line=dict(color='rgb(50,50,50)', width=1)),
            text=traces["labels"], textposition='middle center',
            textfont=dict(color='white', size=9),
            hovertext=traces["text"], hoverinfo='text', showlegend=False))
        max_width: int = max(len(nodes)
                             for nodes in self.nodes_by_date[:max_steps + 1])
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

    # -------------------------------------------------------------------
    # Export Excel
    # -------------------------------------------------------------------
    def export_to_excel(self,
                        elements: Optional[List[str]] = None) -> bytes:
        """Exporte l'arbre au format Excel (multi-feuilles)."""
        wb = Workbook()
        self._write_metadata_sheet(wb.create_sheet("Metadata"))
        self._write_summary_sheet(wb.create_sheet("Summary"))
        self._write_nodes_sheet(wb.create_sheet("Nodes"))
        self._write_diamond(wb.create_sheet("Diamond_und_price"), "und_price")
        self._write_diamond(wb.create_sheet("Diamond_opt_price"), "opt_price")
        self._write_diamond(wb.create_sheet("Diamond_p_node"), "p_node")
        if "Sheet" in wb.sheetnames:
            del wb["Sheet"]
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf.getvalue()

    def _build_metadata_market(self) -> List[Tuple[str, Any]]:
        """Section 'market/option' des métadonnées du run."""
        mkt = self.market_data
        opt = self.option_data
        pr = self.pricer_parameters
        if mkt.dividend > 0:
            ex_div_str: str = mkt.ex_div_date.strftime("%Y-%m-%d")
        else:
            ex_div_str = "— (pas de dividende)"
        return [
            ("Pricing date", pr.pricing_date.strftime("%Y-%m-%d")),
            ("Maturity date", opt.maturity.strftime("%Y-%m-%d")),
            ("T (years)", round(self.T, 6)),
            ("Underlying S0", mkt.s0), ("Strike K", opt.K),
            ("Risk-free rate r", mkt.rate), ("Volatility σ", mkt.vol),
            ("Dividend D", mkt.dividend), ("Ex-div date", ex_div_str),
            ("Option type", opt.call_put), ("Exercise style", opt.eur_am),
        ]

    def _build_metadata_tree(self) -> List[Tuple[str, Any]]:
        """Section 'tree' des métadonnées du run."""
        pr = self.pricer_parameters
        return [
            ("Tree mode", pr.tree_mode), ("Nb steps", self.nb_steps),
            ("Δt (years)", round(self.delta_t, 8)),
            ("Discount factor DF", round(self.df, 8)),
            ("√α_bin", round(self.sqrt_alpha_bin, 8)),
            ("α_tri (HW)", round(self.alpha_tri, 8)),
            ("Pruning enabled", pr.pruning),
            ("p_min", pr.p_min if pr.pruning else "—"),
        ]

    def _build_metadata_counters(self) -> List[Tuple[str, Any]]:
        """Section 'compteurs/timings' des métadonnées du run."""
        return [
            ("Nodes created", self.counters["nodes_created"]),
            ("Merges", self.counters["merges"]),
            ("Clamped probas", self.counters.get("clamped_probas", 0)),
            ("Build time (ms)", round(self.build_time * 1000, 3)),
            ("Backward time (ms)", round(self.backward_time * 1000, 3)),
            ("Total time (ms)", round(self.price_time * 1000, 3)),
        ]

    def _build_metadata_list(self) -> List[Tuple[str, Any]]:
        """Concatène les trois sections de métadonnées."""
        metadata: List[Tuple[str, Any]] = []
        metadata.extend(self._build_metadata_market())
        metadata.extend(self._build_metadata_tree())
        metadata.extend(self._build_metadata_counters())
        return metadata

    def _write_metadata_sheet(self, ws: Worksheet) -> None:
        """Feuille Metadata : tous les paramètres du run."""
        metadata: List[Tuple[str, Any]] = self._build_metadata_list()
        ws.cell(row=1, column=1, value="Parameter")
        ws.cell(row=1, column=2, value="Value")
        for i, (k, v) in enumerate(metadata, start=2):
            ws.cell(row=i, column=1, value=k)
            ws.cell(row=i, column=2, value=v)

    def _write_summary_sheet(self, ws: Worksheet) -> None:
        """Feuille Summary : un enregistrement par step."""
        headers: List[str] = [
            "step", "date", "mode", "n_nodes", "sum_p_node",
            "sum_p_exercised", "min_und", "max_und", "weighted_opt_price"]
        for j, h in enumerate(headers, start=1):
            ws.cell(row=1, column=j, value=h)
        for step_idx, nodes in enumerate(self.nodes_by_date):
            if not nodes:
                continue
            self._write_summary_row(ws, step_idx, nodes)

    def _write_summary_row(self, ws: Worksheet, step_idx: int,
                           nodes: List[Node]) -> None:
        """Écrit une ligne de résumé pour un step."""
        if step_idx < len(self.step_modes):
            mode: str = self.step_modes[step_idx]
        else:
            mode = ""
        p_sum: float = sum(n.p_node for n in nodes)
        p_ex: float = sum(n.p_node for n in nodes if n.is_exerced)
        unds: List[float] = [n.und_price for n in nodes]
        opt_w: float = sum(n.p_node * (n.opt_price or 0.0) for n in nodes)
        row: List[Any] = [step_idx, nodes[0].date.strftime("%Y-%m-%d"), mode,
                          len(nodes), round(p_sum, 10), round(p_ex, 10),
                          round(min(unds), 6), round(max(unds), 6),
                          round(opt_w, 6)]
        for j, v in enumerate(row, start=1):
            ws.cell(row=step_idx + 2, column=j, value=v)

    def _write_nodes_sheet(self, ws: Worksheet) -> None:
        """Feuille Nodes : un enregistrement par nœud."""
        headers: List[str] = [
            "step", "date", "und_price", "opt_price",
            "p_node", "p_up", "p_mid", "p_down",
            "is_trinomial_step", "is_exerced",
            "next_up_und", "next_mid_und", "next_down_und"]
        for j, h in enumerate(headers, start=1):
            ws.cell(row=1, column=j, value=h)
        r: int = 2
        for step_idx, nodes in enumerate(self.nodes_by_date):
            r = self._write_nodes_rows(ws, r, step_idx, nodes)

    def _write_nodes_rows(self, ws: Worksheet, r: int, step_idx: int,
                          nodes: List[Node]) -> int:
        """Écrit les lignes de tous les nœuds d'un step."""
        is_tri_step: bool = (step_idx < len(self.step_modes) - 1
                             and self.step_modes[step_idx + 1] == "TRI")
        for node in nodes:
            row: List[Any] = self._node_row(step_idx, node, is_tri_step)
            for j, v in enumerate(row, start=1):
                ws.cell(row=r, column=j, value=v)
            r += 1
        return r

    def _node_row(self, step_idx: int, node: Node,
                  is_tri_step: bool) -> List[Any]:
        """Construit la ligne Excel d'un nœud."""
        if node.opt_price is not None:
            opt_price: Optional[float] = round(node.opt_price, 6)
        else:
            opt_price = None
        next_up: Optional[float] = (round(node.next_up.und_price, 6)
                                     if node.next_up else None)
        next_mid: Optional[float] = (round(node.next_mid.und_price, 6)
                                      if node.next_mid else None)
        next_down: Optional[float] = (round(node.next_down.und_price, 6)
                                       if node.next_down else None)
        return [step_idx, node.date.strftime("%Y-%m-%d"),
                round(node.und_price, 6), opt_price,
                round(node.p_node, 10),
                round(node.p_up, 10), round(node.p_mid, 10),
                round(node.p_down, 10),
                bool(is_tri_step), bool(node.is_exerced),
                next_up, next_mid, next_down]

    def _write_diamond(self, ws: Worksheet, field: str) -> None:
        """Feuille en losange (und_price / opt_price / p_node)."""
        max_col: int = max((len(c) for c in self.nodes_by_date), default=1)
        center_row: int = max_col + 2
        for col_idx, nodes in enumerate(self.nodes_by_date):
            if not nodes:
                continue
            sorted_nodes: List[Node] = sorted(
                nodes, key=lambda n: n.und_price, reverse=True)
            n: int = len(sorted_nodes)
            start_row: int = center_row - (n - 1) // 2
            for i, node in enumerate(sorted_nodes):
                row: int = max(start_row + i, 1)
                val: Any = getattr(node, field, None)
                if isinstance(val, float):
                    val = round(val, 8)
                ws.cell(row=row, column=col_idx + 1, value=val)

    # -------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------
    def _has_div_step(self) -> bool:
        """Indique si au moins un pas trinomial a été déclenché."""
        return any(m == "TRI" for m in self.step_modes)

    def _overall_mode(self) -> str:
        """Nom lisible du mode global de construction."""
        if self.tree_mode == "trinomial_everywhere":
            return "TRINOMIAL everywhere"
        if self._has_div_step():
            return "MIXED (binomial + trinomial au pas ex-div)"
        return "BINOMIAL (pas de dividende)"

    def _debug_pruning_str(self) -> str:
        """Ligne debug pour le pruning."""
        if self.pricer_parameters.pruning:
            p_min: float = self.pricer_parameters.p_min
            return f"  pruning        = True (p_min={p_min:.2e})"
        return "  pruning        = False"

    def _debug_growth_str(self) -> str:
        """Ligne debug pour le ratio de croissance."""
        ratios: List[float] = [
            self.step_nodes[i + 1] / max(self.step_nodes[i], 1)
            for i in range(len(self.step_nodes) - 1)]
        growth: float = max(ratios) if ratios else 0.0
        modes_after: List[str] = self.step_modes[1:]
        bin_ratios: List[float] = [
            ratios[i] for i, m in enumerate(modes_after) if m == "BIN"]
        bin_growth: float = max(bin_ratios) if bin_ratios else 0.0
        verdict: str = "OK" if bin_growth < 2.5 else "⚠️ EXPLOSION"
        return (f"  Growth ratio (BIN max) = {bin_growth:.4f}  → {verdict}\n"
                f"  Growth ratio (global)  = {growth:.4f}")

    def _debug_report_header(self) -> List[str]:
        """Lignes d'en-tête du rapport debug."""
        lines: List[str] = ["=== Debug report ==="]
        lines.append(f"  mode           = {self._overall_mode()}")
        lines.append(f"  tree_mode param= {self.tree_mode}")
        lines.append(f"  steps          = {self.nb_steps}")
        lines.append(f"  α_bin          = {self.alpha_bin:.8f}")
        lines.append(f"  √α_bin         = {self.sqrt_alpha_bin:.8f}")
        lines.append(f"  α_tri (HW)     = {self.alpha_tri:.8f}")
        lines.append(f"  dividend       = {self.market_data.dividend}")
        if self.market_data.dividend > 0:
            ex_str: str = f"{self.market_data.ex_div_date:%Y-%m-%d}"
            lines.append(f"  ex-div date    = {ex_str}")
        return lines

    def debug_report(self) -> str:
        """Rapport debug sous forme de texte multilignes."""
        lines: List[str] = self._debug_report_header()
        lines.append(self._debug_pruning_str())
        lines.append(f"  nodes_created  = {self.counters['nodes_created']}")
        lines.append(f"  merges         = {self.counters['merges']}")
        lines.append(
            f"  clamped_probas = {self.counters.get('clamped_probas', 0)}")
        lines.append(f"  Taille max col = {max(self.step_nodes)}")
        lines.append(f"  Taille finale  = {self.step_nodes[-1]}")
        n_tri: int = sum(1 for m in self.step_modes if m == "TRI")
        lines.append(f"  Nb steps TRI   = {n_tri}  / "
                     f"{len(self.step_modes) - 1}")
        lines.append(self._debug_growth_str())
        lines.append(f"  build_time     = {self.build_time*1000:.3f} ms")
        lines.append(f"  backward_time  = {self.backward_time*1000:.3f} ms")
        lines.append(f"  total          = {self.price_time*1000:.3f} ms")
        if self.recursion_failed:
            lines.append("  recursion      = ⚠️ FAILED (RecursionError)")
        return "\n".join(lines)

    def debug_steps_dataframe(self) -> pd.DataFrame:
        """DataFrame des stats par step pour le panneau debug."""
        lengths: List[int] = [len(self.step_modes), len(self.step_nodes),
                              len(self.step_candidates),
                              len(self.step_times)]
        n: int = min(lengths)
        modes: List[str] = list(self.step_modes[:n])
        nodes: List[int] = list(self.step_nodes[:n])
        cands: List[int] = list(self.step_candidates[:n])
        times: List[float] = list(self.step_times[:n])
        parents: List[Optional[int]] = [None] + nodes[:-1]
        return pd.DataFrame({
            "step": list(range(n)), "mode": modes,
            "parents": parents, "candidates": cands, "nodes": nodes,
            "time (ms)": [float(t) * 1000.0 for t in times],
        })

    # -------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------
    def _price_variant(self, call_put: Optional[str] = None,
                       eur_am: Optional[str] = None,
                       div: Optional[float] = None) -> float:
        """Re-price l'option avec une variante de paramètres."""
        from option import Option as _Option
        from market import Market as _Market
        opt = self.option_data
        mkt = self.market_data
        new_opt = _Option(opt.K, opt.maturity,
                          call_put or opt.call_put, eur_am or opt.eur_am)
        if div is not None:
            new_div: float = div
        else:
            new_div = mkt.dividend
        new_mkt = _Market(mkt.s0, mkt.rate, mkt.vol, new_div,
                          mkt.ex_div_date)
        return Tree(self.pricer_parameters, new_mkt,
                    new_opt).backward_pricing()

    def _check_european_vs_bs(self) -> Dict[str, str]:
        """Euro sans div ≈ Black-Scholes."""
        from black_scholes import Black_Scholes
        opt, mkt, pr = (self.option_data, self.market_data,
                        self.pricer_parameters)
        name: str = "European without div ≈ BS"
        if opt.is_american() or mkt.dividend != 0 or mkt.vol <= 1e-8:
            return {"name": name, "value": "euro without div only",
                    "verdict": "N/A", "detail": ""}
        try:
            bs: float = Black_Scholes(pr, mkt, opt).pricing_BS()
            diff: float = abs(self.root.opt_price - bs)
            tol: float = _tol_convergence(abs(bs), self.nb_steps)
            if self.nb_steps < 30:
                verdict: str = "N/A"
            else:
                verdict = "PASS" if diff < tol else "FAIL"
            value = (f"tree={self.root.opt_price:.6f} BS={bs:.6f} "
                     f"diff={diff:.2e} tol={tol:.2e}")
            return {"name": name, "value": value,
                    "verdict": verdict, "detail": ""}
        except Exception as e:
            return {"name": name, "value": str(e),
                    "verdict": "N/A", "detail": ""}

    def _check_put_call_parity(self) -> Dict[str, str]:
        """Parité call-put pour une européenne sans dividende."""
        opt, mkt = self.option_data, self.market_data
        name: str = "Put-call parity : C − P = S₀ − K·e^(−rT)"
        if opt.is_american() or mkt.dividend != 0 or mkt.vol <= 1e-8:
            return {"name": "Put-call parity",
                    "value": "euro without div only",
                    "verdict": "N/A", "detail": ""}
        try:
            c: float = self._price_variant(call_put="call")
            p: float = self._price_variant(call_put="put")
            lhs: float = c - p
            rhs: float = mkt.s0 - opt.K * np.exp(-mkt.rate * self.T)
            diff: float = abs(lhs - rhs)
            scale: float = max(abs(lhs), abs(rhs), 1.0)
            tol: float = _tol_invariant(scale, self.nb_steps)
            verdict: str = "PASS" if diff < tol else "FAIL"
            return {"name": name,
                    "value": f"C−P={lhs:.6f} S−Ke^(−rT)={rhs:.6f} "
                             f"diff={diff:.2e} tol={tol:.2e}",
                    "verdict": verdict, "detail": ""}
        except Exception as e:
            return {"name": "Put-call parity", "value": str(e),
                    "verdict": "N/A", "detail": ""}

    def _check_american_call_no_div(self) -> Dict[str, str]:
        """A_call = E_call sans div, r > 0."""
        opt, mkt = self.option_data, self.market_data
        name: str = "A_call = E_call (without div, r>0)"
        if not (mkt.dividend == 0 and mkt.rate > 0
                and opt.call_put == "call"):
            return {"name": name,
                    "value": "call without div and r>0 only",
                    "verdict": "N/A", "detail": ""}
        try:
            a: float = self._price_variant(eur_am="american")
            e: float = self._price_variant(eur_am="european")
            diff: float = a - e
            tol: float = _tol_invariant(max(abs(e), 1.0), self.nb_steps)
            verdict: str = "PASS" if abs(diff) < tol else "FAIL"
            return {"name": name,
                    "value": f"A={a:.6f} E={e:.6f} diff={diff:.2e} "
                             f"tol={tol:.2e}",
                    "verdict": verdict, "detail": ""}
        except Exception as e:
            return {"name": "A_call = E_call", "value": str(e),
                    "verdict": "N/A", "detail": ""}

    def _check_american_put_geq_european(self) -> Dict[str, str]:
        """A_put ≥ E_put (prime d'exercice ≥ 0)."""
        opt = self.option_data
        if opt.call_put != "put":
            return {"name": "A_put ≥ E_put", "value": "put only",
                    "verdict": "N/A", "detail": ""}
        try:
            a: float = self._price_variant(eur_am="american")
            e: float = self._price_variant(eur_am="european")
            diff: float = a - e
            verdict: str = "PASS" if diff >= -1e-8 else "FAIL"
            return {"name": "A_put ≥ E_put",
                    "value": f"A={a:.6f} E={e:.6f} premium={diff:.2e}",
                    "verdict": verdict, "detail": "≥ 0"}
        except Exception as e:
            return {"name": "A_put ≥ E_put", "value": str(e),
                    "verdict": "N/A", "detail": ""}

    def _check_american_call_with_div(self) -> Dict[str, str]:
        """A_call ≥ E_call avec dividende."""
        opt, mkt = self.option_data, self.market_data
        name: str = "A_call ≥ E_call (div > 0)"
        if not (mkt.dividend > 0 and opt.call_put == "call"):
            return {"name": "A_call ≥ E_call (div>0)",
                    "value": "call with dividend only",
                    "verdict": "N/A", "detail": ""}
        try:
            a: float = self._price_variant(eur_am="american")
            e: float = self._price_variant(eur_am="european")
            diff: float = a - e
            verdict: str = "PASS" if diff >= -1e-8 else "FAIL"
            detail: str = "exercise just before ex-div"
            if abs(diff) < 1e-8:
                detail += " — premium = 0 (call OTM before ex-div ?)"
            return {"name": name,
                    "value": f"A={a:.6f} E={e:.6f} premium={diff:.2e}",
                    "verdict": verdict, "detail": detail}
        except Exception as e:
            return {"name": "A_call ≥ E_call (div>0)", "value": str(e),
                    "verdict": "N/A", "detail": ""}

    def sanity_checks(self) -> List[Dict[str, str]]:
        """Exécute tous les sanity checks et retourne la liste."""
        return [
            self._check_european_vs_bs(),
            self._check_put_call_parity(),
            self._check_american_call_no_div(),
            self._check_american_put_geq_european(),
            self._check_american_call_with_div(),
        ]

    def gap(self, given_gap: Optional[float] = None
            ) -> Tuple[float, float, Optional[float]]:
        """Calcule l'écart-type, le gap théorique et N requis."""
        T: float = self.T
        sd: float = (self.market_data.s0 * np.exp(self.market_data.rate * T)
                     * np.sqrt(np.exp(self.market_data.vol ** 2 * T) - 1))
        gap: float = ((3 * self.market_data.s0) / (8 * np.sqrt(2 * np.pi))
                      * (np.exp(self.market_data.vol ** 2 * self.delta_t) - 1)
                      * np.exp(2 * self.market_data.rate * self.delta_t)
                      / np.sqrt(np.exp(self.market_data.vol ** 2 * T) - 1))
        calc: Optional[float] = None
        if given_gap is not None and given_gap > 0:
            up: float = self.market_data.vol ** 2 * T
            down: float = np.log(
                1 + (8 * np.sqrt(2 * np.pi) * given_gap
                     * np.sqrt(np.exp(self.market_data.vol ** 2 * T) - 1))
                / (3 * self.market_data.s0))
            if abs(down) > 1e-15:
                calc = round(up / down)
        return sd, gap, calc
