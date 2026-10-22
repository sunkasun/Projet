"""Nœuds de l'arbre binomial/trinomial."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Tuple

import numpy as np


class Node:
    """Nœud standard de l'arbre."""

    def __init__(self, und_price: float, associated_tree: 'Tree',
                 date: dt.datetime) -> None:
        """Initialise un nœud de l'arbre."""
        self.und_price = und_price
        self.tree = associated_tree
        self.date = date
        self.opt_price: Optional[float] = None
        self.p_up = self.p_mid = self.p_down = 0.0
        self.p_node = 0.0
        self.next_up: Optional['Node'] = None
        self.next_mid: Optional['Node'] = None
        self.next_down: Optional['Node'] = None
        self.up_node: Optional['Node'] = None
        self.down_node: Optional['Node'] = None
        self.is_exerced = False

    def dividend_period(self) -> bool:
        """Indique si ce nœud se trouve avant une date ex-div."""
        if self.tree.market_data.dividend == 0:
            return False
        dt_days = dt.timedelta(days=self.tree.delta_t * 365)
        return (self.date < self.tree.market_data.ex_div_date
                and self.date + dt_days >= self.tree.market_data.ex_div_date)

    def is_trinomial_step(self) -> bool:
        """Indique si ce nœud déclenche un pas trinomial."""
        if self.tree.tree_mode == "trinomial_everywhere":
            return True
        return self.dividend_period()

    def forward_variance(self) -> Tuple[float, float]:
        """Retourne (espérance, variance) du forward sur un pas."""
        r = self.tree.market_data.rate
        sigma = self.tree.market_data.vol
        dt_ = self.tree.delta_t
        S = self.und_price
        if self.dividend_period():
            return self._forward_variance_div(r, sigma, dt_, S)
        E = S * np.exp(r * dt_)
        V = S ** 2 * np.exp(2 * r * dt_) * (np.exp(sigma ** 2 * dt_) - 1)
        return E, V

    def _forward_variance_div(self, r: float, sigma: float, dt_: float,
                              S: float) -> Tuple[float, float]:
        """Forward/variance quand un dividende tombe dans le pas."""
        tau_days = ((self.tree.market_data.ex_div_date
                     - self.date).total_seconds() / 86400)
        dt_days = dt_ * 365
        tau_ratio = tau_days / dt_days if dt_days > 1e-15 else 1.0
        tau_ratio = max(0.0, min(1.0, tau_ratio))
        D = self.tree.market_data.dividend
        E = S * np.exp(r * dt_) - D * np.exp(r * dt_ * (1.0 - tau_ratio))
        V = ((S - (1.0 - tau_ratio) * D) ** 2
             * np.exp(2 * r * dt_) * (np.exp(sigma ** 2 * dt_) - 1))
        return E, V

    def child_specs(self) -> List[Tuple[float, str]]:
        """Retourne [(prix_candidat, position), ...] pour l'étape suivante."""
        E, _ = self.forward_variance()
        if (self.tree.pricer_parameters.pruning
                and self.p_node < self.tree.pricer_parameters.p_min):
            return [(E, 'mid')]
        if self.is_trinomial_step():
            return self._trinomial_specs(E)
        return self._binomial_specs(E)

    def _trinomial_specs(self, E: float) -> List[Tuple[float, str]]:
        """Enfants d'un pas trinomial."""
        if self.tree.tree_mode == "trinomial_everywhere":
            a = self.tree.alpha_tri
        else:
            a = self.tree.alpha_bin
        if abs(a - 1.0) < 1e-10:
            return [(E, 'mid')]
        return [(E * a, 'up'), (E, 'mid'), (E / a, 'down')]

    def _binomial_specs(self, E: float) -> List[Tuple[float, str]]:
        """Enfants d'un pas binomial."""
        b = self.tree.sqrt_alpha_bin
        if abs(b - 1.0) < 1e-10:
            return [(E, 'mid')]
        return [(E * b, 'up'), (E / b, 'down')]

    def probas(self) -> None:
        """Calcule p_up, p_mid, p_down puis normalise (clamp+renormalize)."""
        if (self.tree.pricer_parameters.pruning
                and self.p_node < self.tree.pricer_parameters.p_min):
            self.p_up, self.p_mid, self.p_down = 0.0, 1.0, 0.0
            return
        if self.is_trinomial_step():
            self._compute_trinomial_probas()
        else:
            self._compute_binomial_probas()
        self._clamp_probas()

    def _compute_trinomial_probas(self) -> None:
        """Probabilités trinomiales par matching de moments (Hull-White)."""
        E, V = self.forward_variance()
        if self.tree.tree_mode == "trinomial_everywhere":
            a = self.tree.alpha_tri
        else:
            a = self.tree.alpha_bin
        if abs(a - 1.0) < 1e-10 or self.next_mid is None:
            self.p_up, self.p_mid, self.p_down = 0.0, 1.0, 0.0
            return
        m = self.next_mid.und_price
        self.p_down = ((m ** (-2) * (V + E ** 2) - 1
                        - (a + 1) * (m ** (-1) * E - 1))
                       / ((1 - a) * (a ** (-2) - 1)))
        self.p_up = ((m ** (-1) * E - 1)
                     - (a ** (-1) - 1) * self.p_down) / (a - 1)
        self.p_mid = 1.0 - self.p_up - self.p_down

    def _compute_binomial_probas(self) -> None:
        """Probabilités binomiales (1/1+b, b/1+b)."""
        b = self.tree.sqrt_alpha_bin
        if abs(b - 1.0) < 1e-10:
            self.p_up, self.p_mid, self.p_down = 0.0, 1.0, 0.0
            return
        self.p_up = 1.0 / (b + 1.0)
        self.p_down = b / (b + 1.0)
        self.p_mid = 0.0

    def _clamp_probas(self) -> None:
        """Clamp les probas à >= 0 puis renormalise pour sommer à 1."""
        raw = [self.p_up, self.p_mid, self.p_down]
        clamped = [max(0.0, p) for p in raw]
        total = sum(clamped)
        if total > 0:
            self.p_up, self.p_mid, self.p_down = [
                c / total for c in clamped]
        else:
            self.p_up, self.p_mid, self.p_down = 0.0, 1.0, 0.0
        if any(p < -1e-10 for p in raw):
            self.tree.counters["clamped_probas"] = \
                self.tree.counters.get("clamped_probas", 0) + 1

    def next_nodes_probas(self) -> None:
        """Propage p_node aux enfants."""
        for p, n in ((self.p_up, self.next_up),
                     (self.p_mid, self.next_mid),
                     (self.p_down, self.next_down)):
            if n is not None and p != 0:
                n.p_node += self.p_node * p

    def price(self) -> float:
        """Prix backward récursif (avec exercice anticipé si américain)."""
        if self.opt_price is not None:
            return self.opt_price
        if (self.next_up is None and self.next_mid is None
                and self.next_down is None):
            self.opt_price = self.tree.option_data.payoff(self.und_price)
            return self.opt_price
        pu = self.next_up.price() if self.next_up else 0.0
        pm = self.next_mid.price() if self.next_mid else 0.0
        pd = self.next_down.price() if self.next_down else 0.0
        price = ((pu * self.p_up + pm * self.p_mid + pd * self.p_down)
                 * self.tree.df)
        if self.tree.option_data.is_american():
            ex = self.tree.option_data.payoff(self.und_price)
            if ex > price:
                price = ex
                self.is_exerced = True
        self.opt_price = price
        return price


class TruncNode(Node):
    """Nœud tronqué conservant une référence vers son parent."""

    def __init__(self, und_price: float, associated_tree: 'Tree',
                 date: dt.datetime,
                 prec_node: Optional['Node'] = None) -> None:
        """Initialise le nœud tronqué avec son parent."""
        super().__init__(und_price, associated_tree, date)
        self.prec_node = prec_node