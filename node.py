from __future__ import annotations

import datetime as dt
import numpy as np
from typing import Tuple, List


class Node:
    def __init__(self, und_price: float, associated_tree: 'Tree', date: dt.datetime) -> None:
        self.und_price = und_price
        self.tree = associated_tree
        self.date = date
        self.opt_price = None
        self.p_up = self.p_mid = self.p_down = 0.0
        self.p_node = 0.0
        self.next_up = self.next_mid = self.next_down = None
        self.up_node = self.down_node = None
        self.is_exerced = False

    def dividend_period(self) -> bool:
        if self.tree.market_data.dividend == 0:
            return False
        return (self.date < self.tree.market_data.ex_div_date and
                self.date + dt.timedelta(days=self.tree.delta_t * 365)
                >= self.tree.market_data.ex_div_date)

    def is_trinomial_step(self) -> bool:
        if self.tree.tree_mode == "trinomial_everywhere":
            return True
        return self.dividend_period()

    def forward_variance(self) -> Tuple[float, float]:
        r = self.tree.market_data.rate
        sigma = self.tree.market_data.vol
        dt_ = self.tree.delta_t
        S = self.und_price
        q = self.tree.market_data.continuous_yield
        mu = r - q

        if self.dividend_period():
            tau_days = (self.tree.market_data.ex_div_date - self.date).total_seconds() / 86400
            dt_days = dt_ * 365
            tau_ratio = tau_days / dt_days if dt_days > 1e-15 else 1.0
            tau_ratio = max(0.0, min(1.0, tau_ratio))
            D = self.tree.market_data.dividend
            E = S * np.exp(mu * dt_) - D * np.exp(r * dt_ * (1.0 - tau_ratio))
            V = (S - (1.0 - tau_ratio) * D) ** 2 \
                * np.exp(2 * mu * dt_) * (np.exp(sigma ** 2 * dt_) - 1)
        else:
            E = S * np.exp(mu * dt_)
            V = S ** 2 * np.exp(2 * mu * dt_) * (np.exp(sigma ** 2 * dt_) - 1)
        return E, V

    def child_specs(self) -> List[Tuple[float, str]]:
        """Retourne [(prix_candidat, position), ...] pour l'étape suivante."""
        E, _ = self.forward_variance()

        if (self.tree.pricer_parameters.pruning
                and self.p_node < self.tree.pricer_parameters.p_min):
            return [(E, 'mid')]

        if self.is_trinomial_step():
            a = self.tree.alpha_tri if self.tree.tree_mode == "trinomial_everywhere" \
                else self.tree.alpha_bin
            if abs(a - 1.0) < 1e-10:
                return [(E, 'mid')]
            return [(E * a, 'up'), (E, 'mid'), (E / a, 'down')]
        else:
            b = self.tree.sqrt_alpha_bin
            if abs(b - 1.0) < 1e-10:
                return [(E, 'mid')]
            return [(E * b, 'up'), (E / b, 'down')]

    def probas(self) -> None:
        if (self.tree.pricer_parameters.pruning
                and self.p_node < self.tree.pricer_parameters.p_min):
            self.p_up, self.p_mid, self.p_down = 0.0, 1.0, 0.0
            return

        E, V = self.forward_variance()

        if self.is_trinomial_step():
            a = self.tree.alpha_tri if self.tree.tree_mode == "trinomial_everywhere" \
                else self.tree.alpha_bin
            if abs(a - 1.0) < 1e-10 or self.next_mid is None:
                self.p_up, self.p_mid, self.p_down = 0.0, 1.0, 0.0
                return
            m = self.next_mid.und_price
            self.p_down = ((m ** (-2) * (V + E ** 2) - 1 - (a + 1) * (m ** (-1) * E - 1))
                           / ((1 - a) * (a ** (-2) - 1)))
            self.p_up = ((m ** (-1) * E - 1) - (a ** (-1) - 1) * self.p_down) / (a - 1)
            self.p_mid = 1.0 - self.p_up - self.p_down
        else:
            b = self.tree.sqrt_alpha_bin
            if abs(b - 1.0) < 1e-10:
                self.p_up, self.p_mid, self.p_down = 0.0, 1.0, 0.0
                return
            self.p_up = 1.0 / (b + 1.0)
            self.p_down = b / (b + 1.0)
            self.p_mid = 0.0

        # ---------- FIX B : clamp + renormalize ----------
        raw = [self.p_up, self.p_mid, self.p_down]
        clamped = [max(0.0, p) for p in raw]
        total = sum(clamped)
        if total > 0:
            self.p_up, self.p_mid, self.p_down = [c / total for c in clamped]
        else:
            self.p_up, self.p_mid, self.p_down = 0.0, 1.0, 0.0

        # Compteur : combien de fois le clamp a été nécessaire
        if any(p < -1e-10 for p in raw):
            self.tree.counters["clamped_probas"] = \
                self.tree.counters.get("clamped_probas", 0) + 1

    def next_nodes_probas(self) -> None:
        for p, n in ((self.p_up, self.next_up),
                     (self.p_mid, self.next_mid),
                     (self.p_down, self.next_down)):
            if n is not None and p != 0:
                n.p_node += self.p_node * p

    def price(self) -> float:
        if self.opt_price is not None:
            return self.opt_price
        if self.next_up is None and self.next_mid is None and self.next_down is None:
            self.opt_price = self.tree.option_data.payoff(self.und_price)
            return self.opt_price
        pu = self.next_up.price()   if self.next_up   else 0.0
        pm = self.next_mid.price()  if self.next_mid  else 0.0
        pd = self.next_down.price() if self.next_down else 0.0
        price = (pu * self.p_up + pm * self.p_mid + pd * self.p_down) * self.tree.df
        if self.tree.option_data.is_american():
            ex = self.tree.option_data.payoff(self.und_price)
            if ex > price:
                price = ex
                self.is_exerced = True
        self.opt_price = price
        return price


class TruncNode(Node):
    def __init__(self, und_price, associated_tree, date, prec_node=None):
        super().__init__(und_price, associated_tree, date)
        self.prec_node = prec_node
