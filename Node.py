import math


class Node:
    # Compteurs (class-level, ne ralentissent QUE si COUNTING=True)
    COUNTING: bool = False
    nb_alloc: int = 0
    max_live: int = 0

    __slots__ = (
        "UndPrice", "Payoff",
        "Next_Up", "Next_Down", "Next_Mid",
        "nprec", "UpNode", "DownNode", "MidNode",
        "Proba_Up", "Proba_Down", "Proba_Mid",
        "Cum_Proba", "OptPrice",
    )

    def __new__(cls, *args, **kwargs):
        if cls.COUNTING:
            cls.nb_alloc += 1
        return super().__new__(cls)

    def __init__(self):
        self.UndPrice = 0.0
        self.Payoff = 0.0
        self.Next_Up = self.Next_Down = self.Next_Mid = None
        self.nprec = None
        self.UpNode = self.DownNode = self.MidNode = None
        self.Proba_Up = 0.0
        self.Proba_Down = 0.0
        self.Proba_Mid = 0.0
        self.Cum_Proba = 0.0
        self.OptPrice = 0.0

    # ---- API de comptage (à appeler seulement en debug) ----
    @classmethod
    def live_count(cls) -> int:
        import gc
        return sum(1 for o in gc.get_objects() if isinstance(o, cls))

    @classmethod
    def reset_counters(cls):
        cls.nb_alloc = 0
        cls.max_live = 0

    # ---- Méthodes métier ----
    def forward(self, mkt, dt: float, div: float):
        return self.UndPrice * math.exp(mkt.RiskFree * dt) - div

    def init_node(self, market_value: float):
        self.UndPrice = market_value

    def incremental_node(self, candidate_mid, is_up: bool, condition: bool):
        if is_up:
            if candidate_mid.UpNode is not None:
                candidate_mid = candidate_mid.UpNode
            else:
                condition = False
        else:
            if candidate_mid.DownNode is not None:
                candidate_mid = candidate_mid.DownNode
            else:
                condition = False
        return candidate_mid, condition

    def price(self, candidate_mid, parameters, mkt, tree):
        while candidate_mid.nprec is not None:
            condition = True
            trunk = candidate_mid
            while condition:
                self.payoff_condition(candidate_mid, parameters, mkt, tree)
                candidate_mid, condition = self.incremental_node(candidate_mid, True, condition)
            candidate_mid = trunk.DownNode
            condition = True
            while condition:
                self.payoff_condition(candidate_mid, parameters, mkt, tree)
                candidate_mid, condition = self.incremental_node(candidate_mid, False, condition)
            candidate_mid = trunk
            candidate_mid = candidate_mid.nprec

        self.discounted_price(candidate_mid, parameters, mkt, tree)
        if candidate_mid.UpNode is not None:
            self.discounted_price(candidate_mid.UpNode, parameters, mkt, tree)
            self.discounted_price(candidate_mid.DownNode, parameters, mkt, tree)

    def payoff_condition(self, candidate_mid, parameters, mkt, tree):
        if candidate_mid.Next_Mid is None:
            candidate_mid.OptPrice = parameters.Vi(candidate_mid.UndPrice)
        else:
            self.discounted_price(candidate_mid, parameters, mkt, tree)

    def discounted_price(self, candidate_mid, parameters, mkt, tree):
        val_up = val_mid = val_down = 0.0
        if candidate_mid.Next_Up is not None:
            val_up = candidate_mid.Next_Up.OptPrice
            candidate_mid.Next_Up = None
        if candidate_mid.Next_Mid is not None:
            val_mid = candidate_mid.Next_Mid.OptPrice
            candidate_mid.Next_Mid = None
        if candidate_mid.Next_Down is not None:
            val_down = candidate_mid.Next_Down.OptPrice
            candidate_mid.Next_Down = None

        disc = math.exp(-mkt.RiskFree * tree.dt)
        if parameters.Exercice == "European":
            candidate_mid.OptPrice = (val_up * candidate_mid.Proba_Up +
                                      val_mid * candidate_mid.Proba_Mid +
                                      val_down * candidate_mid.Proba_Down) * disc
        else:
            candidate_mid.OptPrice = max(
                parameters.Vi(candidate_mid.UndPrice),
                (val_up * candidate_mid.Proba_Up +
                 val_mid * candidate_mid.Proba_Mid +
                 val_down * candidate_mid.Proba_Down) * disc,
            )