import datetime as dt


class PricerParameters:
    """
    Paramètres du pricer.
    tree_mode :
      - "binomial_with_div_switch" : binomial partout, trinomial UNIQUEMENT
        sur le pas qui contient la date ex-div (mode du prof).
      - "trinomial_everywhere" : trinomial partout (α = e^{σ√3Δt}).
    """

    def __init__(self, pricing_date: dt.datetime, nb_steps: int,
                 pruning: bool = True, p_min: float = 1e-9,
                 tree_mode: str = "binomial_with_div_switch"):
        """Initialise les paramètres du pricer."""
        self.pricing_date = pricing_date
        self.nb_steps = nb_steps
        self.pruning = pruning
        self.p_min = p_min
        self.tree_mode = tree_mode