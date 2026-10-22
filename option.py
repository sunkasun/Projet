import datetime as dt


class Option:
    """
    Cette classe stocke les informations spécifiques à une option, à savoir :
        - le prix d'exercice, K,
        - la date de maturité,
        - sa nature (call / put),
        - son type (européenne / américaine).
    Elle contient aussi deux méthodes : une pour vérifier si l'option est
    américaine, et une pour calculer le payoff de l'option.
    """

    def __init__(self, K: float, maturity: dt.datetime,
                 call_put: str, eur_am: str) -> None:
        """Initialise l'option et vérifie le format des arguments."""
        # Vérifie que le type d'option est correctement orthographié.
        if (call_put not in ["call", "put"]
                or eur_am not in ["european", "american"]):
            raise TypeError(
                "The option must be a call or put, European or American. "
                "Check your argument format.\n "
                "String format: 'call', 'put', 'european', 'american'.")
        self.K: float = K
        self.maturity: dt.datetime = maturity
        self.call_put: str = call_put
        self.eur_am: str = eur_am

    def payoff(self, und_price: float) -> float:
        """
        Calcule le payoff comme le maximum entre la valeur intrinsèque et 0.
        L'option étant validée à l'instanciation, on gère uniquement call/put.
        """
        if self.call_put == "call":
            payoff = max(und_price - self.K, 0)
        else:
            payoff = max(self.K - und_price, 0)
        return payoff

    def is_american(self) -> bool:
        """Traduit en booléen la possibilité d'exercice anticipé."""
        return self.eur_am == "american"