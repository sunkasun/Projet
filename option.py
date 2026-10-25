import datetime as dt

class Option:
    """
    This class stores the information specific to an option, namely:
        - the strike price, K,
        - the maturity date,
        - its nature (call option / put option),
        - its type (European / American).
    It also contains two methods: one to check whether the option is American,
    and one to compute the option's payoff.
    """
    def __init__(self, K: float, maturity: dt.datetime, call_put: str, eur_am: str) -> None:
        # First, check that the option type is correctly spelled.
        # With the user interface, parameters are automatically spelled correctly.
        if call_put not in ["call", "put"] or eur_am  not in ["european", "american"]:
            raise TypeError("The option must be a call or put, European or American. Check your argument format.\n \
                            String format: 'call', 'put', 'european', 'american'.")
        self.K: float = K
        self.maturity: dt.datetime = maturity
        self.call_put: str = call_put
        self.eur_am: str = eur_am

    def payoff(self, und_price: float) -> float:
        """
        Computes the payoff of the option as the maximum between its intrinsic value and 0.
        Since we verify at instantiation that the option is indeed a call or a put,
        we can safely consider only these two cases.
        """
        payoff = max(und_price - self.K, 0) if self.call_put == "call" else max(self.K - und_price , 0)
        return payoff
    
    def is_american(self) -> bool:
        """
        Boolean translation of the early exercise feature.
        """
        return self.eur_am == "american"
