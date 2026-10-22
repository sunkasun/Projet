import datetime as dt


class Market:
    """
    Cette classe liste les données de marché à la date de calcul. Elle inclut :
        - le prix initial du sous-jacent, s0,
        - le taux sans risque, r,
        - la volatilité du sous-jacent, vol,
        - le montant du dividende discret s'il existe et la date ex-div.
    S'il n'y a pas de dividende, la date ex-div est fixée au 31/12/2099.
    """

    def __init__(self, s0: float, rate: float, vol: float,
                 dividend_amount: float, ex_div_date: dt.datetime) -> None:
        """Initialise les données de marché et valide s0 et vol positifs."""
        if s0 < 0 or vol < 0:
            raise ValueError(
                "Le sous-jacent et la volatilité doivent être positifs.")
        self.s0: float = s0
        self.rate: float = rate
        self.vol: float = vol
        self.dividend: float = dividend_amount
        # Si pas de dividende, on utilise une date sentinelle
        if self.dividend != 0:
            self.ex_div_date: dt.datetime = ex_div_date
        else:
            self.ex_div_date = dt.datetime(2099, 12, 31)