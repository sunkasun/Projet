import datetime as dt

class Market: 
    """
    This class lists the market data as of the calculation date. It includes:
        - the initial price of the underlying asset, s0,
        - the risk-free rate, r,
        - the volatility of the underlying asset, vol,
        - the amount of any discrete dividend, if it exists, and the date immediately after its payment.
    If there are no dividends, the maturity date is set to December 31, 2099.
    """
    def __init__(self, s0: float, rate: float, vol: float, dividend_amount: float, ex_div_date: dt.datetime, continuous_yield: float = 0.0) -> None:
        if s0 < 0 or vol < 0:
            raise ValueError("The underlying asset and volatility must have positive initial values.")
        self.s0: float = s0
        self.rate: float = rate
        self.vol: float = vol
        self.dividend: float = dividend_amount
        self.ex_div_date: dt.datetime = ex_div_date if self.dividend !=0 else dt.datetime(2099,12,31)
        self.continuous_yield: float = continuous_yield
