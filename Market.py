class Market:
    def __init__(self, input: dict):
        self.SpotPrice = input["SpotPrice"]
        self.RiskFree = input["RiskFree"] / 100
        self.volatility = input["Volatility"] / 100
        self.dividend = input["Dividend"]
        self.div_date = self.compute_date_div(input)

    def compute_date_div(self, input: dict) -> float:
        delta_days = (input["DivExDate"] - input["PricingDate"]).days
        return delta_days / 365

    def compute_dividend(self, T: int, dt: float) -> float:
        if (T - 1) * dt < self.div_date <= T * dt:
            return self.dividend
        return 0.0