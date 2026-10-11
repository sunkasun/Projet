import numpy as np
from datetime import timedelta
from scipy.stats import norm


class Extension:
    def __init__(self, market, params, pricer):
        self.market = market
        self.params = params
        self.pricer = pricer

    # ============ BLACK-SCHOLES ============
    def black_sholes(self):
        ttm = (self.params.DateMaturity - self.pricer.PricingDate).days / 365.0
        d1 = self.compute_d1(ttm)
        d2 = self.compute_d2(d1, ttm)
        if self.params.type_contrat == "Call":
            self.pricer.BS_price = self.compute_call_price(ttm, d1, d2)
            delta = self.compute_call_delta(d1)
            rho = self.compute_rho_call(ttm, d2)
            theta = self.compute_theta_call(ttm, d1, d2)
        else:
            self.pricer.BS_price = self.compute_put_price(ttm, d1, d2)
            delta = self.compute_put_delta(d1)
            rho = self.compute_rho_put(ttm, d2)
            theta = self.compute_theta_put(ttm, d1, d2)
        vega = self.compute_vega(ttm, d1)
        gamma = self.compute_gamma(ttm, d1)
        return {"BS Price": self.pricer.BS_price, "Delta": delta, "Vega": vega,
                "Rho": rho, "Gamma": gamma, "Theta": theta}

    def compute_d1(self, ttm):
        q = self.market.dividend / self.market.SpotPrice
        return (np.log(self.market.SpotPrice / self.params.strike) +
                (self.market.RiskFree - q + 0.5 * self.market.volatility ** 2) * ttm) / \
               (self.market.volatility * np.sqrt(ttm))

    def compute_d2(self, d1, ttm):
        return d1 - self.market.volatility * np.sqrt(ttm)

    def compute_call_price(self, ttm, d1, d2):
        q = self.market.dividend / self.market.SpotPrice
        return (self.market.SpotPrice * np.exp(-q * ttm) * norm.cdf(d1) -
                self.params.strike * np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2))

    def compute_put_price(self, ttm, d1, d2):
        q = self.market.dividend / self.market.SpotPrice
        return (self.params.strike * np.exp(-self.market.RiskFree * ttm) * norm.cdf(-d2) -
                self.market.SpotPrice * np.exp(-q * ttm) * norm.cdf(-d1))

    def compute_call_delta(self, d1):
        return norm.cdf(d1)

    def compute_put_delta(self, d1):
        return norm.cdf(d1) - 1

    def compute_vega(self, ttm, d1):
        q = self.market.dividend / self.market.SpotPrice
        return self.market.SpotPrice * np.exp(-q * ttm) * np.sqrt(ttm) * norm.pdf(d1)

    def compute_gamma(self, ttm, d1):
        q = self.market.dividend / self.market.SpotPrice
        return (np.exp(-q * ttm) * norm.pdf(d1)) / \
               (self.market.SpotPrice * self.market.volatility * np.sqrt(ttm))

    def compute_rho_call(self, ttm, d2):
        return self.params.strike * ttm * np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2)

    def compute_rho_put(self, ttm, d2):
        return -self.params.strike * ttm * np.exp(-self.market.RiskFree * ttm) * norm.cdf(-d2)

    def compute_theta_call(self, ttm, d1, d2):
        q = self.market.dividend / self.market.SpotPrice
        return (-self.market.SpotPrice * np.exp(-q * ttm) * norm.pdf(d1) * self.market.volatility / (2 * np.sqrt(ttm))
                - self.market.RiskFree * self.params.strike * np.exp(-self.market.RiskFree * ttm) * norm.cdf(d2)
                + q * self.market.SpotPrice * np.exp(-q * ttm) * norm.cdf(d1)) / 365

    def compute_theta_put(self, ttm, d1, d2):
        q = self.market.dividend / self.market.SpotPrice
        return (-self.market.SpotPrice * np.exp(-q * ttm) * norm.pdf(d1) * self.market.volatility / (2 * np.sqrt(ttm))
                - self.market.RiskFree * self.params.strike * np.exp(-self.market.RiskFree * ttm) * (1 - norm.cdf(d2))
                - q * self.market.SpotPrice * np.exp(-q * ttm) * (1 - norm.cdf(d1))) / 365

    # ============ GREEKS BS FERMÉS ============
    def bs_greeks(self):
        ttm = (self.params.DateMaturity - self.pricer.PricingDate).days / 365.0
        S, K, r, sigma = (self.market.SpotPrice, self.params.strike,
                          self.market.RiskFree, self.market.volatility)
        q = self.market.dividend / S

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * ttm) / (sigma * np.sqrt(ttm))
        d2 = d1 - sigma * np.sqrt(ttm)
        pdf1 = norm.pdf(d1)
        cdf1, cdf2 = norm.cdf(d1), norm.cdf(d2)
        eqT, erT = np.exp(-q * ttm), np.exp(-r * ttm)

        if self.params.type_contrat == "Call":
            delta = eqT * cdf1
            theta = (-S * eqT * pdf1 * sigma / (2 * np.sqrt(ttm)) - r * K * erT * cdf2
                     + q * S * eqT * cdf1)
        else:
            delta = -eqT * norm.cdf(-d1)
            theta = (-S * eqT * pdf1 * sigma / (2 * np.sqrt(ttm)) + r * K * erT * norm.cdf(-d2)
                     - q * S * eqT * norm.cdf(-d1))

        gamma = eqT * pdf1 / (S * sigma * np.sqrt(ttm))
        vega = S * eqT * np.sqrt(ttm) * pdf1
        vanna = -eqT * pdf1 * d2 / sigma
        vomma = vega * d1 * d2 / sigma

        return {"Delta": delta, "Gamma": gamma, "Vega": vega / 100,
                "Vomma": vomma / 10000, "Vanna": vanna / 100, "Theta": theta / 365}

    # ============ MONTE-CARLO ============
    def monte_carlo_price(self, n_simulations: int) -> float:
        ttm = (self.params.DateMaturity - self.pricer.PricingDate).days / 365.0
        payoff_sum = 0.0
        for _ in range(n_simulations):
            s = self.simulate_asset_price(self.market.SpotPrice, ttm)
            payoff = (max(s - self.params.strike, 0) if self.params.type_contrat == "Call"
                      else max(self.params.strike - s, 0))
            payoff_sum += payoff
        self.pricer.MC_price = np.exp(-self.market.RiskFree * ttm) * (payoff_sum / n_simulations)
        return self.pricer.MC_price

    def simulate_asset_price(self, S, ttm):
        dt = ttm / self.pricer.timeSteps
        for _ in range(self.pricer.timeSteps):
            z = np.random.normal()
            S = S * np.exp((self.market.RiskFree - self.market.dividend / self.market.SpotPrice
                            - 0.5 * self.market.volatility ** 2) * dt
                           + self.market.volatility * np.sqrt(dt) * z)
            if S < 0:
                S = 0
        return S

    def monte_carlo_greeks(self, n_sim: int = 50000, seed: int = 42):
        ttm = (self.params.DateMaturity - self.pricer.PricingDate).days / 365.0
        n_steps = self.pricer.timeSteps
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_sim, n_steps))

        def mc_price(S0, sigma, r, T):
            dt = T / n_steps
            log_paths = np.cumsum((r - 0.5 * sigma ** 2) * dt
                                  + sigma * np.sqrt(dt) * Z, axis=1)
            S_T = S0 * np.exp(log_paths[:, -1])
            payoff = (np.maximum(S_T - self.params.strike, 0) if self.params.type_contrat == "Call"
                      else np.maximum(self.params.strike - S_T, 0))
            return np.exp(-r * T) * np.mean(payoff)

        S0, sigma, r = self.market.SpotPrice, self.market.volatility, self.market.RiskFree
        P0 = mc_price(S0, sigma, r, ttm)
        h_S, h_sig = S0 * 0.01, 0.01
        P_Su = mc_price(S0 + h_S, sigma, r, ttm); P_Sd = mc_price(S0 - h_S, sigma, r, ttm)
        P_su = mc_price(S0, sigma + h_sig, r, ttm); P_sd = mc_price(S0, sigma - h_sig, r, ttm)
        P_uu = mc_price(S0 + h_S, sigma + h_sig, r, ttm)
        P_ud = mc_price(S0 + h_S, sigma - h_sig, r, ttm)
        P_du = mc_price(S0 - h_S, sigma + h_sig, r, ttm)
        P_dd = mc_price(S0 - h_S, sigma - h_sig, r, ttm)
        ttm_s = (self.params.DateMaturity - timedelta(days=1) - self.pricer.PricingDate).days / 365.0
        P_sh = mc_price(S0, sigma, r, ttm_s)

        return {
            "Delta": (P_Su - P_Sd) / (2 * h_S),
            "Gamma": (P_Su - 2 * P0 + P_Sd) / (h_S ** 2),
            "Vega": (P_su - P_sd) / (2 * h_sig) / 100,
            "Vomma": (P_su - 2 * P0 + P_sd) / (h_sig ** 2) / 10000,
            "Vanna": (P_uu - P_ud - P_du + P_dd) / (4 * h_S * h_sig) / 100,
            "Theta": P_sh - P0,
        }