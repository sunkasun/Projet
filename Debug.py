import math
import pandas as pd


class DebugChecks:
    """Vérifie les propriétés théoriques connues sur les prix d'options."""

    def __init__(self, run_calc_fn, base_input: dict):
        self.run = run_calc_fn
        self.base = base_input

    def _run(self, **overrides):
        d = dict(self.base)
        d.update(overrides)
        d["Convergence"] = False
        d["StrikeStudy"] = False
        d["TreeBol"] = False
        d["MCCondition"] = False
        d["ComputeGreeks"] = False
        d["DebugMode"] = False
        res = self.run(d)
        return res.get("TreePrice", float("nan")), res

    # ---------- Tests unitaires ----------
    def test_K0_call(self):
        p, _ = self._run(Strike=0, Type="Call")
        S0 = self.base["SpotPrice"]
        div = self.base.get("Dividend", 0.0)
        if div > 0:
            r = self.base["RiskFree"] / 100
            tdiv = (self.base["DivExDate"] - self.base["PricingDate"]).days / 365
            expected = S0 - div * math.exp(-r * tdiv)
        else:
            expected = S0
        return self._row("K=0, call → S0 (ou S0-PV(div))", p, expected, tol=1e-1)

    def test_K0_put(self):
        p, _ = self._run(Strike=0, Type="Put")
        return self._row("K=0, put → 0", p, 0.0, tol=1e-2)

    def test_european_vs_BS(self):
        d = dict(self.base, Dividend=0, Exercice="European", Type="Call")
        p, res = self._run(**{k: d[k] for k in ("Dividend", "Exercice", "Type")})
        p_bs = res.get("BS Price", float("nan"))
        return self._row("Euro call sans div ≈ BS", p, p_bs, tol=5e-2)

    def test_put_call_parity(self):
        pc, _ = self._run(Type="Call", Exercice="European")
        pp, _ = self._run(Type="Put", Exercice="European")
        S0 = self.base["SpotPrice"]
        ttm = (self.base["Maturity"] - self.base["PricingDate"]).days / 365
        K = self.base["Strike"]
        r = self.base["RiskFree"] / 100
        div = self.base.get("Dividend", 0.0)
        if div > 0:
            tdiv = (self.base["DivExDate"] - self.base["PricingDate"]).days / 365
            pvdiv = div * math.exp(-r * tdiv)
        else:
            pvdiv = 0.0
        expected = S0 - pvdiv - K * math.exp(-r * ttm)
        return self._row("Put-call parity (C-P)", pc - pp, expected, tol=1e-1)

    def test_american_put_geq_european(self):
        p_eu, _ = self._run(Type="Put", Exercice="European", Dividend=0)
        p_am, _ = self._run(Type="Put", Exercice="American", Dividend=0)
        gap = p_am - p_eu
        return {"Test": "A_put ≥ E_put", "Valeur": gap,
                "Attendu": "≥ 0", "OK": "✅" if gap >= -1e-6 else "❌"}

    def test_american_call_no_div_equal_eu(self):
        p_eu, _ = self._run(Type="Call", Exercice="European", Dividend=0)
        p_am, _ = self._run(Type="Call", Exercice="American", Dividend=0)
        gap = p_am - p_eu
        return {"Test": "A_call = E_call (sans div, r>0)", "Valeur": gap,
                "Attendu": "≈ 0", "OK": "✅" if abs(gap) < 1e-2 else "❌"}

    def test_american_call_with_div_geq(self):
        div = max(self.base.get("Dividend", 0.0), 1.0)
        p_eu, _ = self._run(Type="Call", Exercice="European", Dividend=div)
        p_am, _ = self._run(Type="Call", Exercice="American", Dividend=div)
        gap = p_am - p_eu
        return {"Test": f"A_call ≥ E_call (div={div})", "Valeur": gap,
                "Attendu": "≥ 0", "OK": "✅" if gap >= -1e-6 else "❌"}

    def test_high_r_increases_am_put_premium(self):
        p_am_hi, _ = self._run(Type="Put", Exercice="American", Dividend=0, RiskFree=10.0)
        p_eu_hi, _ = self._run(Type="Put", Exercice="European", Dividend=0, RiskFree=10.0)
        p_am_lo, _ = self._run(Type="Put", Exercice="American", Dividend=0, RiskFree=-1.0)
        p_eu_lo, _ = self._run(Type="Put", Exercice="European", Dividend=0, RiskFree=-1.0)
        gap_hi = p_am_hi - p_eu_hi
        gap_lo = p_am_lo - p_eu_lo
        return {"Test": "Prime A_put plus grande si r grand",
                "Valeur": gap_hi - gap_lo, "Attendu": "> 0",
                "OK": "✅" if gap_hi >= gap_lo - 1e-6 else "❌"}

    def test_negative_r_put(self):
        """Avec r < 0, la prime d'exercice anticipé sur put est faible."""
        p_am, _ = self._run(Type="Put", Exercice="American", Dividend=0, RiskFree=-2.0)
        p_eu, _ = self._run(Type="Put", Exercice="European", Dividend=0, RiskFree=-2.0)
        gap = p_am - p_eu
        return {"Test": "r<0 → prime A_put faible", "Valeur": gap,
                "Attendu": "petit (≥ 0)", "OK": "✅" if 0 <= gap < 0.5 else "❌"}

    def _row(self, name, got, expected, tol):
        diff = abs(got - expected)
        return {"Test": name, "Valeur": got, "Attendu": expected,
                "Écart": diff, "OK": "✅" if diff < tol else "❌"}

    def run_all(self):
        tests = [
            self.test_K0_call,
            self.test_K0_put,
            self.test_european_vs_BS,
            self.test_put_call_parity,
            self.test_american_put_geq_european,
            self.test_american_call_no_div_equal_eu,
            self.test_american_call_with_div_geq,
            self.test_high_r_increases_am_put_premium,
            self.test_negative_r_put,
        ]
        rows = []
        for t in tests:
            try:
                rows.append(t())
            except Exception as e:
                rows.append({"Test": t.__name__, "Valeur": str(e),
                             "Attendu": "-", "OK": "⚠️"})
        return pd.DataFrame(rows)