class OptionParameters:
    def __init__(self, input: dict):
        self.strike = input["Strike"]
        self.type_contrat = input["Type"]
        self.Exercice = input["Exercice"]
        self.DateMaturity = input["Maturity"]
        self.date_dividende = input["DivExDate"]

    def Vi(self, S: float) -> float:
        if self.type_contrat == "Call":
            return max(S - self.strike, 0)
        elif self.type_contrat == "Put":
            return max(self.strike - S, 0)
        raise ValueError("Type de contrat non valide.")