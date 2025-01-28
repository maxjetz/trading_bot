import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict
from market import Market
from config import Config
import ta  # För tekniska indikatorer

# Förbättrade funktioner för riskhantering, slippage och dynamisk portföljhantering
class EnhancedPortfolio:
    def __init__(self, market: Market, historical_data: Dict[str, pd.DataFrame]):
        self.market = market
        self.historical_data = historical_data
        self.balance = 10000  # Startsaldo
        self.fee_rate = 0.001  # Transaktionsavgift
        self.holdings = {}
        self.risk_limit = 0.05  # Standard riskgräns på 5%
        self.growth_limit = 0.2  # Standard tillväxtgräns på 20%
        self.max_balance = self.balance  # För spårning av toppvärde
        self.performance = 0  # För att spåra prestation
        self.trading_active = True

    def adjust_risk_limit(self, performance: float, volatility: float) -> None:
        """
        Justera riskgränsen dynamiskt baserat på prestation och volatilitet.
        :param performance: Agentens historiska prestation (positiv eller negativ avkastning).
        :param volatility: Marknadens volatilitet.
        """
        if performance > 0:
            self.risk_limit = min(0.1, self.risk_limit * 1.1)  # Öka riskgränsen till max 10%
        else:
            self.risk_limit = max(0.01, self.risk_limit * 0.9)  # Minska riskgränsen till minst 1%

        # Justera riskgränsen baserat på volatilitet
        if volatility > 0.02:
            self.risk_limit = max(0.05, self.risk_limit * 0.9)  # Öka riskhantering vid hög volatilitet
        else:
            self.risk_limit = min(0.1, self.risk_limit * 1.05)  # Minska riskhantering vid låg volatilitet

    def adjust_growth_limit(self, trend_strength: float) -> None:
        """
        Justera tillväxtgränsen dynamiskt baserat på trendstyrka.
        :param trend_strength: Ett värde mellan 0 och 1 som representerar hur stark trenden är.
        """
        self.growth_limit = min(0.8, 0.3 + 0.5 * trend_strength)  # Max 80% allokering vid stark trend

    def apply_slippage(self, price: float, slippage_factor: float = 0.01) -> float:
        """
        Tillämpa slippage för att göra marknadspris mer realistiskt.
        :param price: Det aktuella marknadspriset.
        :param slippage_factor: Justering för slippage (default 1%).
        :return: Det justerade priset.
        """
        return price * (1 + np.random.uniform(-slippage_factor, slippage_factor))

    def calculate_risk(self, price, quantity):
        """
        Beräkna risk baserat på investeringens storlek relativt balans.
        :param price: Pris för tillgången.
        :param quantity: Antal enheter att köpa.
        :return: Risk som en procentandel av balansen.
        """
        total_investment = price * quantity
        return total_investment / self.balance

    def can_invest_more(self, symbol, price, quantity):
        """
        Kontrollera om investeringen överskrider en viss tillgångsandel.
        :param symbol: Symbol för tillgången.
        :param price: Pris för tillgången.
        :param quantity: Antal enheter att köpa.
        :return: True om investeringen är inom tillväxtgränsen, annars False.
        """
        current_value = self.holdings.get(symbol, 0) * price
        new_value = current_value + price * quantity
        total_portfolio_value = self.balance + sum(
            qty * price for qty, price in self.holdings.items()
        )
        return new_value / total_portfolio_value <= self.growth_limit

    def buy(self, symbol: str, price: float, quantity: float) -> None:
        """Köp en tillgång med dynamisk slippage och riskkontroll."""
        price = self.apply_slippage(price)  # Tillämpa slippage
        total_cost = price * quantity * (1 + self.fee_rate)  # Inkludera avgift
        risk = self.calculate_risk(price, quantity)

        # Kontrollera om risk överskrider gränsen
        if risk > self.risk_limit:
            raise ValueError(f"Risk för handeln ({risk:.2%}) överskrider tillåten riskgräns ({self.risk_limit:.2%}).")

        # Kontrollera om investeringen överskrider tillväxtgränsen
        if not self.can_invest_more(symbol, price, quantity):
            raise ValueError(f"Investeringen överskrider tillåten tillgångsandel ({self.growth_limit:.2%}).")

        # Kontrollera om det finns tillräckligt med saldo
        if total_cost > self.balance:
            raise ValueError("Inte tillräckligt med saldo för att köpa.")

        self.balance -= total_cost
        self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity

    def sell(self, symbol: str, price: float, quantity: float) -> None:
        """Sälj en tillgång med dynamisk slippage och riskkontroll."""
        if symbol not in self.holdings or self.holdings[symbol] < quantity:
            raise ValueError("Inte tillräckligt med tillgångar för att sälja.")

        price = self.apply_slippage(price)  # Tillämpa slippage
        total_value = price * quantity
        self.balance += total_value * (1 - self.fee_rate)
        self.holdings[symbol] -= quantity

    def rebalance_portfolio(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Omallokera portföljen baserat på marknadsdata och riskhantering."""
        for asset, data in market_data.items():
            volatility = data["volatility"].iloc[-1]  # Senaste volatiliteten
            trend_strength = data["trend_strength"].iloc[-1]  # Senaste trendstyrkan
            self.adjust_risk_limit(self.performance, volatility)
            self.adjust_growth_limit(trend_strength)
            self.balance *= (1 - self.growth_limit)

    def optimize_portfolio(self, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Optimera portföljens tillgångar baserat på moderna portföljteorier.
        :param returns: DataFrame med historiska avkastningar för varje tillgång.
        :param cov_matrix: Kovariansmatris för tillgångarna.
        :return: En serie som representerar optimal allokering.
        """
        inverse_cov = np.linalg.inv(cov_matrix)
        weights = np.dot(inverse_cov, returns.mean())
        weights /= np.sum(weights)  # Normalisera vikterna så att de summerar till 1
        return pd.Series(weights, index=returns.columns)

    def log_summary(self) -> None:
        """
        Logga en sammanfattning av portföljens status.
        """
        logging.info(f"Portfolio balance: {self.balance}")
        logging.info(f"Risk limit: {self.risk_limit}")
        logging.info(f"Growth limit: {self.growth_limit}")
        logging.info(f"Holdings: {json.dumps(self.holdings)}")

    def stop_trading(self) -> None:
        """
        Stopp handel om drawdown överskrider tröskeln eller om andra riskfaktorer uppstår.
        """
        logging.warning("Trading has been stopped due to excessive drawdown or risk factors.")
        self.trading_active = False
