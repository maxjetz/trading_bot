import numpy as np
import pandas as pd
import random
import logging

class Market:
    def __init__(self):
        """
        Simulerad marknad som tillhandahåller data för flera tillgångar.
        """
        self.active_symbols = ["BTC/USDT", "ETH/USDT", "LTC/USDT"]  # Exempel på symboler

    def fetch_market_data(self, min_volume=10):
        """
        Mocka marknadsdata för aktiva symboler.
        :param min_volume: Minsta volym för att inkludera en symbol.
        :return: DataFrame med marknadsdata.
        """
        data = {
            "symbol": self.active_symbols,
            "volume": [random.uniform(min_volume, min_volume * 10) for _ in self.active_symbols],
            "price": [random.uniform(10, 100) for _ in self.active_symbols],
        }
        df = pd.DataFrame(data)
        df = df[df["volume"] >= min_volume].reset_index(drop=True)  # Filtrera baserat på volym
        return df

    def fetch_historical_data(self, symbol, timeframe="1h", limit=100):
        """
        Mocka historisk data för en given symbol.
        :param symbol: Symbolen för vilken data ska hämtas.
        :param timeframe: Tidsperspektiv för data (t.ex. '1h').
        :param limit: Antal datapunkter att generera.
        :return: DataFrame med rå historisk data.
        """
        if symbol not in self.active_symbols:
            raise ValueError(f"Symbol {symbol} stöds inte av marknaden.")

        # Uppdatera tidsfrekvensen för att undvika framtida varningar
        if timeframe == '1m':  # Om du använder minutintervall, byt till 'T'
            timeframe = 'T'

        timestamp = pd.date_range(start="2023-01-01", periods=limit, freq=timeframe)
        close = np.cumsum(np.random.randn(limit)) + 50  # Simulerat pris
        high = close + np.random.uniform(0.5, 1.5, limit)
        low = close - np.random.uniform(0.5, 1.5, limit)
        volume = np.random.uniform(100, 1000, limit)

        df = pd.DataFrame({
            "timestamp": timestamp,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume
        })

        # Lägg till fler fält för att simulera verklig data
        df["open"] = df["close"].shift(1).fillna(df["close"][0])
        df["volatility"] = (df["high"] - df["low"]) / df["close"]
        df["trend_strength"] = np.random.uniform(0, 1, limit)  # Slumpad trendstyrka

        return df

    def simulate_market_conditions(self, condition="bullish"):
        """
        Simulera marknadsförhållanden och anpassa priser därefter.
        :param condition: Typ av marknadsförhållande ('bullish', 'bearish', 'sideways').
        :return: Modifierad lista över symboler.
        """
        factor_map = {
            "bullish": 1.05,
            "bearish": 0.95,
            "sideways": 1.00
        }

        factor = factor_map.get(condition, 1.00)
        for symbol in self.active_symbols:
            new_data = self.fetch_historical_data(symbol=symbol, limit=100)
            new_data["close"] *= factor  # Anpassa slutpris baserat på förhållande
            logging.info(f"Simulerade {condition} marknadsförhållanden för {symbol}.")

        return self.active_symbols
