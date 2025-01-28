from market import Market
import pandas as pd
import numpy as np
import logging

class MockMarket(Market):
    def __init__(self, *args, **kwargs):
        """
        Mock-version av Market som ärver funktionalitet från den riktiga Market-klassen.
        :param args: Argument till Market.
        :param kwargs: Nyckelord till Market.
        """
        super().__init__(*args, **kwargs)

    def fetch_market_data(self, min_volume=10):
        """
        Mockad version av fetch_market_data som kan innehålla specifik logik för mock-scenarion.
        :param min_volume: Minsta volym för att inkludera en symbol.
        :return: DataFrame med mockad marknadsdata.
        """
        data = super().fetch_market_data(min_volume=min_volume)
        # Extra logik för mockad data
        data["mock_flag"] = True  # Lägg till en flagga för att indikera att det är mockad data
        return data

    def fetch_historical_data(self, symbol, timeframe="1h", limit=100):
        """
        Mockad version av fetch_historical_data.
        :param symbol: Symbolen för vilken data ska hämtas.
        :param timeframe: Tidsperspektiv för data (t.ex. '1h').
        :param limit: Antal datapunkter att generera.
        :return: DataFrame med mockad historisk data.
        """
        data = super().fetch_historical_data(symbol, timeframe=timeframe, limit=limit)
        # Lägg till mock-specifika kolumner
        data["mock_flag"] = True
        data["noise"] = np.random.normal(0, 0.02, size=len(data))  # Simulerat brus
        data["adjusted_close"] = data["close"] * (1 + data["noise"])  # Justerat slutpris
        return data

    def simulate_extreme_conditions(self, symbol, condition="crash", severity=0.5):
        """
        Simulera extrema marknadssituationer som krascher eller bubblor.
        :param symbol: Symbolen som påverkas.
        :param condition: Typ av situation ('crash', 'bubble').
        :param severity: Hur stark effekten ska vara (standard 0.5).
        :return: Modifierad historisk data.
        """
        if symbol not in self.active_symbols:
            raise ValueError(f"Symbol {symbol} stöds inte av marknaden.")

        data = self.fetch_historical_data(symbol)
        if condition == "crash":
            data["close"] *= (1 - severity)
            logging.info(f"Simulerade en krasch för {symbol} med en allvarlighetsgrad på {severity}.")
        elif condition == "bubble":
            data["close"] *= (1 + severity)
            logging.info(f"Simulerade en bubbla för {symbol} med en allvarlighetsgrad på {severity}.")
        else:
            logging.warning(f"Okänt marknadsvillkor: {condition}")

        return data

# Exempelanvändning
if __name__ == "__main__":
    mock_market = MockMarket()

    # Hämta mockad marknadsdata
    market_data = mock_market.fetch_market_data(min_volume=1000)
    print("Mockad marknadsdata:")
    print(market_data)

    # Hämta mockad historisk data
    historical_data = mock_market.fetch_historical_data(symbol="BTC/USDT", timeframe="1h", limit=10)
    print("Mockad historisk data:")
    print(historical_data.head())

    # Simulera extrema förhållanden
    extreme_data = mock_market.simulate_extreme_conditions(symbol="BTC/USDT", condition="crash", severity=0.3)
    print("Simulerad kraschdata:")
    print(extreme_data.head())