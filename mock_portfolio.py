import logging

class MockPortfolio:
    def __init__(self, initial_balance=10000, fee_rate=0.001):
        """
        Mock-version av Portfolio.
        :param initial_balance: Startsaldo för den mockade portföljen.
        :param fee_rate: Avgiftssats för transaktioner.
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee_rate = fee_rate
        self.holdings = {}  # Dict som lagrar innehav per symbol
        self.trading_active = True

    def reset(self):
        """Återställ portföljens tillstånd till det ursprungliga läget."""
        self.balance = self.initial_balance
        self.holdings = {}  # Rensa alla innehav
        self.trading_active = True
        logging.info("Portföljens tillstånd har återställts.")

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
        return new_value / total_portfolio_value <= 0.2  # Exempelgräns på 20%

    def buy(self, symbol, price, quantity):
        """
        Simulera köp av en tillgång.
        :param symbol: Symbol för tillgången.
        :param price: Pris för tillgången.
        :param quantity: Antal enheter att köpa.
        """
        if not self.trading_active:
            raise ValueError("Handel är avstängd.")

        total_cost = price * quantity * (1 + self.fee_rate)

        if total_cost > self.balance:
            raise ValueError("Inte tillräckligt med saldo för att köpa.")

        self.balance -= total_cost
        self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
        logging.info(f"Köpt {quantity} av {symbol} för {price}. Totalkostnad: {total_cost}.")

    def sell(self, symbol, price, quantity):
        """
        Simulera försäljning av en tillgång.
        :param symbol: Symbol för tillgången.
        :param price: Pris för tillgången.
        :param quantity: Antal enheter att sälja.
        """
        if not self.trading_active:
            raise ValueError("Handel är avstängd.")

        if symbol not in self.holdings or self.holdings[symbol] < quantity:
            raise ValueError("Inte tillräckligt med tillgångar för att sälja.")

        total_value = price * quantity * (1 - self.fee_rate)
        self.balance += total_value
        self.holdings[symbol] -= quantity

        if self.holdings[symbol] == 0:
            del self.holdings[symbol]  # Ta bort symbolen om inga innehav finns kvar

        logging.info(f"Sålt {quantity} av {symbol} för {price}. Totalvärde: {total_value}.")

    def log_summary(self):
        """
        Logga en sammanfattning av portföljens status.
        """
        logging.info(f"Mocked Portfolio Balance: {self.balance}")
        logging.info(f"Holdings: {self.holdings}")

    def get_total_value(self):
        """
        Beräkna det totala värdet på portföljen.
        :return: Totalt värde av portföljen, inklusive kontanter och tillgångar.
        """
        total_value = self.balance
        for symbol, quantity in self.holdings.items():
            # För varje innehav, multiplicera med aktuellt pris
            total_value += quantity * self.get_current_price(symbol)
        return total_value

    def get_current_price(self, symbol):
        """
        Hämtar det aktuella priset för en tillgång.
        :param symbol: Symbol för tillgången.
        :return: Aktuellt pris för symbolen.
        """
        # Här kan du ersätta med riktig logik för att hämta aktuellt pris från en API eller annan källa.
        # För nu använder vi ett mockat pris (exempelvis ett fast pris).
        mock_prices = {
            "BTC/USDT": 30000,
            "ETH/USDT": 2000,
            "LTC/USDT": 150
        }
        return mock_prices.get(symbol, 0)  # Returnera 0 om symbolen inte finns
