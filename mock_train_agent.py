import sys
import os

# Lägg till projektets rotmapp till sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

from config import Config
from agent import MultiAssetTradingEnv
from mock_market import MockMarket
from mock_portfolio import MockPortfolio

if __name__ == "__main__":
    # Konfigurera logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Ladda konfiguration
    config = Config()
    config.validate_config()
    training_config = config.get("training")
    env_config = config.get("environment")

    # Initiera mockad marknad och portfölj
    market = MockMarket()
    portfolio = MockPortfolio(initial_balance=env_config.get("initial_balance", 10000))

    # Hämta mockad marknadsdata och historisk data
    market_data = market.fetch_market_data(min_volume=env_config["min_volume"])
    symbols = market_data["symbol"].tolist()
    logging.info(f"Mockade aktiva symboler: {symbols}")

    historical_data = {}
    for symbol in symbols:
        historical_data[symbol] = market.fetch_historical_data(
            symbol=symbol,
            timeframe=env_config["timeframe"],
            limit=env_config["data_limit"]
        )

    # Skapa miljö
    def create_env():
        return MultiAssetTradingEnv(market=market, portfolio=portfolio, historical_data=historical_data)

    # Kontrollera miljöns kompatibilitet
    try:
        check_env(create_env(), warn=True)
        logging.info("Miljön är kompatibel.")
    except Exception as e:
        logging.error(f"Miljön är inte kompatibel: {e}")
        sys.exit(1)

    # Skapa en SubprocVecEnv för parallellisering
    def make_env():
        return MultiAssetTradingEnv(market=market, portfolio=portfolio, historical_data=historical_data)

    env = SubprocVecEnv([make_env for _ in range(training_config.get("num_envs", 4))])

    # Exempelträning med PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=training_config["learning_rate"],
        gamma=training_config["gamma"],
        gae_lambda=training_config["gae_lambda"],
        ent_coef=training_config["ent_coef"],
        verbose=1
    )

    logging.info("Startar mockad träning...")
    try:
        model.learn(total_timesteps=training_config["total_timesteps"])
    except Exception as e:
        logging.error(f"Fel under träning: {e}")
        sys.exit(1)

    # Spara mockad modell
    model_path = "mocked_trading_agent.zip"
    model.save(model_path)
    logging.info(f"Mockad tränad modell sparad som {model_path}")