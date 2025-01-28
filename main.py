from config import Config
from market import Market
from portfolio import Portfolio
from agent import MultiAssetTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

if __name__ == "__main__":
    # Initiera loggning
    config = Config()
    config.validate_config()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Hämta konfigurationsinställningar
    binance_config = config.get("binance")
    training_config = config.get("training")
    env_config = config.get("environment")
    portfolio_config = config.get("portfolio")

    # Initiera Market och Portfolio
    market = Market(api_key=binance_config["api_key"], secret_key=binance_config["api_secret"])
    portfolio = Portfolio(api_key=binance_config["api_key"], secret_key=binance_config["api_secret"])

    # Dynamiskt hämta symboler baserat på volym
    symbols = market.get_active_symbols(min_volume=env_config["min_volume"])
    if not symbols:
        raise ValueError("Inga aktiva symboler hittades med tillräcklig volym.")

    # Hämta historisk data
    historical_data = {}
    for symbol in symbols:
        historical_data[symbol] = market.fetch_historical_data(
            symbol=symbol,
            timeframe=env_config["timeframe"],
            limit=env_config["data_limit"]
        )

    # Skapa miljö
    env = MultiAssetTradingEnv(
        market=market,
        portfolio=portfolio,
        historical_data=historical_data
    )

    # Använd DummyVecEnv för att möjliggöra parallellisering
    env = DummyVecEnv([lambda: env])

    # Initiera PPO-agenten
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=training_config["learning_rate"],
        gamma=training_config["gamma"],
        gae_lambda=training_config["gae_lambda"],
        ent_coef=training_config["ent_coef"],
        verbose=1
    )

    # Träna agenten
    logging.info("Startar träning av agenten...")
    model.learn(total_timesteps=training_config["total_timesteps"])

    # Spara modellen
    model.save("trained_agent.zip")
    logging.info("Agenten har sparats som trained_agent.zip")
