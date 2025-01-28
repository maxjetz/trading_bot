import logging
import os
import json
from dotenv import load_dotenv

class Config:
    def __init__(self, env_file="Nycklar.env"):
        """
        Initiera Config-klassen som laddar miljövariabler och konfigurationsinställningar dynamiskt.
        :param env_file: Sökvägen till .env-filen med API-nycklar.
        """
        load_dotenv(env_file)  # Ladda miljövariabler från .env-filen

        # Ladda rätt config-fil baserat på miljövariabel
        config_mode = os.getenv("CONFIG_MODE", "live")  # Standard: live
        self.config_file = f"config_{config_mode}.json" if config_mode != "live" else "config.json"
        self.config = self._load_config()

    def _load_config(self):
        """
        Ladda konfigurationer från en JSON-fil.
        :return: Ordbok med konfigurationsinställningar.
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading config file {self.config_file}: {e}")
        else:
            logging.warning(f"Config file {self.config_file} not found. Using default values.")
        return {}

    def get(self, section, key=None):
        """
        Hämta ett specifikt värde från konfigurationen.
        :param section: Sektionen i konfigurationen (t.ex. "binance").
        :param key: Nyckeln i sektionen (t.ex. "api_key"). Om None returneras hela sektionen.
        :return: Det efterfrågade värdet eller None om det inte finns.
        """
        section_data = self.config.get(section, {})
        if key:
            return section_data.get(key)
        return section_data

    def validate_config(self):
        """
        Kontrollera att viktiga inställningar finns, t.ex. API-nycklar och träningsparametrar.
        :raises ValueError: Om nödvändiga inställningar saknas.
        """
        required_sections = {
            "binance": ["api_key", "api_secret"],
            "training": ["learning_rate", "gamma", "gae_lambda", "ent_coef", "total_timesteps", "checkpoint_interval"],
            "environment": ["min_volume", "fee", "timeframe", "data_limit", "risk_limit", "growth_limit"],
            "market": ["min_volume"]
        }

        for section, keys in required_sections.items():
            section_data = self.get(section)
            if not section_data:
                raise ValueError(f"Missing section '{section}' in config.")

            for key in keys:
                if key not in section_data:
                    raise ValueError(f"Missing key '{key}' in section '{section}' of config.")

        # Kontrollera att API-nycklar är laddade från .env
        if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_SECRET_KEY"):
            raise ValueError("API-nycklar för Binance saknas. Kontrollera din .env-fil.")
        logging.info(f"Config validation passed for {self.config_file}")

# Exempelanvändning
if __name__ == "__main__":
    config = Config()

    # Validera konfigurationen
    config.validate_config()

    # Hämta miljöinställningar
    environment_config = config.get("environment")
    print(f"Miljöinställningar: {environment_config}")
