import os
import json
import logging
import numpy as np
import pandas as pd
from gymnasium import spaces, Env
from typing import Dict
import ta  # För tekniska indikatorer

# Uppdaterade importer baserat på indikatorfiler
from indicators.momentum_indicators import MomentumIndicators
from indicators.trend_indicators import TrendIndicators
from indicators.volatility_indicators import VolatilityIndicators
from indicators.volume_indicators import VolumeIndicators

from market import Market
from mock_portfolio import MockPortfolio
from config import Config

class MultiAssetTradingEnv(Env):
    def __init__(self, market: Market, portfolio: MockPortfolio, historical_data: Dict[str, pd.DataFrame]):
        super(MultiAssetTradingEnv, self).__init__()

        # Ladda konfigurationer
        config = Config()
        config.validate_config()
        env_config = config.get("environment")

        self.market = market
        self.portfolio = portfolio
        self.historical_data = self._add_indicators(historical_data)
        self.assets = list(historical_data.keys())
        self.num_assets = len(self.assets)
        self.fee = env_config["fee"]
        self.current_step = 0
        self.portfolio.previous_value = self.portfolio.get_total_value(
            {asset: self.historical_data[asset].iloc[0]['close'] for asset in self.assets}
        )

        # Dynamiskt sätt observation_space baserat på data
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_assets * (self.historical_data[self.assets[0]].shape[1] - 1),),
            dtype=np.float32
        )

        # Action space: One action per asset (buy, hold, sell)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_assets,), dtype=np.float32)

        # Sammanfattad loggning
        logging.info(f"Miljön initialiserad med {self.num_assets} tillgångar och observation_space {self.observation_space.shape}")
        
    def _add_indicators(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Lägg till tekniska indikatorer till historiska data. Robust felhantering tillagd.
        Flyttad batch-bearbetning av indikatorer till respektive moduler.
        """
        for asset, data in historical_data.items():
            try:
                data = pd.DataFrame(data)

                # Batch-bearbeta indikatorer per tillgång
                data = MomentumIndicators.add_all_indicators(data)
                data = TrendIndicators.add_all_indicators(data)
                data = VolatilityIndicators.add_all_indicators(data)
                data = VolumeIndicators.add_all_indicators(data)

                historical_data[asset] = data

            except Exception as e:
                logging.warning(f"Kunde inte lägga till indikatorer för {asset}: {e}")

        return historical_data
    
    def reset(self, seed=None, options=None):
        """Återställ miljön till startläget."""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.portfolio.reset()  # Återställ portföljen
        obs = self._get_observation()

        logging.info("Miljön återställd till initialt tillstånd.")
        return obs, {}

    def step(self, action):
        """Utför en action och gå till nästa steg."""
        self.current_step += 1
        action_summary = []  # För summerad loggning

        for i, asset in enumerate(self.assets):
            try:
                price = self.historical_data[asset].iloc[self.current_step]['close']
                trend_strength = self.historical_data[asset].iloc[self.current_step].get('ADX', 0)

                # Justera growth limit baserat på trend
                self.portfolio.adjust_growth_limit(trend_strength)

                if action[i] > 0:  # Köp
                    quantity = action[i] * self.portfolio.balance / price
                    self.portfolio.buy(asset, price, quantity)
                    action_summary.append(f"Köp: {asset} Mängd: {quantity:.4f} Pris: {price:.2f}")
                elif action[i] < 0:  # Sälj
                    quantity = -action[i] * self.portfolio.holdings.get(asset, 0)
                    self.portfolio.sell(asset, price, quantity)
                    action_summary.append(f"Sälj: {asset} Mängd: {quantity:.4f} Pris: {price:.2f}")

            except ValueError as e:
                logging.warning(f"Åtgärd misslyckades för {asset}: {e}")
            except KeyError:
                logging.warning(f"Data saknas för {asset} vid steg {self.current_step}.")

        reward = self._calculate_reward(action)
        done = self.current_step >= len(self.historical_data[self.assets[0]]) - 1
        truncated = False  # gymnasium kräver detta för kompatibilitet
        obs = self._get_observation()

        # Sammanfattad loggning
        logging.info(f"Steg {self.current_step}: Reward={reward:.4f}. Sammanfattning: {', '.join(action_summary)}")

        return obs, reward, done, truncated, {}
    
    def _get_observation(self):
        """Hämta nuvarande observation och normalisera."""
        try:
            # Hämta observationerna för alla tillgångar och normalisera dem
            obs = np.concatenate([
                self._normalize(self.historical_data[asset].iloc[self.current_step, 1:].to_numpy())
                for asset in self.assets
            ]).astype(np.float32)

            # Säkerställ att inga NaN eller inf-värden finns i observationen
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        except Exception as e:
            logging.error(f"Fel vid skapande av observation: {e}")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs

    def _normalize(self, data):
        """Normalisera data mellan -1 och 1."""
        min_data = np.min(data)
        max_data = np.max(data)

        # Förhindra att vi delar med noll om min_data == max_data
        if max_data == min_data:
            return np.zeros_like(data)

        # Normalisera mellan -1 och 1
        return 2 * (data - min_data) / (max_data - min_data) - 1

    def _calculate_reward(self, action):
        """Beräkna belöning baserat på action."""
        try:
            current_prices = {asset: self.historical_data[asset].iloc[self.current_step]['close'] for asset in self.assets}
            current_value = self.portfolio.get_total_value(current_prices)
            reward = (current_value - self.portfolio.previous_value) / max(self.portfolio.previous_value, 1e-8)

            # Justera riskgränser dynamiskt
            self.portfolio.adjust_risk_limit(reward)
            self.portfolio.previous_value = current_value
        except Exception as e:
            logging.error(f"Fel vid beräkning av belöning: {e}")
            reward = -1  # Straffa vid fel

        return reward
