# trading_rl_pipeline.py
# -------------------------------------------
import os
import json
import warnings

import numpy as np
import pandas as pd
import polars as pl
import gymnasium as gym
from gymnasium import spaces
from datetime import timedelta

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback
# from sb3_contrib.common.maskable.utils import get_action_masks  # 仅用于演示，可不装 sb3-contrib

warnings.filterwarnings("ignore")

# -------- 1. 数据加载与切分 --------------------------------------------------
RAW_PATH = "kline_data/origin_data_1m_5000000_BTC-USDT-SWAP_2025-05-06_with_feature.parquet"  # <—— 请确认文件格式与读取方式一致
# 注意：文件名后缀为 CSV，但这里调用的是 read_parquet，请确保文件格式正确
df_raw = pl.read_parquet(RAW_PATH).to_pandas()

# 必须包含列：
# ['timestamp','open','high','low','close','volume',
#  'log_ret','log_vol','sma20','sma60','sma120','rsi14','atr14','bollw',
#  'vol60','skew60','kurt60','bull','bear','side',
#  'dummy_pos','dummy_pnl','dummy_margin']
df_raw.sort_values("timestamp", inplace=True)
df_raw.reset_index(drop=True, inplace=True)

size = len(df_raw)
train_end = int(0.6 * size)
val_end = int(0.8 * size)
df_train = df_raw.iloc[:train_end].copy()
df_val = df_raw.iloc[train_end:val_end].copy()
df_test = df_raw.iloc[val_end:].copy()

# 计算 μ/σ  (仅计算训练集，用于归一化)
FEAT_COLS = df_raw.columns.drop(["timestamp", "open", "high", "low", "close", "volume"])
mu = df_train[FEAT_COLS].mean()
sig = df_train[FEAT_COLS].std().replace(0, 1)


# -------- 2. Gym 环境实现 -----------------------------------------------------
class DiscreteTradingEnv(gym.Env):
    """
    三挡仓位 (Flat/Long/Short)；含：
      - 0.10% 交易成本
      - 回撤风险、平滑惩罚
      - 交易间隔惩罚、持仓时长惩罚
      - 强制 4h 平仓

    在此版本中，我们去除了 cash 变量，直接使用 NAV 进行更新，
    交易成本直接从 NAV 中扣除。
    """
    metadata = {"render_modes": []}
    ACTION_DICT = {0: 'Flat', 1: 'Long', 2: 'Short'}

    def __init__(
            self,
            df: pd.DataFrame,
            feature_cols=FEAT_COLS,
            window: int = 120,
            cost_rate: float = 0.001,
            λ: float = 1.5,
            γ: float = 0.002,
            η: float = 0.30,
            ζ: float = 0.0005,
            T0: int = 45,
            T_max: int = 240,
            target_gap: int = 30,
            seed: int = 0
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.fcols = feature_cols
        self.w = window
        self.cost_rate = cost_rate
        self.lam = λ
        self.gam = γ
        self.eta = η
        self.zeta = ζ
        self.T0 = T0
        self.Tmax = T_max
        self.target_gap = target_gap
        self.rng = np.random.default_rng(seed)

        # 定义 observation 与 action 的空间
        self.observation_space = spaces.Box(-5, 5, (len(self.fcols), self.w), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: Flat, 1: Long, 2: Short

        self._ptr = None
        self._reset_inner()

    def _make_obs(self):
        slice_ = self.df.iloc[self._ptr - self.w + 1:self._ptr + 1][self.fcols]
        obs = ((slice_ - mu) / sig).clip(-5, 5).values.T.astype(np.float32)
        return obs

    def _reset_inner(self):
        self.pos = 0        # (-1, 0, +1) 表示空头、空仓、多头
        self.nav = 1_000.0  # 初始净值
        self.last_trade_step = 0
        self.age = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._ptr = self.rng.integers(self.w, len(self.df) - 1)
        self._reset_inner()
        obs = self._make_obs()
        info = {"nav": self.nav}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        done = False
        trunc = False

        # 1. 交易逻辑：判断是否需要改变仓位
        prev_pos = self.pos
        if action == 1:
            desired = 1
        elif action == 2:
            desired = -1
        else:
            desired = 0

        trade_happened = False
        if prev_pos != desired:
            if prev_pos != 0:
                # 平仓：扣除手续费后平仓
                trade_happened = True
                cost = self.cost_rate * self.nav
                self.nav -= cost
                self.pos = 0
            if desired != 0:
                # 开仓：扣除手续费后建仓
                trade_happened = True
                cost = self.cost_rate * self.nav
                self.nav -= cost
                self.pos = desired

        # 2. 市值更新（仅当持仓时考虑价格变动）
        price_ret = self.df.close.iloc[self._ptr + 1] / self.df.close.iloc[self._ptr] - 1
        if self.pos != 0:
            self.nav = self.nav * (1 + price_ret)
        # Flat 状态下 NAV 不变

        # 3. 计算奖励，只有刚发生交易时才计算 pnl，否则为 0
        if self.last_trade_step == 0:
            raw_pnl_pct = (self.nav - 1_000.0) / 1_000.0
        else:
            raw_pnl_pct = 0.0

        risk_pen = self.lam * max(0, -raw_pnl_pct)
        smooth_pen = self.eta * abs(raw_pnl_pct)

        gap_pen = 0
        if trade_happened:
            gap_t = self.last_trade_step
            gap_pen = self.gam * abs(gap_t - self.target_gap) / self.target_gap
            self.last_trade_step = 0
            self.age = 0
        else:
            self.last_trade_step += 1
            if self.pos != 0:
                self.age += 1

        dur_pen = 0
        if self.pos != 0 and self.age > self.T0:
            dur_pen = self.zeta * (self.age - self.T0) / self.T0

        reward = raw_pnl_pct - risk_pen - smooth_pen - gap_pen - dur_pen

        # 4. 强制平仓：若持仓时间超过 Tmax，则扣除手续费并平仓
        if self.pos != 0 and self.age >= self.Tmax:
            cost = self.cost_rate * self.nav
            self.nav -= cost
            self.pos = 0
            done = True

        # 5. 检查是否破产或数据用尽
        if self.nav < 0.2 * 1_000 or self._ptr >= len(self.df) - 2:
            done = True
        self._ptr += 1

        obs = self._make_obs()
        info = {"nav": self.nav, "pos": self.pos}
        return obs, reward, done, trunc, info


# -------- 3. 封装多进程 VecEnv ----------------------------------------------
def make_env(df_slice, seed):
    def _init():
        env = DiscreteTradingEnv(df_slice, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _init


# -------- 5. Walk-Forward & Bootstrap ----------------------------------------
def run_episode(env, model):
    """
    由于我们的 env 是向量化环境（即 DummyVecEnv），
    reset() 与 step() 返回的 observation 是数组，info 是列表，
    因此需取第一个环境的信息。
    """
    obs, _ = env.reset()
    done = [False]
    nav = []
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        nav.append(info[0]["nav"])
    return pd.Series(nav, index=range(len(nav)))


def wf_eval(df_full, model, train_vec):
    seg = timedelta(days=730)  # 2年训练
    val = timedelta(days=180)  # 6月验证
    test = timedelta(days=180)  # 6月测试
    ts = pd.to_datetime(df_full.timestamp, unit='s')
    curves = []
    start = ts.min()
    while start + seg + val + test < ts.max():
        # 仅使用测试集数据进行回测
        test_idx = (ts >= start + seg + val) & (ts < start + seg + val + test)
        single_env = DiscreteTradingEnv(df_full[test_idx])
        # 使用 DummyVecEnv 包装，使其向量化
        env = DummyVecEnv([lambda: single_env])
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        env.obs_rms = train_vec.obs_rms
        nav = run_episode(env, model)
        curves.append(nav)
        start += val
    return curves


# -------- 主程序 -------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # -------- 4. 训练 -------------------------------------------------------------
    N_ENVS = 1
    train_vec = SubprocVecEnv([make_env(df_train, i) for i in range(N_ENVS)])
    train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=False)

    # 构建验证环境时，用 DummyVecEnv 包装单一环境
    val_env_single = DiscreteTradingEnv(df_val)
    val_env = DummyVecEnv([lambda: val_env_single])
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False)
    val_env.obs_rms = train_vec.obs_rms  # 验证时直接使用训练时的归一化数据

    model = PPO(
        "MlpPolicy",
        train_vec,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=4096,
        gamma=0.99,
        gae_lambda=0.9,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[64, 128, 64]),
        verbose=1
    )

    eval_cb = EvalCallback(
        val_env,
        best_model_save_path="./best",
        log_path="./logs",
        eval_freq=10_000,
        deterministic=True,
        render=False,
        callback_on_new_best=StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=3,
            min_evals=5,
            verbose=1
        )
    )
    model.learn(1_000_000, callback=eval_cb)
    model.save("artifacts/model.zip")
    train_vec.save("artifacts/vec_norm.pkl")

    # 回测部分：Walk-Forward & Bootstrap
    curves = wf_eval(df_raw, model, train_vec)
    os.makedirs("artifacts/nav_curves", exist_ok=True)
    metrics = []
    for i, nav in enumerate(curves, 1):
        ret = nav.pct_change().dropna()
        sharpe = ret.mean() / ret.std() * np.sqrt(525_600)  # 以分钟频率年化
        mdd = (nav / nav.cummax() - 1).min()
        metrics.append(dict(wf=i, sharpe=sharpe, maxdd=mdd))
        nav.to_parquet(f"artifacts/nav_curves/wf{i}.parquet")
    json.dump(metrics, open("artifacts/metrics.json", "w"), indent=2)
    print("WF metrics:", metrics)