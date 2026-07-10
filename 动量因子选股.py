import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt

# =========================
# 1. 数据获取
# =========================
ts.set_token("936c4938d77bac8afb762c4af6e563ae5a24bd6041ff2de42712a67d")
pro = ts.pro_api()

stocks = ["000001.SZ", "000002.SZ", "000858.SZ"]

def get_data(stocks):
    all_data = []
    for code in stocks:
        df = pro.daily(
            ts_code=code,
            start_date="20230101",
            end_date="20231231"
        )
        df["ts_code"] = code
        all_data.append(df)

    data = pd.concat(all_data)
    data = data.sort_values(["ts_code", "trade_date"])
    return data


df = get_data(stocks)


# =========================
# 2. 基础收益率
# =========================
df["ret"] = df.groupby("ts_code")["close"].pct_change()


# =========================
# 3. 因子（20日动量）
# =========================
df["momentum"] = df.groupby("ts_code")["close"].transform(
    lambda x: x / x.shift(20) - 1
)

# 防未来函数
df["momentum"] = df.groupby("ts_code")["momentum"].shift(1)


# =========================
# 4. 选股信号（Top 20%）
# =========================
df["signal"] = 0

df.loc[
    df.groupby("trade_date")["momentum"].rank(pct=True) > 0.8,
    "signal"
] = 1

# 再 shift 防未来函数
df["position"] = df.groupby("ts_code")["signal"].shift(1)


# =========================
# 5. 策略收益（等权组合）
# =========================
df["strategy_ret"] = df["position"] * df["ret"]

daily_ret = df.groupby("trade_date")["strategy_ret"].mean()
cum_ret = (1 + daily_ret).cumprod()


# =========================
# 6. 基准收益（买入持有）
# =========================
bench = df.groupby("trade_date")["ret"].mean()
bench_cum = (1 + bench).cumprod()


# =========================
# 7. 交易成本（关键加分项）
# =========================
cost = 0.001  # 0.1%
df["trade"] = df.groupby("ts_code")["position"].diff().abs()
df["strategy_ret_tc"] = df["strategy_ret"] - df["trade"] * cost

daily_ret_tc = df.groupby("trade_date")["strategy_ret_tc"].mean()
cum_ret_tc = (1 + daily_ret_tc).cumprod()


# =========================
# 8. 风险指标函数（面试重点）
# =========================
def performance(series):
    ret = series.pct_change().dropna()

    sharpe = np.sqrt(252) * ret.mean() / ret.std()

    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min()

    return sharpe, mdd


sharpe, mdd = performance(cum_ret_tc)

print("Sharpe:", sharpe)
print("Max Drawdown:", mdd)


# =========================
# 9. 可视化
# =========================
plt.figure()
plt.plot(cum_ret, label="Strategy")
plt.plot(cum_ret_tc, label="Strategy (cost)")
plt.plot(bench_cum, label="Benchmark")
plt.legend()
plt.title("Momentum Strategy Backtest")
plt.show()