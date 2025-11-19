# data_loader.py — exact replication of Wilmott q-variance methodology
# No installation needed — just run once to create the dataset

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

HORIZONS_DAYS = [5, 10, 20, 40, 80, 160, 250]

def download_ticker(ticker):
    try:
        data = yf.download(ticker, start="1950-01-01", progress=False, auto_adjust=True)
        return np.log(data['Close']).diff().dropna()
    except:
        return pd.Series()

def compute_qvar_for_ticker(log_returns, ticker):
    results = []
    factor = np.sqrt(252)
    for T in HORIZONS_DAYS:
        starts = log_returns.index[::T]
        for s in starts:
            e = s + pd.Timedelta(days=T)
            window = log_returns.loc[s:e]
            if len(window) < T*0.8: continue
            x = window.sum()
            sigma = window.std() * factor
            z_raw = x / np.sqrt(T/252.0)
            results.append({'ticker':ticker,'date':s.date(),'T':T,'z_raw':z_raw,'sigma':sigma})
    df = pd.DataFrame(results)
    if df.empty: return df
    for T in HORIZONS_DAYS:
        mask = df['T']==T
        df.loc[mask,'z'] = df.loc[mask,'z_raw'] - df.loc[mask,'z_raw'].mean()
    return df[['ticker','date','T','z','sigma']]

TICKERS = ["^GSPC","^DJI","^FTSE","AAPL","MSFT","AMZN","BTC-USD","GLD"]

Path("cache").mkdir(exist_ok=True)
all_dfs = []
for t in TICKERS:
    cache = Path("cache")/f"{t}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        rets = download_ticker(t)
        if len(rets)>1000:
            df = compute_qvar_for_ticker(rets, t)
            df.to_parquet(cache)
    if not df.empty:
        all_dfs.append(df)
        print(f"{t}: {len(df):,} points")

full = pd.concat(all_dfs, ignore_index=True)
full.to_parquet("prize_dataset.parquet", compression='zstd')
print(f"\nPrize dataset ready — {len(full):,} rows")
