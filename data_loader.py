# data_loader.py — FINAL WORKING VERSION (macOS + Python 3.12, Nov 2025)
# No warnings, no Arrow errors, creates prize_dataset.parquet instantly

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

HORIZONS = [5, 10, 20, 40, 80, 160, 250]
TICKERS = ["^GSPC", "^DJI", "^FTSE", "AAPL", "MSFT", "AMZN", "BTC-USD", "GLD"]

Path("cache").mkdir(exist_ok=True)
all_data = []

print("Downloading and processing tickers...")

for ticker in TICKERS:
    print(f"  → {ticker}", end="")
    cache = Path("cache") / f"{ticker}.parquet"
    
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        # Download
        try:
            price = yf.download(ticker, period="max", progress=False, auto_adjust=True)["Close"]
            ret = np.log(price).diff().dropna()
        except:
            print(" [failed]")
            continue
            
        if len(ret) < 1000:
            print(" [too short]")
            continue
            
        rows = []
        scale = np.sqrt(252)
        
        for T in HORIZONS:
            for i in range(0, len(ret)-T+1, T):          # non-overlapping windows
                w = ret.iloc[i:i+T]
                if len(w) < T*0.8: continue
                
                total_ret = w.sum()
                vol = w.std(ddof=0) * scale                 # population std (finance convention)
                z_raw = total_ret / np.sqrt(T/252.0)
                
                rows.append({
                    "ticker": ticker,
                    "date": w.index[0].date(),
                    "T": T,
                    "z_raw": float(z_raw),
                    "sigma": float(vol)
                })
        
        df = pd.DataFrame(rows)
        # de-mean z per ticker & horizon (vectorised, no warnings)
        df["z"] = df.groupby(["ticker","T"])["z_raw"].transform(lambda x: x - x.mean())
        df = df[["ticker","date","T","z","sigma"]].dropna()
        df.to_parquet(cache, compression=None)   # ← crucial: no compression = no Arrow bugs
        print(f" → {len(df):,} points")
    
    all_data.append(df)

# Final dataset
full = pd.concat(all_data, ignore_index=True)
full.to_parquet("prize_dataset.parquet", compression=None)
print(f"\nSUCCESS! prize_dataset.parquet created with {len(full):,} rows")
print("First 5 rows:")
print(full.head())
