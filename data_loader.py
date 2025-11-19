# data_loader.py — Exact replication of Wilmott q-variance methodology (July 2025)
# Bulletproof for Python 3.12: No FutureWarnings, handles sparse data, generates valid rows
# Run once: python data_loader.py → prize_dataset.parquet (~70K rows for 8 tickers)

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date  # For date() if needed

# Horizons exactly as in the article (p.36)
HORIZONS_DAYS = [5, 10, 20, 40, 80, 160, 250]

def download_ticker(ticker: str, start="1950-01-01", end="2025-11-19") -> pd.Series:
    """Download adjusted close and return log returns (handles empty data gracefully)"""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            print(f"Warning: No data for {ticker}")
            return pd.Series(dtype=float)
        prices = data['Close']
        log_returns = np.log(prices).diff().dropna()
        log_returns.name = 'log_return'
        return log_returns
    except Exception as e:
        print(f"Warning: Failed to download {ticker}: {e}")
        return pd.Series(dtype=float)

def compute_qvar_for_ticker(log_returns: pd.Series, ticker: str) -> pd.DataFrame:
    """Compute z and sigma for all horizons T (non-overlapping windows, Wilmott p.36)"""
    if len(log_returns) < min(HORIZONS_DAYS):
        print(f"Skipping {ticker}: Insufficient data ({len(log_returns)} < {min(HORIZONS_DAYS)})")
        return pd.DataFrame()
    
    results = []
    annual_factor = np.sqrt(252)
    
    for T_days in HORIZONS_DAYS:
        # Non-overlapping: Step by T_days using iloc for index safety
        step = T_days
        valid_windows = 0
        for i in range(0, len(log_returns) - T_days + 1, step):
            window = log_returns.iloc[i:i + T_days]
            if len(window) < T_days * 0.8:  # Tolerate minor gaps
                continue
            valid_windows += 1
            
            # Total log return x over window
            x = window.sum()
            
            # Annualized sigma over same window
            sigma = window.std() * annual_factor if len(window) > 1 else np.nan
            
            # Raw z (scaled, before global de-mean)
            z_raw = x / np.sqrt(T_days / 252.0) if T_days > 0 else np.nan
            
            results.append({
                'ticker': ticker,
                'date': window.index[0].date(),
                'T': T_days,
                'x': float(x),  # Force float early
                'z_raw': float(z_raw) if not np.isnan(z_raw) else np.nan,
                'sigma': float(sigma) if not np.isnan(sigma) else np.nan
            })
        print(f"   T={T_days}: {valid_windows} valid windows")
    
    if not results:
        print(f"Skipping {ticker}: No valid windows")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df['z'] = np.nan  # Pre-allocate z column as float
    
    # Global de-meaning per T: Skip if <10 points (avoids NaN mean), use .item() for scalar
    for T in HORIZONS_DAYS:
        mask = df['T'] == T
        if mask.sum() < 10:  # Threshold for reliable mean
            print(f"   Skipping de-mean for T={T} ({mask.sum()} points)")
            df.loc[mask, 'z'] = df.loc[mask, 'z_raw']
            continue
        z_raw_series = df.loc[mask, 'z_raw']
        mean_z_series = z_raw_series.mean()
        if pd.isna(mean_z_series):
            print(f"   NaN mean for T={T} — using raw z")
            df.loc[mask, 'z'] = z_raw_series
            continue
        mean_z = mean_z_series.item()  # .item() extracts scalar float — kills FutureWarning
        df.loc[mask, 'z'] = z_raw_series - mean_z
        # Force numeric
        df.loc[mask, 'z'] = pd.to_numeric(df.loc[mask, 'z'], errors='coerce')
    
    # Drop invalid rows *after* all calcs (now z is filled)
    df = df.dropna(subset=['z', 'sigma'])
    
    print(f"   Final for {ticker}: {len(df):,} points after drops")
    return df[['ticker', 'date', 'T', 'z', 'sigma']]

def load_full_dataset(tickers: list, cache_dir="cache") -> pd.DataFrame:
    """Load/cache full dataset (~70K rows for 8 tickers; scales to 2.9M for 200+)"""
    Path(cache_dir).mkdir(exist_ok=True)
    all_dfs = []
    
    for ticker in tickers:
        cache_file = Path(cache_dir) / f"{ticker}_qvar.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                print(f"Loaded cached {ticker}: {len(df):,} points")
            except Exception as e:
                print(f"Cache read failed for {ticker}: {e} — regenerating")
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
        
        if df.empty:
            print(f"Processing {ticker}...")
            log_ret = download_ticker(ticker)
            if len(log_ret) > 1000:  # Skip tiny histories
                df = compute_qvar_for_ticker(log_ret, ticker)
                if not df.empty:
                    df.to_parquet(cache_file)  # Per-ticker save
                    print(f"   Cached {ticker}: {len(df):,} points")
            else:
                print(f"   Skipping {ticker}: Too few returns ({len(log_ret)})")
        
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        print("No valid data generated — check yfinance (pip install yfinance --upgrade)")
        return pd.DataFrame()
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Final dtype cleanup
    numeric_cols = ['z', 'sigma', 'x']
    for col in numeric_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    full_df = full_df.dropna(subset=['z', 'sigma'])
    
    # Save full dataset
    full_df.to_parquet("prize_dataset.parquet")
    print(f"\nTotal dataset ready: {len(full_df):,} rows")
    print(full_df.dtypes)
    print("\nSample head:\n", full_df.head())
    
    return full_df

# Wilmott article tickers (p.37) — start small
PRIZE_TICKERS = [
    "^GSPC", "^DJI", "^FTSE", "AAPL", "MSFT", "AMZN", "BTC-USD", "GLD"
]

if __name__ == "__main__":
    # Generate dataset
    dataset = load_full_dataset(PRIZE_TICKERS)
    
    if not dataset.empty:
        print("\nSuccess! Dataset generated. Run baseline_fit.py for Wilmott plot.")
    else:
        print("Error: Empty dataset — try upgrading yfinance: pip install yfinance --upgrade")
