# data_loader.py - read price data for stocks in S&P 500 and save a parquet file
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

HORIZONS = 5*(np.arange(26)+1)   # does 1 to 26 weeks, can also do [5, 10, 20, 40, 80, 160]

# from R: l.out <- BatchGetSymbols(tickers=tickers,first.date=as.Date('1950-01-01'),last.date=as.Date('2025-12-03'),thresh.bad.data=0.25)# stocks with at least 0.25 of dates since 1950, so about 19 years of data
TICKERS = ["MMM", "AOS", "ABT", "ACN", "ADBE", "AMD", "AES", "AFL", "A", "APD", "AKAM", "ALB", "ARE",              "ALGN", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AEE", "AEP", "AXP", "AIG", "AMT",              "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "AAPL", "AMAT", "ACGL", "ADM", "AJG",              "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX",              "BDX", "BBY", "TECH", "BIIB", "BLK", "BK", "BA", "BKNG", "BSX", "BMY", "BRO", "BLDR", "BG",              "BXP", "CHRW", "CDNS", "CPT", "CPB", "COF", "CAH", "CCL", "CAT", "CBRE", "COR", "CNC", "CNP",              "CF", "CRL" , "SCHW", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CLX",              "CME", "CMS" , "KO", "CTSH", "CL", "CMCSA", "CAG", "COP", "ED", "STZ", "COO", "CPRT", "GLW",              "CSGP", "COST", "CTRA", "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DVA", "DECK", "DE", "DVN",              "DXCM", "DLR", "DLTR", "D", "DPZ", "DOV", "DHI", "DTE", "DUK", "DD", "ETN", "EBAY", "ECL",              "EIX", "EW", "EA", "ELV", "EME", "EMR", "ETR", "EOG", "EQT", "EFX", "EQIX", "EQR", "ERIE",              "ESS", "EL", "EG", "EVRG", "ES", "EXC", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO",              "FAST", "FRT", "FDX", "FIS", "FITB", "FSLR", "FE", "FISV", "F", "BEN", "FCX", "GRMN", "IT",              "GE", "GEN", "GD", "GIS", "GPC", "GILD", "GPN", "GL", "GS", "HAL", "HIG", "HAS", "DOC",              "HSIC", "HSY", "HOLX", "HD", "HON", "HRL", "HST", "HPQ", "HUBB", "HUM", "HBAN", "IBM", "IEX",              "IDXX", "ITW", "INCY", "INTC", "ICE", "IFF", "IP", "INTU", "ISRG", "IVZ", "IRM", "JBHT", "JBL",              "JKHY", "J", "JNJ", "JCI", "JPM", "K", "KEY", "KMB", "KIM", "KLAC", "KR", "LHX", "LH", "LRCX",              "LVS", "LDOS", "LEN", "LII", "LLY", "LIN", "LYV", "LKQ", "LMT", "L", "LOW", "MTB", "MAR",              "MMC", "MLM", "MAS", "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "MET", "MTD", "MGM",              "MCHP", "MU", "MSFT", "MAA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS",              "MSI", "NDAQ", "NTAP", "NFLX", "NEM", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NRG",              "NUE", "NVDA", "NVR", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE", "ORCL", "PCAR", "PKG",              "PSKY", "PH", "PAYX", "PNR", "PEP", "PFE", "PCG", "PNW", "PNC", "POOL", "PPG", "PPL", "PFG",              "PG", "PGR", "PLD", "PRU", "PEG", "PTC", "PSA", "PHM", "PWR", "QCOM", "DGX", "RL", "RJF",              "RTX", "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP", "ROST", "RCL",              "SPGI", "CRM", "SBAC", "SLB", "STX", "SRE", "SHW", "SPG", "SWKS", "SJM", "SNA", "SO", "LUV",              "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SNPS", "SYY", "TROW", "TTWO", "TPR", "TGT",              "TDY", "TER", "TXN", "TPL", "TXT", "TMO", "TJX", "TKO", "TSCO", "TT", "TDG", "TRV", "TRMB",              "TFC", "TYL", "TSN", "USB", "UDR", "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR",              "VRSN", "VZ", "VRTX", "VTRS", "VMC", "WRB", "GWW", "WAB", "WMT", "DIS", "WBD", "WM", "WAT",              "WEC", "WFC", "WELL", "WST", "WDC", "WY", "WSM", "WMB", "WTW", "WYNN", "XEL", "YUM", "ZBRA", "ZBH"]
ntick = len(TICKERS)

# Path("cache").mkdir(exist_ok=True)
all_data = []

print("Generating Q-Variance Challenge Dataset...")

for ticker in TICKERS:
    print(f"→ {ticker}", end="")
    price = yf.download(ticker, period="max", progress=False, auto_adjust=True)["Close"]
    
    ret = np.log(price).diff().dropna().values

    rows = []
    scale = np.sqrt(252)

    for T in HORIZONS:
        i = 0
        while i + T <= len(ret):
            window = ret[i:i+T]     # T points
            if len(window) < T * 0.8:
                break

            x = window.sum()   # total price change over the period
            sigma = np.std(window, ddof=0) * scale  # use np.std to get std over period, ddof=1 means divisor is N-1
            z_raw = x / np.sqrt(T / 252.0)

            # REJECT BAD WINDOWS
            if not (np.isfinite(sigma) and sigma > 0 and np.isfinite(z_raw)):
                i += T
                continue

            rows.append({          # append row of data for this period
                "ticker": ticker,
                "date": price.index[i + T - 1].date(),
                "T": T,
                "z_raw": float(z_raw),
                "sigma": float(sigma)
            })
            i += T

    if not rows:
        print(" [no data]")
        continue

    df = pd.DataFrame(rows)

    # CLEAN BEFORE DE-MEANING 
    df = df[np.isfinite(df['z_raw']) & np.isfinite(df['sigma']) & (df['sigma'] > 0)]

    # NOW de-mean safely, this step groups by ticker and T, and subtracts the group mean 
    df["z"] = df.groupby(["ticker", "T"])["z_raw"].transform(lambda g: g - g.mean())

    df = df.drop(columns="z_raw")
    df = df.dropna().reset_index(drop=True)  # Final clean

    #df.to_parquet(Path("cache") / f"{ticker}.parquet")
    print(f" → {len(df)} clean windows")
    all_data.append(df)

full = pd.concat(all_data, ignore_index=True)

n = len(full) // 3
part1 = full.iloc[:n]
part2 = full.iloc[n:2*n]
part3 = full.iloc[2*n:]

# Save 3 small files
part1.to_parquet("dataset_part1.parquet", compression=None)
part2.to_parquet("dataset_part2.parquet", compression=None)
part3.to_parquet("dataset_part3.parquet", compression=None)

print("Done! 3 files created — each <25 MB")
