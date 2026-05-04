"""Quick check: sector from assetProfile and SPK OCF breakdown"""
import requests
import yfinance as yf
import pandas as pd

headers = {"User-Agent": "Mozilla/5.0"}

# Check sector from YF assetProfile module
for ticker in ["SPK.NZ", "FPH.NZ", "AIR.NZ"]:
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=assetProfile"
    resp = requests.get(url, headers=headers, timeout=20)
    data = resp.json()
    result = data.get("quoteSummary", {}).get("result", [{}])
    if result:
        profile = result[0].get("assetProfile", {})
        sector = profile.get("sector")
        industry = profile.get("industry")
        country = profile.get("country")
        print(f"{ticker}: sector={sector}, industry={industry}, country={country}")
    else:
        err = data.get("quoteSummary", {}).get("error")
        print(f"{ticker}: ERROR - {err}")

print()

# SPK OCF via yfinance
t = yf.Ticker("SPK.NZ")
cf = t.cashflow
if cf is not None and not cf.empty:
    print(f"SPK.NZ CF items ({cf.shape[0]} rows):")
    key_rows = [r for r in cf.index if any(kw in r for kw in ["Operating", "Cash Flow From", "Activities Direct"])]
    for row in key_rows:
        vals = {}
        for c in cf.columns:
            if pd.notna(cf.loc[row, c]):
                vals[str(c.date())] = round(cf.loc[row, c] / 1e6, 1)
        print(f"  {row}: {vals}")

print()
# Validate SPK OCF FY2023 vs reference ~540M
print("SPK FY2023 OCF reference: ~540M NZD")
