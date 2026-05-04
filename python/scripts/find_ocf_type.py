"""Find correct OCF type name for Yahoo Finance fundamentals-timeseries API."""
import requests

headers = {"User-Agent": "Mozilla/5.0"}
ticker = "SPK.NZ"  # Uses direct method

# Try every possible OCF-related naming variant for the timeseries API
ocf_candidates = [
    "annualCashFlowsfromusedinOperatingActivitiesDirect",
    "annualCashFlowsFromUsedInOperatingActivitiesDirect",
    "annualNetCashFromOperatingActivities",
    "annualTotalCashFromOperatingActivities",
    "annualOperatingActivities",
    "annualNetCashProvidedByOperatingActivities",
    "annualNetCashProvidedByFinancingActivities",
    "annualNetCashProvidedByInvestingActivities",
    "annualClassesOfCashReceiptsfromOperatingActivities",
    "annualReceiptsfromCustomers",
    "annualClassesofCashReceiptsfromOperatingActivities",
    "annualClassesofCashPayments",
    "annualPaymentstoSuppliersforGoodsandServices",
    "annualOperatingCashFlow",
    "annualCashGeneratedFromOperations",
]

type_str = ",".join(ocf_candidates)
url = (
    f"https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}"
    f"?merge=false&padTimeSeries=true&period1=493590046&period2=2750557599"
    f"&type={type_str}&lang=en-NZ&region=NZ"
)

resp = requests.get(url, headers=headers, timeout=30)
resp.raise_for_status()
data = resp.json()
results = data.get("timeseries", {}).get("result", [])
print(f"Returned {len(results)} result objects for {ticker}\n")

for result in results:
    meta = result.get("meta", {})
    feat = meta.get("type", ["?"])[0]
    if feat in result and result[feat]:
        pts = [p for p in result[feat] if p is not None]
        if pts:
            dates = [p["asOfDate"] for p in pts]
            vals = [p.get("reportedValue", {}).get("raw") for p in pts]
            print(f"[FOUND] {feat}: {len(pts)} obs [{min(dates)}-{max(dates)}]")
            for d, v in zip(dates, vals):
                print(f"  {d}: {v/1e6:.1f}M" if v else f"  {d}: null")
