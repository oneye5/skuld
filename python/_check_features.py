from skuld_research.data.csv_loader import load_raw_csv
from pathlib import Path

raw = load_raw_csv(Path('../data/data_long.csv'))

fund_features = sorted(raw.fundamentals.columns.tolist()) if not raw.fundamentals.empty else []
macro_features = sorted(raw.macro.columns.tolist()) if not raw.macro.empty else []

keywords = ['book', 'equity', 'asset', 'liabilit', 'cash_flow', 'gross', 'operating',
            'income', 'earn', 'revenue', 'profit', 'balance', 'shares']

print(f'Fundamental features ({len(fund_features)}):')
for f in fund_features:
    print(' ', f)

print()
print(f'Macro features ({len(macro_features)}):')
for f in macro_features:
    print(' ', f)

print()
print('Potentially relevant for value/quality (in fundamentals):')
for f in fund_features:
    if any(k in f for k in keywords):
        n = int(raw.fundamentals[f].notna().sum())
        print(f'  {f}  ({n} non-null observations)')
