"""Quick benchmark of clustering speed."""

import time
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)

from features.cluster_fast import compute_clusters_fast
from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers

# Load once
print('Loading data...')
long_df = load_long_data()
wide_df = long_to_wide(add_macro_prefix(clean_and_classify_tickers(long_df)))
print(f'Data loaded: {len(wide_df)} rows')

# Time just the clustering (3 runs)
print('\nBenchmarking clustering...')
times = []
for i in range(3):
    start = time.perf_counter()
    cluster_map = compute_clusters_fast(wide_df, n_clusters=10)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f'  Run {i+1}: {elapsed:.0f}ms')
    
print(f'\nAverage: {sum(times)/len(times):.0f}ms')
print(f'Clusters: {len(set(cluster_map.values()))}')
print(f'Tickers: {len(cluster_map)}')
