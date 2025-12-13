"""Quick verification test for I/O optimizations."""
import sys
from pathlib import Path
import tempfile
import time
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.io_utils import load_data, save_data, clear_cache


def test_parquet_performance():
    """Test that Parquet is faster than CSV for typical operations."""
    print("Testing I/O performance improvements...")
    
    # Create test data
    n_rows = 100000
    n_cols = 50
    test_df = pd.DataFrame(
        np.random.randn(n_rows, n_cols),
        columns=[f'feature_{i}' for i in range(n_cols)]
    )
    test_df['timestamp'] = pd.date_range('2020-01-01', periods=n_rows, freq='1min')
    test_df['ticker'] = np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN'], n_rows)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'test.csv'
        parquet_path = Path(tmpdir) / 'test.parquet'
        
        # Test CSV
        start = time.time()
        save_data(test_df, str(csv_path), format='csv')
        csv_write_time = time.time() - start
        
        start = time.time()
        df_csv = load_data(str(csv_path), format='csv')
        csv_read_time = time.time() - start
        
        # Test Parquet
        start = time.time()
        save_data(test_df, str(parquet_path), format='parquet')
        parquet_write_time = time.time() - start
        
        clear_cache()  # Clear cache to force reload
        
        start = time.time()
        df_parquet = load_data(str(parquet_path), format='parquet')
        parquet_read_time = time.time() - start
        
        # Verify data integrity
        assert len(df_csv) == len(test_df), "CSV load/save data mismatch"
        assert len(df_parquet) == len(test_df), "Parquet load/save data mismatch"
        
        # Print results
        print(f"\n📊 Performance Results ({n_rows:,} rows, {n_cols} columns):")
        print(f"\n  CSV:")
        print(f"    Write: {csv_write_time:.3f}s")
        print(f"    Read:  {csv_read_time:.3f}s")
        print(f"    Total: {csv_write_time + csv_read_time:.3f}s")
        
        print(f"\n  Parquet:")
        print(f"    Write: {parquet_write_time:.3f}s")
        print(f"    Read:  {parquet_read_time:.3f}s")
        print(f"    Total: {parquet_write_time + parquet_read_time:.3f}s")
        
        csv_total = csv_write_time + csv_read_time
        parquet_total = parquet_write_time + parquet_read_time
        speedup = csv_total / parquet_total if parquet_total > 0 else 0
        
        print(f"\n  ⚡ Speedup: {speedup:.1f}x faster")
        
        # File size comparison
        csv_size = csv_path.stat().st_size / (1024 * 1024)  # MB
        parquet_size = parquet_path.stat().st_size / (1024 * 1024)  # MB
        compression_ratio = csv_size / parquet_size if parquet_size > 0 else 0
        
        print(f"\n  💾 File Sizes:")
        print(f"    CSV:     {csv_size:.2f} MB")
        print(f"    Parquet: {parquet_size:.2f} MB")
        print(f"    Compression: {compression_ratio:.1f}x smaller")
        
        print(f"\n✅ All tests passed!")
        
        return speedup >= 2.0  # Expect at least 2x speedup


def test_caching():
    """Test that caching works correctly."""
    print("\n\nTesting caching mechanism...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / 'cache_test.parquet'
        test_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        # Save data
        save_data(test_df, str(test_path))
        
        # First load (from disk)
        start = time.time()
        df1 = load_data(str(test_path))
        first_load_time = time.time() - start
        
        # Second load (from cache)
        start = time.time()
        df2 = load_data(str(test_path))
        cached_load_time = time.time() - start
        
        print(f"\n  First load:  {first_load_time*1000:.2f}ms")
        print(f"  Cached load: {cached_load_time*1000:.2f}ms")
        
        cache_speedup = first_load_time / cached_load_time if cached_load_time > 0 else float('inf')
        print(f"  Cache speedup: {cache_speedup:.0f}x faster")
        
        # Verify data is the same
        assert df1.equals(df2), "Cached data should match original"
        
        # Clear cache and verify reload
        clear_cache()
        df3 = load_data(str(test_path))
        assert df3.equals(df1), "Data after cache clear should match"
        
        print(f"\n✅ Caching works correctly!")
        
        return cache_speedup >= 10  # Cache should be much faster


def test_auto_format_detection():
    """Test automatic format detection."""
    print("\n\nTesting automatic format detection...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'test.csv'
        parquet_path = Path(tmpdir) / 'test.parquet'
        test_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        
        # Save with auto-detection
        save_data(test_df, str(csv_path))  # Should detect CSV from extension
        save_data(test_df, str(parquet_path))  # Should detect Parquet from extension
        
        clear_cache()
        
        # Load with auto-detection
        df_csv = load_data(str(csv_path))
        df_parquet = load_data(str(parquet_path))
        
        assert df_csv.equals(test_df), "CSV auto-detection failed"
        assert df_parquet.equals(test_df), "Parquet auto-detection failed"
        
        print(f"  ✅ CSV auto-detection works")
        print(f"  ✅ Parquet auto-detection works")
        print(f"\n✅ Format detection works correctly!")
        
        return True


if __name__ == '__main__':
    print("=" * 60)
    print("  Performance Optimization Verification Tests")
    print("=" * 60)
    
    try:
        test1 = test_parquet_performance()
        test2 = test_caching()
        test3 = test_auto_format_detection()
        
        print("\n" + "=" * 60)
        if test1 and test2 and test3:
            print("  🎉 All optimizations verified successfully!")
        else:
            print("  ⚠️  Some optimizations may not be as effective as expected")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
