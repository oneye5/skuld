import pandas as pd
from pathlib import Path
from config.file_paths import OUTPUT_DIR

class DataSampler:
    def __init__(self, run_id: str = "latest"):
        self.debug_dir = OUTPUT_DIR / "debug" / run_id
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def sample(self, df: pd.DataFrame, stage_name: str, window_id: int | None = None):
        """
        Save a sample of the dataframe to the debug directory.
        """
        if df is None or df.empty:
            print(f"Warning: DataFrame is empty or None at stage {stage_name}")
            return

        prefix = f"window{window_id}_" if window_id is not None else ""
        filename = f"{prefix}{stage_name}.csv"
        filepath = self.debug_dir / filename

        # Save head, tail, and some stats
        sample_df = pd.concat([df.head(5), df.tail(5)])
        
        # Also save a summary file
        summary_path = self.debug_dir / f"{prefix}{stage_name}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Columns: {list(df.columns)}\n")
            f.write(f"Dtypes:\n{df.dtypes}\n")
            f.write(f"Missing values:\n{df.isnull().sum().sort_values(ascending=False).head(10)}\n")

        sample_df.to_csv(filepath)
        print(f"Sampled data at {stage_name} to {filepath}")

_global_sampler = None

def get_sampler(run_id: str = "latest") -> DataSampler:
    global _global_sampler
    if _global_sampler is None:
        _global_sampler = DataSampler(run_id)
    return _global_sampler
