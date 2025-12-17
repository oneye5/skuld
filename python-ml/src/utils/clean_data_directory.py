from src.utils.path_utils import get_skuld_root


def clean_data_directory():
    _root = get_skuld_root()
    _dir = _root / "python-ml" / "data"

    # 1. Safety check: Ensure directory exists
    if not _dir.exists():
        print(f"Directory not found: {_dir}")
        return

    print(f"Cleaning directory: {_dir}")

    # 2. Iterate and delete files
    deleted_count = 0
    for item in _dir.iterdir():
        try:
            if item.is_file():
                # specific check to avoid deleting gitkeep if you use it
                if item.name == ".gitkeep":
                    continue

                item.unlink()
                print(f"Deleted: {item.name}")
                deleted_count += 1
        except Exception as e:
            print(f"Error deleting {item.name}: {e}")

    print(f"--- Cleanup Complete. {deleted_count} files deleted. ---")


if __name__ == "__main__":
    clean_data_directory()