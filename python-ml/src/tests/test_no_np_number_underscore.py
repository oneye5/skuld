import pkgutil
import importlib
import os


def test_no_np_number_underscore_in_repo():
    """Fail if any source file contains the deprecated `np.number_` usage."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    matches = []
    for root, _, files in os.walk(repo_root):
        for f in files:
            if f.endswith(('.py', '.md')):
                path = os.path.join(root, f)
                with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                    text = fh.read()
                    if 'np.number_' in text:
                        matches.append(path)

    assert not matches, f"Found np.number_ in files: {matches}"
