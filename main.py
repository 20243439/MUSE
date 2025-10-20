import os
import sys


def main():
    # Ensure we can import src.* modules
    base = os.path.dirname(__file__)
    repo_root = os.path.join(base, 'MUSE')
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    src_core = os.path.join(repo_root, 'src', 'core')
    if src_core not in sys.path:
        sys.path.insert(0, src_core)
    # import and call the entry
    from src.core.run import run_entry
    run_entry()


if __name__ == "__main__":
    main()
