"""Root conftest.py — add the project parent to sys.path so `import porecdft`
works regardless of how the editable install pth is wired."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
