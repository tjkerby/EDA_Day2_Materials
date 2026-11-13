from pathlib import Path
root=Path(__file__).resolve().parents[1]
(root/'docs').mkdir(exist_ok=True)
(root/'docs/.nojekyll').touch()
