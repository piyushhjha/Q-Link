# Q-Link (Starter): Superdense Coding Simulator

This is a minimal, production-friendly starter for your AQVH913 project.

## Setup

### Option A: Jupyter (fastest to start)
```bash
# 1) Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda
```bash
conda create -y -n qlink python=3.11
conda activate qlink
pip install -r requirements.txt
```

## Run quick demo (CLI)
```bash
python simulate.py --p 0.0 0.02 0.04 0.06 0.08 --shots 4096
```

## Launch the Streamlit UI
```bash
streamlit run streamlit_app.py
```

## Run tests
```bash
pytest -q
```

## Files
- `superdense.py` — core circuits, noise, and utilities
- `simulate.py` — CLI to sweep noise and print QBER + confusion matrix
- `streamlit_app.py` — simple UI to experiment interactively
- `tests/test_superdense.py` — unit tests for noiseless and extreme-noise cases
