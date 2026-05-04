# SP500 Stock Predictor

A machine learning system that generates buy signals for S&P 500 stocks using XGBoost ensemble models trained on technical indicators, earnings data, sector performance, and Google Trends.

## Quick Start

Price data, trends, earnings, and sector data are already included in `data_cache/`. Model selection parameters for 503 tickers are included in `params_log/report_selected.csv`. You only need to install dependencies and generate the models:

```bash
pip install -r requirements.txt
python training/quick_model_generator.py        # all 503 tickers
# or for specific tickers only:
python training/quick_model_generator.py AAPL MSFT NVDA
```

Then start the live signal server:

```bash
cd live && uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download data

```bash
python data/downloader.py
```

### 2. Train models

Run the full hyperparameter search + ensemble selection (slow):
```bash
python training/ml_optimizer.py        # grid search over features and XGBoost params
python training/ensemble_selector.py   # select best K=3 ensemble per ticker/fold
```

Or retrain directly from the included `params_log/report_selected.csv` (fast, no grid search):
```bash
python training/quick_model_generator.py
# specific tickers only:
python training/quick_model_generator.py AAPL MSFT NVDA
```

### 3. Run simulation

```bash
python simulation/simulation.py
```

### 4. Start live signal server

```bash
cd live && uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

## Notes

- `data_cache/` is included (price, trends, earnings, sector data). `saved_models/` is excluded — run step 2 to generate models locally.
- The live server refreshes signals every 30 minutes during NYSE trading hours.
- Tested on Python 3.11+.
