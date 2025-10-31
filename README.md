## Stock Market Anomaly Detection

Detect anomalies in stock price data (example analysis on GME) using classical and deep-learning methods. This repository contains a Streamlit app (`app.py`) for interactive exploration and a Jupyter notebook (`Anomalies Detection.ipynb`) with the analysis, visualizations, and model comparisons.

Key ideas:
- Pull historical price data with `yfinance`
- Build time-series features (returns, volatility, moving averages, Bollinger Bands)
- Detect anomalies using Z-score, Isolation Forest, DBSCAN, LSTM-based prediction residuals, and an Autoencoder reconstruction error
- Visualize and compare model outputs interactively with Plotly inside Streamlit

## Features

- Data retrieval via `yfinance`
- EDA: price, volume, returns, volatility, technical indicators
- Multiple anomaly detection techniques: Z-Score, Isolation Forest, DBSCAN, LSTM, Autoencoder
- Interactive Streamlit dashboard to explore detected anomalies and compare models
- Notebook with reproducible analysis and plots

## Tech / Dependencies

Primary libraries used (see `requirements.txt`):

- Python 3.8+
- streamlit, pandas, numpy
- plotly, matplotlib, seaborn
- scikit-learn (IsolationForest, DBSCAN, StandardScaler)
- yfinance (data retrieval)
- tensorflow (LSTM / Autoencoder)

## Quick start (Windows PowerShell)

1. Clone the repository:

```powershell
git clone https://github.com/yourusername/Stock-Market-Anomaly-Detection.git
cd "Stock-Market-Anomaly-Detection"
```

2. (Optional) Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1    # PowerShell
# For cmd.exe: venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run the interactive app:

```powershell
streamlit run app.py
```

5. Or open and run the analysis notebook:

```powershell
jupyter notebook "Anomalies Detection.ipynb"
```

## How to use the Streamlit app

- Enter a ticker symbol (default `GME`) and set the date range in the sidebar.
- Explore EDA panels (price, MA, Bollinger Bands, returns, volatility, correlation).
- Inspect anomaly plots for each method (Z-Score, Isolation Forest, DBSCAN, LSTM, Autoencoder).
- Compare all methods together and view summary statistics showing the number of anomalies detected by each model.

Notes: LSTM and Autoencoder model training runs in the app and can take significant time/CPU. Consider using smaller datasets or pre-trained models for a snappier UI.

## Notebook

The notebook (`Anomalies Detection.ipynb`) contains step-by-step data preparation, feature engineering, plotting, and model implementations used in the Streamlit app (including code for performance comparison: precision, recall, F1-score).

## Project structure

- `app.py` — Streamlit app with EDA and anomaly detection visualizations
- `Anomalies Detection.ipynb` — exploratory notebook and model experiments
- `requirements.txt` — dependency list
- `README.md` — this file

## Methodology (brief)

1. Data retrieval with `yfinance`.
2. Feature engineering: returns, rolling volatility, moving averages, Bollinger Bands, RSI, etc.
3. Anomaly detection methods implemented:
   - Z-Score thresholding on price
   - Isolation Forest on scaled feature vectors
   - DBSCAN clustering; label -1 as anomaly
   - LSTM: sequence prediction; anomalies by high prediction MSE
   - Autoencoder: reconstruction error > percentile threshold
4. Compare methods using precision/recall/F1 against a combined anomaly label when available.

## Caveats & Tips

- Deep learning models (LSTM/Autoencoder) are resource-intensive; prefer a GPU or smaller data slices for experimentation.
- Results depend heavily on choice of windows, thresholds, contamination parameter (for IsolationForest), and DBSCAN hyperparameters.
- Time-series cross-validation and careful labeling are required for robust evaluation in production.

## Suggested next steps / Improvements

- Save trained models to `models/` and load them instead of re-training in the app.
- Add parameter controls in the Streamlit UI (thresholds, contamination, DBSCAN eps/min_samples, LSTM epochs).
- Add unit tests for data preprocessing and deterministic model components.
- Add CI to check dependency compatibility and run lightweight tests.

## Contributing

Contributions are welcome. Please open an issue to discuss changes or submit a pull request with tests and a clear description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

