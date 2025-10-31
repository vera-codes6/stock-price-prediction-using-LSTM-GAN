<div align="center">

# Stock Price Prediction using GANs (with LSTM/GRU baselines)

Capstone project (DATS6501) — Team: Chen Chen, HungChun Lin

[Published: Journal of Computer Science — "Stock price prediction using Generative Adversarial Networks"](https://thescipub.com/abstract/jcssp.2021.188.196)

</div>

## Overview
This repository implements time-series models for stock price prediction, comparing classic sequence models (LSTM/GRU) against Generative Adversarial Networks (GAN) including WGAN-GP. The pipeline enriches price data with technical indicators, Fourier components, and news sentiment (FinBERT), then trains and evaluates models on next-step forecasting tasks.

Key features:
- Data engineering: technical indicators and Fourier transform features
- Optional news sentiment from SeekingAlpha headlines via FinBERT
- Baselines: LSTM and GRU
- Adversarial models: Basic GAN and WGAN-GP (GRU-based generator, CNN discriminator)
- Pretrained weights for quick inference

## Repository structure
```
Code/
	1. Load_data.py              # Adds technical indicators & Fourier features
	2. data_preprocessing.py     # Build train/test arrays, scalers, indices
	3. Baseline_LSTM.py          # LSTM baseline training & evaluation
	3. Baseline_GRU.py           # GRU baseline training & evaluation
	4. Basic_GAN.py              # Basic GAN training (GRU gen + CNN disc)
	5. WGAN_GP.py                # WGAN-GP training
	6. Test_prediction.py        # Inference & plotting for a saved generator
	DATA.csv                     # Sample market/economic data (Apple)
	News.csv                     # Sample news sentiment (FinBERT)
	Models/
		GAN_30to3.h5, GAN_3to1.h5
		GRU_30to3.h5, GRU_3to1.h5
		LSTM_30to3.h5, LSTM_3to1.h5
		WGAN_GP_30to3.h5, WGAN_GP_3to1.h5
	NLP/
		web_scraping.ipynb         # Scrape SeekingAlpha headlines
		finbert_training.ipynb     # Score headlines with FinBERT

Meeting Note/                  # Weekly progress notes
Paper and Report/              # Final report and materials
Presentation_Document/         # Presentation slides
Relevant Articles/             # References collected
```

## Data
- Market/economic data: `Code/DATA.csv` (includes Apple price series and indices)
- News sentiment: `Code/News.csv` (pre-computed FinBERT sentiment scores)

You can regenerate `News.csv` using notebooks in `Code/NLP/`:
- `web_scraping.ipynb` scrapes news headlines (requires a web driver)
- `finbert_training.ipynb` uses ProsusAI/FinBERT to score headlines from -1 to 1

## Environment
Recommended setup:
- Python 3.8–3.10
- TensorFlow 2.x (GPU recommended for GAN/WGAN-GP)
- NumPy, Pandas, scikit-learn, statsmodels, Matplotlib
- Optional (NLP): Selenium + a browser driver, transformers, PyTorch

Example installation (create/activate your virtual environment first):
```
pip install tensorflow numpy pandas scikit-learn statsmodels matplotlib
# Optional for NLP
pip install selenium transformers torch
```

## Quickstart
All scripts assume execution from the `Code/` directory.

1) Feature engineering
- Place `DATA.csv` and `News.csv` in `Code/` (sample files are provided).
- Run `1. Load_data.py` to compute technical indicators and Fourier features.
	- Output: `Finaldata_with_Fourier.csv` and exploratory plots.

2) Build datasets
- Run `2. data_preprocessing.py` to create arrays and scalers.
	- Outputs include: `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`, `yc_train.npy`, `yc_test.npy`, `index_train.npy`, `index_test.npy`, plus `X_scaler.pkl`, `y_scaler.pkl`.
	- Configure input/output horizons via `n_steps_in` and `n_steps_out` in this script (defaults currently set in the file).

3) Train a model (choose one)
- LSTM baseline: `3. Baseline_LSTM.py` (saves e.g., `LSTM_3to1.h5`)
- GRU baseline: `3. Baseline_GRU.py` (saves e.g., `GRU_3to1.h5`)
- Basic GAN: `4. Basic_GAN.py` (periodically saves generator checkpoints)
- WGAN-GP: `5. WGAN_GP.py` (periodically saves generator checkpoints)

4) Evaluate / visualize
- `6. Test_prediction.py` loads a saved generator model, plots predicted vs. actual price on the test set, saves `test_predicted.csv`, and prints RMSE.
- Update the model path in `Test_prediction.py` (variable `G_model = tf.keras.models.load_model(...)`) to point to your trained or pretrained `.h5` file.

## Pretrained models
Pretrained weights are provided under `Code/Models/` for quick testing across setups and horizons, e.g.:
- `WGAN_GP_3to1.h5`, `WGAN_GP_30to3.h5`
- `GAN_3to1.h5`, `GAN_30to3.h5`
- `LSTM_3to1.h5`, `GRU_3to1.h5` (and 30→3 variants)

To use one in `6. Test_prediction.py`, set the path, for example:
```
G_model = tf.keras.models.load_model('Models/WGAN_GP_3to1.h5')
```

## Configuration tips
- Forecast horizon: adjust `n_steps_in` and `n_steps_out` in `2. data_preprocessing.py`.
- Learning rate, batch size, and epochs: tune in each training script (e.g., `LR`, `BATCH_SIZE`, `N_EPOCH`).
- GPU: GAN/WGAN-GP training benefits greatly from a CUDA-enabled GPU.

## Results and plots
- Training scripts plot loss curves and predicted vs. actual prices.
- `6. Test_prediction.py` writes `test_predicted.csv` and reports RMSE on the test window.

## Citation
If you use this work, please cite the paper:

Journal: Journal of Computer Science  
Title: Stock price prediction using Generative Adversarial Networks  
Link: https://thescipub.com/abstract/jcssp.2021.188.196

## License
MIT License — see `LICENSE` for details.

## Acknowledgments
- ProsusAI FinBERT: https://github.com/ProsusAI/finBERT
- Community resources and articles listed under `Relevant Articles/`

## Contact
Questions or issues: feel free to open an issue.  
Team: Chen Chen, HungChun Lin

