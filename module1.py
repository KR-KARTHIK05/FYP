"""Deterministic LSTM carbon-intensity forecast service.

Data pipeline
-------------
1. Load Ember India monthly dataset (india_monthly_full_release_long_format.csv).
2. Compute true carbon intensity (gCO2/kWh) via the ratio:
       CI = Total_emissions (MtCO2) / Electricity_generation (TWh) * 1000
   Falls back to IPCC-weighted fuel-mix calculation, then a proportional
   scaling of total emissions to India's known 600–800 gCO2/kWh band.
3. Hourly-interpolate the monthly series for the LSTM sequence window.

Model
-----
Two-layer LSTM (hidden=64, dropout=0.2) trained for up to MAX_EPOCHS with
early stopping (patience=PATIENCE).  Model + scaler state are persisted to
disk so each cold start re-uses the last trained weights if the dataset has
not changed (mtime-keyed cache).

Live integration
----------------
When ELECTRICITY_MAPS_TOKEN is set in the environment, `get_current_carbon_intensity`
fetches live gCO2/kWh from the Electricity Maps REST API (zone defaults to 'IN').
Otherwise it falls back to the first step of the LSTM 24-hour forecast.
"""

from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
from threading import RLock
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH             = Path(__file__).with_name('india_monthly_full_release_long_format.csv')
PREDICTION_HISTORY_PATH = Path(__file__).with_name('india_forecast_predictions.csv')
MODEL_CHECKPOINT_PATH = Path(__file__).with_name('carbon_lstm_checkpoint.pt')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_EPOCHS      = 200
PATIENCE        = 30    # early-stopping patience (epochs without val improvement)
SEQUENCE_LENGTH = 24    # input window (hours)
PRED_LENGTH     = 24    # output horizon (hours)
TRAIN_FRACTION  = 0.80
HIDDEN_SIZE     = 64
NUM_LAYERS      = 2
DROPOUT         = 0.2
LEARNING_RATE   = 3e-3

# India grid carbon intensity plausible range (gCO2/kWh)
_CI_MIN = 300.0
_CI_MAX = 950.0

# IPCC AR6 WG3 Table 6.1 lifecycle emission factors (gCO2eq/kWh, median)
_IPCC_FACTORS: dict[str, float] = {
    'Coal': 820.0, 'Gas': 490.0, 'Oil': 650.0,
    'Nuclear': 12.0, 'Hydro': 24.0, 'Wind': 11.0,
    'Solar': 45.0, 'Bioenergy': 230.0, 'Other renewables': 30.0,
}

LIVE_AGGREGATION_MINUTES = max(
    1, int(os.environ.get('FORECAST_AGGREGATION_MINUTES', '15')))

_FORECAST_CACHE: dict = {}
_DATA_LOCK = RLock()
_LIVE_BUCKETS: dict = {}


# ---------------------------------------------------------------------------
# Electricity Maps live API
# ---------------------------------------------------------------------------

def get_current_carbon_intensity(fallback: float | None = None) -> float:
    """Return current India grid carbon intensity (gCO2/kWh).

    Priority:
      1. Electricity Maps REST API (when ELECTRICITY_MAPS_TOKEN is set)
      2. Caller-supplied fallback value
      3. First step of the LSTM 24-hour forecast
    """
    token = os.environ.get('ELECTRICITY_MAPS_TOKEN')
    zone  = os.environ.get('ELECTRICITY_MAPS_ZONE', 'IN')
    if token:
        req = Request(
            f'https://api.electricitymaps.com/v3/carbon-intensity/latest?zone={zone}',
            headers={'auth-token': token},
        )
        with urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        return float(payload['carbonIntensity'])

    if fallback is not None:
        return float(fallback)

    return float(run_forecast()['forecast'][0])


# ---------------------------------------------------------------------------
# LSTM model definition
# ---------------------------------------------------------------------------

class CarbonForecasterLSTM(nn.Module):
    """Two-layer LSTM with dropout that maps a 24-h history to a 24-h forecast."""

    def __init__(self,
                 input_size: int = 1,
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS,
                 output_size: int = PRED_LENGTH,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))  # last time-step only


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def _compute_ci_series(data: pd.DataFrame) -> pd.Series:
    """Compute carbon intensity (gCO2/kWh) from the Ember long-format dataset.

    Strategy 1 (preferred): direct ratio of total emissions / total generation.
    Strategy 2: IPCC-weighted fuel-mix average.
    Strategy 3: proportional scaling of raw emissions to India's known CI band.
    """
    india = data[data['Country'].str.strip() == 'India'].copy()

    # Identify the date column
    date_col = next(
        (c for c in ('Date', 'Month', 'Year') if c in india.columns), None
    )
    if date_col is None:
        raise ValueError('No recognisable date column (Date/Month/Year) in dataset.')

    india[date_col] = pd.to_datetime(india[date_col], errors='coerce')
    india['Value']  = pd.to_numeric(india['Value'], errors='coerce')
    india = india.dropna(subset=[date_col, 'Value'])

    # ---- Strategy 1: MtCO2 / TWh ratio --------------------------------
    em = (india[india['Variable'].str.strip() == 'Total emissions']
          .groupby(date_col)['Value'].sum())
    gen = (india[india['Variable'].str.strip() == 'Electricity generation']
           .groupby(date_col)['Value'].sum())

    if not em.empty and not gen.empty:
        combined = pd.DataFrame({'em': em, 'gen': gen}).dropna()
        combined = combined[combined['gen'] > 0]
        if len(combined) >= 12:
            # MtCO2 / TWh × 1000 = gCO2/kWh
            ci = (combined['em'] / combined['gen']) * 1000.0
            ci = ci.clip(_CI_MIN, _CI_MAX).sort_index()
            logger.info('CI strategy 1 (emission/generation ratio): %d points', len(ci))
            return ci

    # ---- Strategy 2: IPCC-weighted fuel mix ----------------------------
    fuel_frames = {}
    for variable, factor in _IPCC_FACTORS.items():
        subset = india[india['Variable'].str.strip() == variable]
        if not subset.empty:
            fuel_frames[variable] = subset.groupby(date_col)['Value'].sum()

    if len(fuel_frames) >= 3:
        gen_df = pd.DataFrame(fuel_frames).fillna(0)
        total_gen = gen_df.sum(axis=1).replace(0, np.nan)
        weighted_em = sum(
            gen_df[v] * _IPCC_FACTORS[v]
            for v in gen_df.columns
        )
        ci = (weighted_em / total_gen).dropna().clip(_CI_MIN, _CI_MAX).sort_index()
        logger.info('CI strategy 2 (IPCC fuel mix): %d points', len(ci))
        return ci

    # ---- Strategy 3: proportional scaling of raw emissions -------------
    # Total emissions correlate monotonically with CI; map to India's 600–800 band.
    raw = em if not em.empty else india[
        india['Variable'].str.strip() == 'Total emissions'
    ].groupby(date_col)['Value'].sum()

    if raw.empty:
        raise ValueError('No usable emission data found for India in the dataset.')

    p5, p95 = float(raw.quantile(0.05)), float(raw.quantile(0.95))
    if p95 > p5:
        ci = 600 + ((raw - p5) / (p95 - p5)) * 200   # maps to 600–800 gCO2/kWh
    else:
        ci = pd.Series(700.0, index=raw.index)

    ci = ci.clip(_CI_MIN, _CI_MAX).sort_index()
    logger.warning('CI strategy 3 (proportional scaling, least accurate): %d points', len(ci))
    return ci


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model: CarbonForecasterLSTM,
                     scaler: MinMaxScaler,
                     meta: dict,
                     data_version: int) -> None:
    """Persist model weights + scaler state to disk."""
    torch.save({
        'data_version': data_version,
        'model_state': model.state_dict(),
        'scaler_data_min': scaler.data_min_,
        'scaler_data_max': scaler.data_max_,
        'scaler_data_range': scaler.data_range_,
        'scaler_scale': scaler.scale_,
        'scaler_feature_range': scaler.feature_range,
        'meta': meta,
    }, MODEL_CHECKPOINT_PATH)
    logger.info('Checkpoint saved → %s', MODEL_CHECKPOINT_PATH)


def _load_checkpoint(data_version: int):
    """Load a checkpoint if it matches the current dataset version.

    Returns (model, scaler, meta) or (None, None, None) on mismatch/error.
    """
    if not MODEL_CHECKPOINT_PATH.exists():
        return None, None, None
    try:
        ckpt = torch.load(MODEL_CHECKPOINT_PATH, weights_only=True)
    except Exception as exc:
        logger.warning('Checkpoint load failed (%s) — will retrain.', exc)
        return None, None, None

    if ckpt.get('data_version') != data_version:
        logger.info('Dataset changed — checkpoint invalidated; retraining.')
        return None, None, None

    model = CarbonForecasterLSTM()
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    scaler = MinMaxScaler(feature_range=tuple(ckpt['scaler_feature_range']))
    # Reconstruct scaler internals without calling .fit()
    scaler.data_min_   = ckpt['scaler_data_min']
    scaler.data_max_   = ckpt['scaler_data_max']
    scaler.data_range_ = ckpt['scaler_data_range']
    scaler.scale_      = ckpt['scaler_scale']
    scaler.n_features_in_  = 1
    scaler.n_samples_seen_ = 1

    return model, scaler, ckpt.get('meta', {})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_sequences(scaled: np.ndarray):
    """Slice scaled series into overlapping (input, target) sequence pairs."""
    X, y = [], []
    n = len(scaled)
    for i in range(n - SEQUENCE_LENGTH - PRED_LENGTH + 1):
        X.append(scaled[i : i + SEQUENCE_LENGTH])
        y.append(scaled[i + SEQUENCE_LENGTH : i + SEQUENCE_LENGTH + PRED_LENGTH, 0])
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32),
    )


def _train(x_train, y_train, x_val, y_val) -> tuple[CarbonForecasterLSTM, dict]:
    """Train the LSTM with early stopping.  Returns (best_model, training_meta)."""
    torch.manual_seed(42)
    model = CarbonForecasterLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss  = float('inf')
    patience_left  = PATIENCE
    best_state     = None
    stopped_epoch  = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                stopped_epoch = epoch
                logger.info('Early stop at epoch %d (val_loss=%.6f)', epoch, best_val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    meta = {
        'stopped_epoch': stopped_epoch,
        'best_val_mse': round(best_val_loss, 8),
        'trained_at': datetime.now().isoformat(timespec='seconds'),
    }
    return model, meta


# ---------------------------------------------------------------------------
# Public forecast entry point
# ---------------------------------------------------------------------------

def run_forecast() -> dict:
    """Return the 24-hour carbon intensity forecast for India.

    Uses an mtime-keyed in-memory cache, then a disk checkpoint, then
    triggers a full retrain if neither is valid.

    Returns a dict with keys:
        forecast       — list of 24 floats (gCO2/kWh, hours 1-24)
        labels         — list of 24 label strings
        rmse           — float, validation RMSE in gCO2/kWh
        val_mse        — float, raw validation MSE
        stopped_epoch  — int
        forecast_source — 'cache' | 'checkpoint' | 'retrained'
        last_trained_at — ISO-8601 string
    """
    data_version = DATA_PATH.stat().st_mtime_ns

    # 1 — in-memory cache hit
    if data_version in _FORECAST_CACHE:
        result = _FORECAST_CACHE[data_version].copy()
        result['forecast_source'] = 'cache'
        return result

    # 2 — disk checkpoint hit
    model, scaler, saved_meta = _load_checkpoint(data_version)
    if model is not None:
        return _predict_and_cache(model, scaler, data_version,
                                  saved_meta, 'checkpoint')

    # 3 — full retrain
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)

    data     = pd.read_csv(DATA_PATH)
    ci       = _compute_ci_series(data)

    # Hourly interpolation from monthly data
    series = ci.resample('h').interpolate(method='time').dropna()

    # Keep the last 3 years of data (≈26,280 hourly points) for speed + relevance
    series = series.tail(26_280)
    if len(series) < SEQUENCE_LENGTH + PRED_LENGTH + 10:
        raise ValueError('Not enough historical observations to train the model.')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(series.to_numpy().reshape(-1, 1))

    X, y = _build_sequences(scaled)

    # Train/validation split (chronological — no shuffling for time-series)
    split   = int(len(X) * TRAIN_FRACTION)
    x_train, y_train = X[:split], y[:split]
    x_val,   y_val   = X[split:], y[split:]

    if len(x_val) == 0:
        raise ValueError('Validation set is empty — need more data.')

    model, train_meta = _train(x_train, y_train, x_val, y_val)

    # Compute RMSE on the validation set in original scale (gCO2/kWh)
    with torch.no_grad():
        val_pred_scaled = model(x_val).numpy().reshape(-1, 1)
        val_true_scaled = y_val.numpy().reshape(-1, 1)

    val_pred = scaler.inverse_transform(val_pred_scaled)
    val_true = scaler.inverse_transform(val_true_scaled)
    rmse     = float(np.sqrt(np.mean((val_pred - val_true) ** 2)))
    logger.info('Validation RMSE: %.2f gCO2/kWh  (epoch %d)',
                rmse, train_meta['stopped_epoch'])

    train_meta['rmse'] = round(rmse, 4)
    _save_checkpoint(model, scaler, train_meta, data_version)

    return _predict_and_cache(model, scaler, data_version, train_meta, 'retrained')


def _predict_and_cache(model: CarbonForecasterLSTM,
                       scaler: MinMaxScaler,
                       data_version: int,
                       meta: dict,
                       source: str) -> dict:
    """Run inference on the most recent 24-h window and cache the result."""
    data   = pd.read_csv(DATA_PATH)
    ci     = _compute_ci_series(data)
    series = ci.resample('h').interpolate(method='time').dropna().tail(26_280)

    scaled = scaler.transform(series.to_numpy().reshape(-1, 1))
    recent = torch.tensor(
        scaled[-SEQUENCE_LENGTH:], dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = torch.clamp(model(recent), -1, 1).numpy()

    forecast = np.round(
        scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1), 2
    )
    forecast = np.clip(forecast, _CI_MIN, _CI_MAX).tolist()

    result = {
        'state':          'India',
        'variable':       'Carbon intensity (gCO2/kWh)',
        'forecast':       [float(v) for v in forecast],
        'labels':         [f'Hour {i}' for i in range(1, PRED_LENGTH + 1)],
        'rmse':           meta.get('rmse', None),
        'val_mse':        meta.get('best_val_mse', None),
        'stopped_epoch':  meta.get('stopped_epoch', None),
        'last_trained_at': meta.get('trained_at', datetime.now().isoformat(timespec='seconds')),
        'forecast_source': source,
    }
    _FORECAST_CACHE[data_version] = result
    _record_prediction_history(result)
    return result.copy()


# ---------------------------------------------------------------------------
# Live observation ingestion
# ---------------------------------------------------------------------------

def record_live_observation(timestamp, value) -> None:
    """Append a single validated live observation to the training dataset."""
    ts  = _parse_and_validate(timestamp, value)
    val = float(value)
    _append_observation(ts, val)
    _FORECAST_CACHE.clear()


def record_live_sample(timestamp, value) -> dict:
    """Buffer live samples; write one aggregate per LIVE_AGGREGATION_MINUTES."""
    ts  = _parse_and_validate(timestamp, value)
    val = float(value)

    bucket = ts.floor(f'{LIVE_AGGREGATION_MINUTES}min')
    with _DATA_LOCK:
        _LIVE_BUCKETS.setdefault(bucket, []).append(val)
        closed = sorted(k for k in _LIVE_BUCKETS if k < bucket)

        aggregate = None
        agg_ts    = None
        for cb in closed:
            samples = _LIVE_BUCKETS.pop(cb)
            aggregate = float(np.mean(samples))
            agg_ts    = cb
            _append_observation(agg_ts, aggregate)

        if aggregate is not None:
            _FORECAST_CACHE.clear()

        return {
            'aggregated': aggregate is not None,
            'aggregation_interval_minutes': LIVE_AGGREGATION_MINUTES,
            'sample_count': len(_LIVE_BUCKETS[bucket]),
            'aggregate_timestamp': agg_ts.isoformat() if agg_ts else None,
            'aggregate_value': aggregate,
        }


def _parse_and_validate(timestamp, value) -> pd.Timestamp:
    ts = pd.to_datetime(timestamp, errors='raise')
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    val = float(value)
    if not np.isfinite(val) or val < 0:
        raise ValueError('value must be a finite non-negative number')
    return ts


def _append_observation(timestamp: pd.Timestamp, value: float) -> None:
    """Append a single row to the training CSV (thread-safe, deduplicates)."""
    with _DATA_LOCK:
        dataset = pd.read_csv(DATA_PATH)
        dataset['Date'] = pd.to_datetime(dataset.get('Date', dataset.get('Month', dataset.get('Year'))), errors='coerce')
        row = {col: '' for col in dataset.columns}
        row.update({
            'Date': timestamp, 'Country': 'India', 'State': 'India',
            'Variable': 'Total emissions', 'Value': value,
        })
        dup_mask = (
            (dataset['Date'] == timestamp) &
            (dataset['Country'] == 'India') &
            (dataset['Variable'] == 'Total emissions')
        )
        dataset = pd.concat(
            [dataset.loc[~dup_mask], pd.DataFrame([row])],
            ignore_index=True,
        )
        dataset.sort_values('Date').to_csv(
            DATA_PATH, index=False, date_format='%Y-%m-%dT%H:%M:%S'
        )


def _record_prediction_history(result: dict) -> None:
    generated_at = datetime.now().replace(second=0, microsecond=0)
    rows = pd.DataFrame({
        'generated_at': [generated_at] * len(result['forecast']),
        'forecast_for': [generated_at + timedelta(hours=i) for i in range(1, 25)],
        'Country':      result['state'],
        'Variable':     result['variable'],
        'Value':        result['forecast'],
        'RMSE':         result.get('rmse'),
        'Value_type':   'prediction',
    })
    rows.to_csv(
        PREDICTION_HISTORY_PATH,
        mode='a',
        header=not PREDICTION_HISTORY_PATH.exists(),
        index=False,
    )