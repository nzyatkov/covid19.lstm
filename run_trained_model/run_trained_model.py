import importlib

import numpy as np
import pandas as pd
from datetime import timedelta
import tensorflow as tf
from keras.models import load_model
from pathlib import Path
import plotly.graph_objects as go

from data_processing.region_type import RegionType
from data_processing.rnn_data_processing import RnnDataProcessing
from logger.logger import get_logger

def run_trained_model(region_type: RegionType):
    _logger = get_logger()

    DATA_PATH = Path('data')

    if region_type == RegionType.Nso:
        CONSTANTS = importlib.import_module('run_trained_model.constants.constants_nso')
    elif region_type == RegionType.Spb:
        CONSTANTS = importlib.import_module('run_trained_model.constants.constants_spb')
    elif region_type == RegionType.Moscow:
        CONSTANTS = importlib.import_module('run_trained_model.constants.constants_moscow')
    else:
        _logger.error('Region type not supported')
        raise ValueError(f"Unknown region type: {region_type}")

    # Parameters
    covid_ml_data_file = CONSTANTS.covid_ml_data_file
    scaler_name = CONSTANTS.scaler_name
    nsample = CONSTANTS.nsample
    nsample_forward = CONSTANTS.nsample_forward
    start_inference = CONSTANTS.start_inference
    sma_window = CONSTANTS.sma_window
    targets = CONSTANTS.targets
    selected_features = CONSTANTS.selected_features
    features_to_normalize = CONSTANTS.features_to_normalize

    targets_days = [d for d in range(1, nsample_forward + 1)]
    target_vars = [None] * len(targets)
    for i in range(len(targets)):
        target_vars[i] = [f'target_{targets[i]}_log_diff_sma{sma_window}_ch_{t}d' for t in targets_days]

    target_vars_flatten = list(np.array(target_vars).reshape(-1))

    # Choosing a device for computing
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        _logger.info('Using GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        _logger.info('Using CPU')

    # Load model
    model_path = DATA_PATH / Path('model', f'{region_type.name.lower()}_models')
    rnn = load_model(
        model_path / CONSTANTS.model_name,
        compile=False
    )
    rnn.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Preparing data in RNN format
    X, y1, y2 = RnnDataProcessing.prepare_rnn_dataset(covid_ml_data_file, model_path / scaler_name,
                                                      selected_features, features_to_normalize,
                                                      target_vars, target_vars_flatten,
                                                      start_inference, nsample, nsample_forward)


    # =============================================================================
    # Forecast
    all_y = [None] * len(targets)
    all_y_predicted = [None] * len(targets)

    y_predicted = rnn.predict(np.stack(X.values), verbose=0)
    _logger.info(f'Forecast for {region_type.name} done, len = {len(y_predicted)}')

    all_y_predicted[0] = pd.DataFrame(data=y_predicted[0], index=y1.index)
    all_y[0] = pd.DataFrame(data=np.stack(y1.values), index=y1.index, columns=target_vars[0])

    all_y_predicted[1] = pd.DataFrame(data=y_predicted[1], index=y2.index)
    all_y[1] = pd.DataFrame(data=np.stack(y2.values), index=y2.index, columns=target_vars[1])

    # =============================================================================

    results_path = DATA_PATH / Path('results') / f'{region_type.name.lower()}_results'
    results_path.mkdir(parents=True, exist_ok=True)

    # Plotting true and pred graphs
    for j in range(len(targets)):
        for i, target_var in enumerate(target_vars[j]):
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=all_y[j].index,
                                     y=all_y[j][target_var],
                                     name=f'true F(t+{i+1})'))
            fig.add_trace(go.Scatter(x=all_y_predicted[j].index,
                                     y=all_y_predicted[j][i],
                                     name=f'predicted F(t+{i+1})'))

            fig.update_layout(template='plotly_white',
                              title=f'{targets[j]}_log_diff, days forward = {targets_days[target_vars[j].index(target_var)]}')

            fig.write_image(results_path / f'{targets[j]}_log_diff_F(t+{i + 1}).png',
                            width=1000,
                            height=500,
                            scale=1,
                            format="png")

            _logger.info(f'{region_type.name} {targets[j]}_log_diff_F(t+{i + 1}).png successfully saved')

    # =============================================================================
    # Inverse transformation MinMaxScaler

    _logger.info(f'Start MinMaxScaler inverse transform for {region_type.name}...')

    y_pred_transformed = RnnDataProcessing.pred_inverse_transform(targets, target_vars, all_y, all_y_predicted)

    _logger.info(f'MinMaxScaler inverse transform for {region_type.name} done.')

    # =============================================================================
    # Recovering real forecasts from stationary targets

    _logger.info(f'Start real forecasts f transform from targets F for {region_type.name}...')

    forecast = RnnDataProcessing.transform_from_stationary(y_pred_transformed, targets, targets_days,
                                                           sma_window, nsample_forward)

    _logger.info(f'Real forecasts f transform from targets F for {region_type.name} done.')

    # =============================================================================
    # Plot real data and forecasts
    for i in range(len(targets)):
        new_diagnoses_addition = pd.DataFrame(index=pd.date_range(start=forecast[i].index[-1] + timedelta(days=1),
                                                                  end=forecast[i].index[-1] + timedelta(days=45)),
                                              columns=[f'{targets[i]}_{j}d' for j in targets_days])

        forecast_all = pd.concat([forecast[i], new_diagnoses_addition])

        for j, target_var in enumerate(targets_days):

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=forecast_all[f'{targets[i]}_{target_var}d'].shift(target_var).index,
                                     y=forecast_all[f'{targets[i]}_{target_var}d'].shift(target_var),
                                     name=f'{targets[i]}_forecast_{target_var}d'))

            fig.add_trace(go.Scatter(x=RnnDataProcessing.data[targets[i]].index,
                                     y=RnnDataProcessing.data[targets[i]],
                                     line_color='rgba(229,90,66,0.8)',
                                     name=targets[i]))

            fig.update_layout(template='plotly_white',
                              title=f'Forecast {targets[i]} for {region_type.name} region for the day '
                                    f't+{target_var} from {y1.index[0]}',
                              legend_orientation="v",
                              legend=dict(x=0.5, xanchor="center"))

            fig.write_image(results_path / f'{targets[i]}_f(t+{j + 1}).png',
                            width=1000,
                            height=500,
                            scale=1,
                            format="png")

            _logger.info(f'{region_type.name} {targets[i]}_f(t+{j + 1}).png successfully saved')
