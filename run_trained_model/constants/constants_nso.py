from datetime import datetime

covid_ml_data_file = "./data/covid_ml_data_NSO.csv"

model_name = 'lstm_NSO_model_2022-07-01.h5'
scaler_name = 'scaler_nso.pkl'

nsample = 21
nsample_forward = 5

start_inference = datetime(2020, 7, 12)

sma_window = 7
targets = ['new_diagnoses', 'ventilation_ema']

selected_features = [
    'new_diagnoses',
    'hospitalised',
    'ventilation_ema',
    'new_deaths_burial_ema',

    'new_diagnoses_log_3d',
    'new_diagnoses_log_7d',
    'new_diagnoses_log_14d',
    'hospitalised_log_3d',
    'hospitalised_log_7d',
    'hospitalised_log_14d',
    'ventilation_ema_log_3d',
    'ventilation_ema_log_7d',
    'ventilation_ema_log_14d',
    'new_deaths_burial_ema_log_3d',
    'new_deaths_burial_ema_log_7d',
    'new_deaths_burial_ema_log_14d',

    'asympt_percent_ema',
    'asympt_percent_ema_pc_3d',
    'asympt_percent_ema_pc_7d',

    'IgG_ema',
    'IgG_ema_pc_7d',
    'IgG_ema_pc_14d',

    'weekday_0',
    'weekday_1',
    'weekday_2',
    'weekday_3',
    'weekday_4',
    'weekday_5',
    'weekday_6',

    'yandex_index'
]

features_to_normalize = [
    'new_diagnoses',
    'hospitalised',
    'ventilation_ema',
    'new_deaths_burial_ema'
]