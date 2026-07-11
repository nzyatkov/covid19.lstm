from datetime import datetime

covid_ml_data_file = "./data/covid_ml_data_Moscow.csv"

model_name = 'lstm_Moscow_model_2022-07-01.h5'
scaler_name = 'scaler_moscow.pkl'

nsample = 5
nsample_forward = 5

start_inference = datetime(2022, 6, 14)

sma_window = 7
targets = ['new_diagnoses_ema', 'Hospitalized_ema']

selected_features = [
    'new_diagnoses_ema',
    'Hospitalized_ema',

    'new_diagnoses_ema_log_3d',
    'new_diagnoses_ema_log_7d',
    'new_diagnoses_ema_log_14d',
    'Hospitalized_ema_log_3d',
    'Hospitalized_ema_log_7d',
    'Hospitalized_ema_log_14d',
    'new_deaths_ema_log_3d',
    'new_deaths_ema_log_7d',

    'new_cases_world_log_3d',
    'new_cases_world_log_7d',

    'new_cases_europe_log_3d',
    'new_cases_europe_log_7d',

    'asympt_percent_ema',
    'asympt_percent_ema_pc_7d',

    'IgG_ema',
    'IgG_ema_pc_7d',
    'IgG_ema_pc_14d',

    'yandex_index'
]

features_to_normalize = [
    'new_diagnoses_ema',
    'Hospitalized_ema'
]