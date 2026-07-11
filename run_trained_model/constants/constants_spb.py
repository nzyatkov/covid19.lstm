from datetime import datetime

covid_ml_data_file = "./data/covid_ml_data_Spb.csv"

model_name = 'lstm_Spb_model_2022-07-01.h5'
scaler_name = 'scaler_spb.pkl'

nsample = 14
nsample_forward = 5

start_inference = datetime(2020, 6, 14)

sma_window = 7
targets = ['new_diagnoses_tsa', 'hospitalized_tsa']

selected_features = [
    'new_diagnoses_ema',
    'new_cases_world_minus_china',
    'new_diagnoses__new_tests_ema',

    'new_diagnoses_ema_log_3d',
    'new_diagnoses_ema_log_7d',
    'new_diagnoses_ema_log_10d',

    'new_diagnoses__new_tests_ema_log_3d',
    'new_diagnoses__new_tests_ema_log_7d',
    'new_diagnoses__new_tests_ema_log_10d',

    'hospitalized_log_3d',
    'hospitalized_log_7d',
    'hospitalized_log_10d',

    'new_deaths_ema_log_21d',

    'new_cases_world_minus_china_log_7d',
    'new_cases_world_minus_china_log_14d',
    'new_cases_world_minus_china_log_21d',
    'new_cases_world_minus_china_log_28d',
    'new_cases_world_minus_china_log_42d',

    'IgG_ema',

    'yandex_index'
]

features_to_normalize = [
    'new_diagnoses_ema',
    'new_cases_world_minus_china',
    'new_diagnoses__new_tests_ema'
]