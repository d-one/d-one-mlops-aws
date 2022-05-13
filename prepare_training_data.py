from preprocessing import (
    get_train_test_split,
    wrap_transform_data
)
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# ----- CONSTANTS ----- #
# Columns of df
# Error column <> target
COL_ERRORS = 'categories_sk'
# Power produced column (used for filtering out small values)
COL_POWER = 'power'
# Error (i.e., target) classification list
ERRORS_TO_CLASSIFY = [0, 3, 5, 7, 8]
# Features to consider for the model
FEATURES = ['wind_speed', 'power', 'nacelle_direction', 'wind_direction',
            'rotor_speed', 'generator_speed', 'temp_environment',
            'temp_hydraulic_oil', 'temp_gear_bearing', 'cosphi',
            'blade_angle_avg', 'hydraulic_pressure']
# Power values to filter out
MIN_POWER = 0.05
# Number of days in the (hidden) test set
N_TEST_DAYS = 20
# Number of days in the validation set
N_VAL_DAYS = 20



# Read raw input data
df = pd.read_csv(RAW_DATA_PATH)

# Split data into training+validation and test set
df_train_valid, df_test = get_train_test_split(df=df, n_days_test=N_TEST_DAYS) 

# Split training+validation into training and validation set
df_train, df_val = get_train_test_split(df=df, n_days_test=N_VAL_DAYS) 

x_train, y_test = wrap_transform_data(
    df=df_train,
    col_power=COL_POWER,
    min_power=MIN_POWER,
    col_errors=COL_ERRORS,
    errors_to_classify=ERRORS_TO_CLASSIFY,
    features=FEATURES,
    target=COL_ERRORS
)

x_val, y_val = wrap_transform_data(
    df=df_val,
    col_power=COL_POWER,
    min_power=MIN_POWER,
    col_errors=COL_ERRORS,
    errors_to_classify=ERRORS_TO_CLASSIFY,
    features=FEATURES,
    target=COL_ERRORS
)