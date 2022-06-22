# ================================================================================
# Author:      Heiko Kromer @ D ONE - 2022
# Description: This script contains the preprocessing logic for the windturbines
#              dataset.
# ================================================================================

import argparse
import logging
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Union

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

pd.options.mode.chained_assignment = None  # default='warn'

# ----------------- #
# --- Constants --- #
# ----------------- #

# S3 Bucket where the data is stored
BUCKET_NAME = "aws-sagemaker-blogpost"
BUCKET = f's3://{BUCKET_NAME}'

# Raw data paths
RAW_DATA_FOLDER = 'data'
RAW_DATA_FILE = 'wind_turbines.csv'
RAW_DATA_PATH = os.path.join(BUCKET, RAW_DATA_FOLDER, RAW_DATA_FILE)

# Path where the processed objects will be stored
now = datetime.now() # get current time to ensure uniqueness of the output folders
PROCESSED_DATA_FOLDER = 'processed_' + now.strftime("%Y-%m-%d_%H%M_%S%f")
PROCESSED_DATA_PATH = os.path.join(BUCKET, PROCESSED_DATA_FOLDER)

# Paths for model train, validation, test split
# We create one version of the preprocessed files with headers and one without
# headers. The implementation of XGBoost in AWS does not support headers.
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'train.csv')
TRAIN_DATA_PATH_W_HEADER = os.path.join(PROCESSED_DATA_PATH, 'train_w_header.csv')
VALIDATION_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'validation.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'test.csv')
TEST_DATA_PATH_W_HEADER = os.path.join(PROCESSED_DATA_PATH, 'test_w_header.csv')

# Constants for preprocessing
# Error column <> target
COL_ERRORS = 'subtraction'
# Power produced column (used for filtering out small values)
COL_POWER = 'power'
# Features to consider for the model
FEATURES = ['wind_speed', 'power', 'nacelle_direction', 'wind_direction',
            'rotor_speed', 'generator_speed', 'temp_environment',
            'temp_hydraulic_oil', 'temp_gear_bearing', 'cosphi',
            'blade_angle_avg', 'hydraulic_pressure']
# Power values to filter out
MIN_POWER = 0.049





def assert_col_of_df(df: pd.DataFrame, col: Union[List[str], str]) -> None:
    """Helper function to assert that a column `col` is a column of `df`.
    
    Args:
        df: Dataframe.
        col: String value to test.
    
    Returns:
        None.
        
    Raises:
        ValueError if `col` is not a column of `df`.
    """
    if isinstance(col, str):
        col = [col]

    for c in col:
        try:
            assert c in df.columns      
        except AssertionError:
            raise ValueError(f"Invalid input value. Column {c} is not a column of df.")

            
def get_train_test_split(
        df: pd.DataFrame,
        n_days_test: int
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the input data frame into a training and test set.

    Args:
        df: Raw input data.
        n_days_test: Number of days to consider for the test split. The n_days_test last 
            days of the input data will be selected for the test split.

    Returns:
        Tuple[pd.DataFrame]: Raw train and test data splits.
    """
    _date_col = 'date'
    _measured_at_col = 'measured_at'
    
    assert_col_of_df(df=df, col=_measured_at_col)
    
    # Take only the date part of the string, i.e., the first 10 characters
    df[_date_col] = df[_measured_at_col].apply(lambda x: x[:10])
    # Convert to date object
    df[_date_col] = pd.to_datetime(df[_date_col], format='%Y-%m-%d')
    
    # Get the test dates
    min_date = df[_date_col].min()
    max_date = df[_date_col].max()
    
    test_dates = [
        datetime.strftime(max_date - timedelta(days=i), '%Y-%m-%d') for i in range(n_days_test)
    ]
    
    df_train = df[~df[_date_col].isin(test_dates)].drop(_date_col, axis=1)
    df_test = df[df[_date_col].isin(test_dates)].drop(_date_col, axis=1)
    
    logger.info(f"Train set ranges from {min_date} until {min(test_dates)} (not included).")
    logger.info(f"Test set ranges from {min(test_dates)} until {max(test_dates)}.")
    
    return df_train, df_test


def fill_nulls(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Fills nulls in column `col` of dataframe `df`.
    
    Args:
        df: Raw input dataframe.
        col: Column of `df` with nulls filled with 0.
        
    Returns:
        Dataframe with nulls filled.
    """
    assert_col_of_df(df=df, col=col)
        
    df.loc[:, col] = (
        df
        .loc[:, col]
        .fillna(0)
    )
    
    logger.info(f"Filled nulls in column {col} with 0.")

    return df


def filter_power(df: pd.DataFrame, col_power: str, min_power: float) -> pd.DataFrame:
    """Filters the `df` on the power column `col_power`.
    
    
    Args:
        df: Raw input dataframe.
        col_power: Column of `df` with the power production.
        min_power: Minimum values of power production considered. Rows with smaller
            values are filtered out.
        
    Returns:
        Dataframe filtered on `min_power`.
    """
    assert_col_of_df(df=df, col=col_power)
    
    filter_power = df[col_power] > min_power
    
    rowcount_before = df.shape[0]
    df = (
        df
        .loc[filter_power]
    )
    rowcount_after = df.shape[0]
    logger.info(f"Removed {rowcount_before-rowcount_after} rows which had power below {min_power}.")
    
    return df


def process_target(df: pd.DataFrame, col_target: str):
    """Processes the target column by:
        1. Replacing error types 1 and 0 with 1.
        2. Filling nulls with 0.
        3. Make sure the target column is in the first column of the dataframe.
    
    Args:
        df: Raw input dataframe.
        col_target: Target column
    
    Returns:
        Dataframe with target column processed.
    """
    assert_col_of_df(df=df, col=col_target)
    
    # Make sure that the 0 error type is also mapped to 1 (we do a binary classification later)
    df.loc[df[col_target] == 0, col_target] = 1
    
    df = fill_nulls(df=df, col=col_target)
    
    # Reorder columns
    colnames = list(df.columns)
    colnames.insert(0, colnames.pop(colnames.index(col_target)))
    df = df[colnames]
    
    return df


def wrap_transform_data(
    df: pd.DataFrame,
    col_power: str,
    min_power: float,
    features: List[str],
    target: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wrapper for transforming the data for the model
    
    Processing is applied in the following steps:
        1. Filtering out low power values
        2. Process target column
        3. Fill nulls in all of the feature columns
        4. Select only relevant columns
        
    Args:
        df: Input dataframe.
        col_power: Column of `df` with the power production.
        min_power: Minimum values of power production considered. Rows with smaller
            values are filtered out.
        features: List of the features to be included in the transformation.
        target: Target column.
    
    Returns:
        
    """
    # 1. Filter out low power
    df = filter_power(df=df, col_power=col_power, min_power=min_power)
    
    # 2. Process target clumn
    df = process_target(df=df, col_target=target)

    # 3. Fill nulls in all of the feature columns and select them
    for feat in features:
        df = fill_nulls(df=df, col=feat)
    
    # 4. Select only relevant columns
    df = df[[target] + features]
    
    return df


# ----- CONSTANTS ----- #
# Columns of df
# Error column <> target
COL_ERRORS = 'subtraction'
# Power produced column (used for filtering out small values)
COL_POWER = 'power'
# Features to consider for the model
FEATURES = ['wind_speed', 'power', 'nacelle_direction', 'wind_direction',
            'rotor_speed', 'generator_speed', 'temp_environment',
            'temp_hydraulic_oil', 'temp_gear_bearing', 'cosphi',
            'blade_angle_avg', 'hydraulic_pressure']
# Power values to filter out
MIN_POWER = 0.05
# Filname of the raw data file
RAW_DATA_FILE = 'wind_turbines.csv'


if __name__ == '__main__':
    
    logger.info(f'Preprocessing job started.')
    # Parse the SDK arguments that are passed when creating the SKlearn container
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test_days", type=int, default=10)
    parser.add_argument("--n_val_days", type=int, default=10)
    args, _ = parser.parse_known_args()

    logger.info(f"Received arguments {args}.")

    # Read in data locally in the container
    input_data_path = os.path.join("/opt/ml/processing/input", RAW_DATA_FILE)
    logger.info(f"Reading input data from {input_data_path}")
    # Read raw input data
    df = pd.read_csv(input_data_path)
    logger.info(f"Shape of data is: {df.shape}")

    # ---- Preprocess the data set ----
    logger.info("Split data into training+validation and test set.")
    df_train_valid, df_test = get_train_test_split(df=df, n_days_test=args.n_test_days) 

    logger.info("Split training+validation into training and validation set.")
    df_train, df_val = get_train_test_split(df=df_train_valid, n_days_test=args.n_val_days) 

    logger.info("Transforming training data.")
    train = wrap_transform_data(
        df=df_train,
        col_power=COL_POWER,
        min_power=MIN_POWER,
        features=FEATURES,
        target=COL_ERRORS
    )
    
    logger.info("Transforming validation data.")
    val = wrap_transform_data(
        df=df_val,
        col_power=COL_POWER,
        min_power=MIN_POWER,
        features=FEATURES,
        target=COL_ERRORS
    )

    logger.info("Transforming test data.")
    test = wrap_transform_data(
        df=df_test,
        col_power=COL_POWER,
        min_power=MIN_POWER,
        features=FEATURES,
        target=COL_ERRORS
    )
    
    # Create local output directories. These directories live on the container that is spun up.
    try:
        os.makedirs("/opt/ml/processing/train")
        os.makedirs("/opt/ml/processing/validation")
        os.makedirs("/opt/ml/processing/test")
        print("Successfully created directories")
    except Exception as e:
        # if the Processing call already creates these directories (or directory otherwise cannot be created)
        logger.debug(e)
        logger.debug("Could Not Make Directories.")
        pass

    # Save data locally on the container that is spun up.
    try:
        pd.DataFrame(train).to_csv("/opt/ml/processing/train/train.csv", header=False, index=False)
        pd.DataFrame(train).to_csv("/opt/ml/processing/train/train_w_header.csv", header=True, index=False)
        pd.DataFrame(val).to_csv("/opt/ml/processing/validation/val.csv", header=False, index=False)
        pd.DataFrame(val).to_csv("/opt/ml/processing/validation/val_w_header.csv", header=True, index=False)
        pd.DataFrame(test).to_csv("/opt/ml/processing/test/test.csv", header=False, index=False)
        pd.DataFrame(test).to_csv("/opt/ml/processing/test/test_w_header.csv", header=True, index=False)
        logger.info("Files Successfully Written Locally")
    except Exception as e:
        logger.debug("Could Not Write the Files")
        logger.debug(e)
        pass

    logger.info("Finished running processing job")