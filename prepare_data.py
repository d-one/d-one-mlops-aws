
import argparse
import logger
import pandas as pd
from typing import List, Tuple, Union
from datetime import datetime, timedelta

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

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


def transform_error_types(
    df: pd.DataFrame,
    col_errors: str,
    errors_to_classify: List[int]
    )-> pd.DataFrame:
    """Transforms error types.
    
    Error types are classified according to the `errors_to_classify`. If the error
    type is not in the list, it will be replaced with 9. Nulls are replaced with
    0.

    Args:
        df: Raw input data.
        col_errors: Column of `df` with the error types.
        errors_to_classify: List of error types that should be considered.
    
    Returns:
        Dataframe with transformed error types.
    """
    assert_col_of_df(df=df, col=col_errors)

    df[col_errors] = (
        df
        .loc[:, col_errors]
        .fillna(0)
        .apply(lambda x: int(x))
        .apply(lambda x: x if x in errors_to_classify else 9)
    )
    logger.info(f"Transformed error types in column {col_errors}.")

    return df


def split_features_target(
    df: pd.DataFrame,
    features: List[str],
    target: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function to split the dataframe into input features and labels.
    
    Args:
        df: Raw input data.
        features: List of the features to be included in the transformation.
        target: Target column.
    
    Returns:
        Tuple[pd.DataFrame]: Transformed dataframes (input features and labels).
    """
    assert_col_of_df(df=df, col=(features + [target]))

    y = df.loc[:, target]
    x = df.loc[:, features]
    logger.info(f"Split dataframe into features with shape {x.shape} and target with shape {y.shape}.")
    return x, y


def wrap_transform_data(
    df: pd.DataFrame,
    col_power: str,
    min_power: float,
    col_errors: str,
    errors_to_classify: List[int],
    features: List[str],
    target: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wrapper for transforming the data for the model
    
    Processing is applied in the following steps:
        1. Filtering out low power values
        2. Transforming error types
        3. Fill nuls in all of the feature columns
        4. Split dataset into features and target (x, y)
        
    Args:
        df: Input dataframe.
        col_power: Column of `df` with the power production.
        min_power: Minimum values of power production considered. Rows with smaller
            values are filtered out.
        col_errors: Column of `df` with the error types.
        errors_to_classify: List of error types that should be considered.
        features: List of the features to be included in the transformation.
        target: Target column.
    
    Returns:
        
    """
    # 1. Filter out low power
    df = filter_power(df=df, col_power=col_power, min_power=min_power)

    # 2. Transform error types
    df = transform_error_types(df=df, col_errors=col_errors, errors_to_classify=errors_to_classify)

    # 3. Fill nulls in all of the feature columns
    for feat in features:
        df = fill_nulls(df=df, col=feat)

    # 4. Split dataset into features and target
    x, y = split_features_target(df=df, features=features, target=target) 
    
    return x, y


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
# Filname of the raw data file
RAW_DATA_FILE = 'wind_turbines.csv'


if __name__ == '__main__':
    
    logger.debug(f'Preprocessing job started.')
    # Parse the SDK arguments that are passed when creating the SKlearn container
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test_days", type=int, default=10)
    parser.add_argument("--n_val_days", type=int, default=10)
    args, _ = parser.parse_known_args()

    logger.debug(f"Received arguments {args}.")

    # Read in data locally in the container
    input_data_path = os.path.join("/opt/ml/processing/input", RAW_DATA_FILE)
    logger.debug(f"Reading input data from {input_data_path}")
    # Read raw input data
    df = pd.read_csv(input_data_path)
    logger.debug(f"Shape of data is:{df.shape}")

    # ---- Preprocess the data set ----
    logger.debug("Split data into training+validation and test set.")
    df_train_valid, df_test = get_train_test_split(df=df, n_days_test=args.n_test_days) 

    logger.debug("Split training+validation into training and validation set.")
    df_train, df_val = get_train_test_split(df=df, n_days_test=args.n_val_days) 

    logger.debug("Transforming training data.")
    x_train, y_train = wrap_transform_data(
        df=df_train,
        col_power=COL_POWER,
        min_power=MIN_POWER,
        col_errors=COL_ERRORS,
        errors_to_classify=ERRORS_TO_CLASSIFY,
        features=FEATURES,
        target=COL_ERRORS
    )
    
    logger.debug("Transforming validation data.")
    x_val, y_val = wrap_transform_data(
        df=df_val,
        col_power=COL_POWER,
        min_power=MIN_POWER,
        col_errors=COL_ERRORS,
        errors_to_classify=ERRORS_TO_CLASSIFY,
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
        pd.DataFrame(x_train).to_csv("/opt/ml/processing/train/x_train.csv", header=True, index=False)
        pd.DataFrame(y_train).to_csv("/opt/ml/processing/train/y_train.csv", header=True, index=False)
        pd.DataFrame(x_val).to_csv("/opt/ml/processing/validation/x_val.csv", header=True, index=False)
        pd.DataFrame(y_val).to_csv("/opt/ml/processing/validation/y_val.csv", header=True, index=False)
        pd.DataFrame(df_test).to_csv("/opt/ml/processing/test/test.csv", header=True, index=False)
        logger.debug("Files Successfully Written Locally")
    except Exception as e:
        logger.debug("Could Not Write the Files")
        logger.debug(e)
        pass

    logger.debug("Finished running processing job")
