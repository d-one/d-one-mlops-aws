# Author:      CD4ML Working Group @ D ONE
# Description: Use this script to transform raw input data into a feature set (x)
#              and label set (y)
# ================================================================================

import logging
import pandas as pd
from typing import List, Tuple

pd.options.mode.chained_assignment = None  # default='warn'


_POWER = 0.05
_COL_POWER = 'power'
_COL_ERRORS = 'categories_sk'
_ERRORS_TO_CLASSIFY = [0, 3, 5, 7, 8]

def select_and_transform_data(
    df: pd.DataFrame,
    features: List[str],
    col_power: str = _COL_POWER,
    min_power: float = _POWER,
    col_erros: str = _COL_ERRORS,
    errors_to_classify: List[int] = _ERRORS_TO_CLASSIFY
    )-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects and transforms raw data into input features and labels.
    
    Function filters the dataframe to include only meaningful values of produced power
    using the column `col_power`. Nulls in the `features` columns are replaced with 0,
    only errors in the `errors_to_classify` are considered

    Args:
        df: Raw input data.
        features: List of the features to be included in the transformation.
        col_power: Column of `df` with the power production.
        min_power: Minimum values of power production considered. Rows with smaller
            values are filtered out.
        col_power: Column of `df` with the error codes.
        errors_to_classify: List of error types that should be considered.
    
    Returns:
        Tuple[pd.DataFrame]: Transformed dataframes (input features and labels).
    """
    # Filter power
    df = df.loc[ df[col_power] > min_power ]
    
    # Fill nulls in the featues with 0
    x = df[features].fillna(0)
    
    # Transform error types
    if 'categories_sk' in df.columns:
        df['categories_sk'] = (
            df['categories_sk']
            .fillna(0)
            .apply(lambda x: int(x))
        )
        df['categories_sk'] = (
            df['categories_sk']
            .apply(lambda x: x if x in errors_to_classify else 9)
        )
        y = df['categories_sk']
    else:
        y = pd.DataFrame(None)
    return x, y