# Author:      CD4ML Working Group @ D ONE
# Description: Use this script for assertions
# ================================================================================

import logging
import pandas as pd
from typing import List, Union


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