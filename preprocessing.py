# Author:      CD4ML Working Group @ D ONE
# Description: Use this script to split the raw input data into a train and
#              test split
# ================================================================================

import logging
import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_train_test_split(df: pd.DataFrame, n_days_test: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the input data frame into a training and test set.

    Args:
        df: Raw input data.
        n_days_test: Number of days to consider for the test split. The n_days_test last 
            days of the input data will be selected for the test split

    Returns:
        Tuple[pd.DataFrame]: Raw train and test data splits.
    """
    # Take only the date part of the string, i.e., the first 10 characters
    df['date'] = df['measured_at'].apply(lambda x: x[:10])
    # Convert to date object
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    # Get the test dates
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    test_dates = [
        datetime.strftime(max_date - timedelta(days=i), '%Y-%m-%d') for i in range(n_days_test)
    ]
    
    df_train = df[~df['date'].isin(test_dates)].drop('date', axis=1)
    df_test = df[df['date'].isin(test_dates)].drop('date', axis=1)
    
    logger.info(f"Train set ranges from {min_date} until {min(test_dates)} (not included).")
    logger.info(f"Test set ranges from {min(test_dates)} until {max(test_dates)}.")
    
    return df_train, df_test