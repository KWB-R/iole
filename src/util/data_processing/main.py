import pandas as pd

def try_read_parquet(path):
    try:
        data = pd.read_parquet(path)
        return data
    except Exception as e:
        print(e)

def reindex_freq(df: pd.DataFrame, f=pd.Timedelta(minutes=5)) -> pd.DataFrame:
    new_idx = pd.timedelta_range(
        df.first_valid_index(),
        df.last_valid_index(),
        freq=f
    )
    return df.loc[new_idx,:]

def reindex_cyclic(df: pd.DataFrame, new_idx: pd.Index) -> pd.DataFrame:
    n_reps = (len(new_idx) // len(df)) + 1
    _df = pd.concat(n_reps * [df], ignore_index=True, axis=0).iloc[:len(new_idx)]
    _df.index = new_idx

    return _df

def wrap_cyclic_patterns(
    pattern_df: pd.DataFrame,
    pattern_start_dayofweek: int,
    start_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """ Wraps a cyclic pattern to have it start at start_date day of week and time """

    if not isinstance(start_timestamp, pd.Timestamp):
        raise TypeError("No valid startdate provided.")

    if (pattern_start_dayofweek > 6) or (pattern_start_dayofweek < 0):
        raise ValueError("Start day has to be between 0 and 6.")

    if not isinstance(pattern_df.index, pd.TimedeltaIndex):
        raise TypeError("Pattern dataframe has no valid timedelta index.")

    if pattern_df.index[0] != pd.Timedelta(days=0):
        raise ValueError("Patterns have to start at 00:00:00.")

    # get start day of datetime_df
    dow = start_timestamp.dayofweek
    st = (start_timestamp - start_timestamp.normalize())

    dow_offset = (dow - pattern_start_dayofweek) % 7

    pattern_start_index = st + pd.Timedelta(days=dow_offset)

    wrapped_patterns = pd.concat([
        pattern_df.loc[pattern_start_index:],
        pattern_df.loc[:pattern_start_index - pd.Timedelta("1ns")]
    ]).set_index(pattern_df.index)

    return wrapped_patterns


def get_first_full_week(df: pd.DataFrame,
                        start_dow: int = 0,
                        exclusive: bool = True):

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Invalid index type.")

    dow = df.index.dayofweek

    num_idx = dow[dow.get_loc(start_dow)][0]

    if num_idx is None:
        raise ValueError(f"Weekday {start_dow} not detected.")

    full_week = df.iloc[num_idx:].loc[:df.index[num_idx] + pd.Timedelta(days=7)]

    assert (full_week.last_valid_index() - full_week.first_valid_index()) == pd.Timedelta(days=7), f"No full week in dataframe."

    if exclusive:
        return full_week.loc[:full_week.last_valid_index() - pd.Timedelta("1ns")]
    return full_week


