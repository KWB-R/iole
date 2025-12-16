from typing import Literal
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

def wrap_cyclic_dataframe(
    pattern_df: pd.DataFrame,
    pattern_start_dayofweek: int,
    start_timestamp: pd.Timestamp,
    reset_index: bool = True,
) -> pd.DataFrame:
    """ Wraps a cyclic pattern to have it start at start_date day of week and time """

    if not isinstance(start_timestamp, pd.Timestamp):
        raise TypeError(f"No valid startdate provided (expected pd.Timestamp, got {start_timestamp} of type {type(start_timestamp)}).")

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
    ])

    if reset_index:
        wrapped_patterns = wrapped_patterns.set_index(pattern_df.index)

    return wrapped_patterns

def week_means(df: pd.DataFrame, first_dow: int = 0, adjust_index: bool = True) -> pd.DataFrame:
    """Takes dataframe and builds mean values for every dayofweek, hour and minute

    Args:
        df: datetime dataframe to be aggregated
        first_dow: 0==Monday, 6==Sunday
        adjust_index: True if index should start at Timedelta(0), else index day is not ordered if first_dow != 0
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"Invalid index type {type(df.index).__name__}, expected pd.DatetimeIndex"
        )

    if df.index.inferred_freq == None:
        raise ValueError("No inferrable frequency of data.")

    # group by dayofweek, hour and minute
    _df = df.copy().groupby([df.index.dayofweek, df.index.hour, df.index.minute]).mean()

    # 3 levels?
    if _l:=len(_df.index.names) != 3:
        raise ValueError(f"Invalid number of index levels ({_l}), expected 3.")

    # reorder so first day == start day of correction pattern
    day_order = list(range(0,7))[first_dow:] + list(range(0,7))[:first_dow]

    _df = _df.loc[pd.IndexSlice[day_order,:,:]]

    # restore timedelta
    _df.index = pd.to_timedelta(
        _df.index.get_level_values(0) * 60 * 60 * 24
        + _df.index.get_level_values(1) * 60 * 60
        + _df.index.get_level_values(2) * 60,
        unit="s",
    )

    # convert to day range 0..7
    if adjust_index:
        _df.index = (_df.index - pd.Timedelta(days=first_dow)) % pd.Timedelta(days=7)

    return _df

def calculate_correction_flows(
        df: pd.DataFrame,
        start_dow: int = 0,
        adjust_index: bool = True,
        period: Literal["day", "week"] = "week"):

    match period:
        case "week":
            return week_means(df, start_dow, adjust_index)
        case "day":
            raise NotImplementedError("Only weekly correction flow patterns supported at this time.")
