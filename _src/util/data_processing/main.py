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