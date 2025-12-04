from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

def merge_parquet_series(pathes: list[Path], out_path: Path):
    # First file: convert index to datetime and rename data column
    lfs = [
        pl.scan_parquet(pathes[0])
        .with_columns(pl.col("__index_level_0__").cast(pl.Duration).alias("datetime"))
        .drop("__index_level_0__")

        .rename({"0": pathes[0].stem.split("_")[-1]})
    ]

    # Rest: drop index (we already have it from first file), keep renamed data column
    for file in pathes[1:]:
        s = (
            pl.scan_parquet(file)
            .rename({"0": file.stem.split("_")[-1]})
            .drop("__index_level_0__")
        )
        lfs.append(s)

    result = pl.concat(lfs, how="horizontal")
    result.sink_parquet(out_path)


def merge_result_folder(result_folder_path: Path,
                        data_out_path: Path):

    # get file pathes
    result_folders = [p for p in result_folder_path.glob("*") if p.is_dir()]
    parquet_files = []
    for f in result_folders:
        parquet_files.extend([p for p in f.rglob("*.parquet")])

    # 1. merge parquet files
    merge_parquet_series(parquet_files, data_out_path)


if __name__ == "__main__":
    result_folder = Path(r"C:\Users\jkoslo\Documents\iOLE (lok)\programming\iole_current\data\networks\L-TOWN_DO_NOT_TOUCH_LPS_1year_04-12-2025_u2ukj5nt")
    data_out = Path("test_pq.parquet")

    merge_result_folder(result_folder, data_out)
