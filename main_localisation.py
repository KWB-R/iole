"""For parallised localisation on Rupelton"""

import json
import os
import argparse

import pandas as pd

from src.model import HydraulicNetwork, Localiser


def _test_timedelta(freq: str) -> pd.Timedelta:
    try:
        td = pd.to_timedelta(freq)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid pandas frequency string for -simdur: {freq!r} ({e})"
        )
    if pd.isna(td) or td <= pd.Timedelta(0):
        raise argparse.ArgumentTypeError(
            f"-simdur must be a positive pandas frequency string, got {freq!r}"
        )
    return freq


if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Run localisation")

    parser.add_argument("-tr", type=_test_timedelta)

    args = parser.parse_args()

    # setup
    dm_long_path: os.PathLike = r"data/networks/localisation_dual_model.inp"
    vfs_path: os.PathLike = r"data/datasets/localisation_virtual_flows.parquet"
    loc_periods_path: os.PathLike = r"data/datasets/localisation_periods.parquet"

    nw_slices_output_folder: os.PathLike = r"data/output/network_slices"

    data_output_folder: os.PathLike = r"data/output/data"

    # read long dm
    dm = HydraulicNetwork(dm_long_path)

    # read localisation periods
    with open(loc_periods_path, "r") as file:
        loc_periods = json.load(file)

        for k, v in loc_periods.items():
            for k1, v1 in v.items():
                v[k1] = pd.Timedelta(seconds=v1)

    vfs = pd.read_parquet(vfs_path)
    vps = vfs.columns.to_list()
    vf = vfs.sum(axis=1)

    # export slices
    ## 1. pattern slices
    ## 2. network slices
    loc_patterns = {}
    loc_pathes = {}
    patterns_long = dm.get_pattern_df()

    for k, v in loc_periods.items():
        s = v["start"]
        e = v["end"]
        pattern_slice = patterns_long.loc[s:e]
        vf_slice = vf.loc[s:e]

        _loc_data = {
            "patterns": pattern_slice,
            "virtual_flow": vf_slice,
        }

        loc_patterns[k] = _loc_data

        slice_path = os.path.join(nw_slices_output_folder, f"{k}.inp")
        dm.export_simulation_slice(s, e).save(slice_path)

        loc_pathes[k] = slice_path

    for pipe, path in loc_pathes.items():
        data_output_folder_ = os.path.join(data_output_folder, pipe)
        if not os.path.exists(data_output_folder_):
            os.mkdir(data_output_folder_)

        loc = Localiser(
            dual_model_inp_path=path,
            patterns=loc_patterns[pipe]["patterns"],
            virtual_flow=loc_patterns[pipe]["virtual_flow"],
            virtual_pipes=vps,
            aggregation="none",
        )

        if args.tr is None:
            TEMPORAL_RESOLUTION = "3 h"
        else:
            TEMPORAL_RESOLUTION = args.tr

        loc_result = loc.run(temporal_resolution=TEMPORAL_RESOLUTION)

        for test_pipe, (vf, head) in loc_result.items():
            path_vf = os.path.join(data_output_folder_, f"{test_pipe}_vf.parquet")
            path_head = os.path.join(data_output_folder_, f"{test_pipe}_head.parquet")

            vf.to_parquet(path_vf)
            head.to_parquet(path_head)
