"""Starts parallel simulation of all scenarios defined in the SensitivitySetup json"""

import os
import argparse
import pandas as pd
from src import SimulationFC, SensitivitySetup, Options


def _parse_timedelta(freq: str) -> pd.Timedelta:
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
    return td


def _parse_nonneg_int(text: str) -> int:
    try:
        val = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected integer, got {text!r}")
    if val < 0:
        raise argparse.ArgumentTypeError(f"Value must be >= 0, got {val}")
    return val


def main(
    inp_path: os.PathLike,
    options_path: os.PathLike,
    sensitivity_setup_path: os.PathLike,
    artificial_leaks_path: os.PathLike,
    # new optional overrides
    max_simulation_time: pd.Timedelta | None = None,
    stop_at: int | None = None,
    start_at: int | None = None,
):

    options = Options.from_json(options_path)
    setup = SensitivitySetup.from_json(sensitivity_setup_path)

    # Apply CLI overrides if provided
    if max_simulation_time is not None:
        options.max_simulation_time = max_simulation_time

    if stop_at is not None:
        setup.stop_at = stop_at

    if start_at is not None:
        setup.resume_at = start_at

    # Final validation: start_at cannot exceed stop_at
    # (If either is missing from CLI, we validate using values from JSON / DEBUG / above)
    if hasattr(setup, "start_at") and hasattr(setup, "stop_at"):
        if setup.resume_at is None:
            setup.resume_at = 0  # sensible default if missing
        if setup.stop_at is not None:
            if setup.resume_at > setup.stop_at:
                raise ValueError(
                    f"start_at ({setup.resume_at}) cannot exceed stop_at ({setup.stop_at})."
                )

    simulation = SimulationFC(
        source_inp=inp_path,
        artificial_leaks_path=artificial_leaks_path,
        options=options,
        sensitivity_setup=setup,
    )

    simulation.run()


## MAIN CALL
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sensitivity analysis simulation")

    # New CLI options
    parser.add_argument(
        "-sim_dur",
        type=_parse_timedelta,
        help="Simulation duration as pandas frequency string, e.g. 1d, 12h, 30min",
    )
    parser.add_argument(
        "-stop_at",
        type=_parse_nonneg_int,
        help="Last scenario index to simulate (integer, inclusive)",
    )
    parser.add_argument(
        "-start_at",
        type=_parse_nonneg_int,
        help="First scenario index to simulate (integer, must be <= -stop_at)",
    )

    args = parser.parse_args()

    to_unix_abspath = lambda path: os.path.abspath(os.path.normpath(path))

    opt_path = to_unix_abspath(r"data/configuration/options_no_repl.json")
    senset_path = to_unix_abspath(r"data/configuration/sensitivity_setup_demo_data.json")
    inp_path = to_unix_abspath(r"data/networks/L-TOWN_DO_NOT_TOUCH_LPS_1year.inp")
    leak_csv_path = to_unix_abspath(
        r"data/artificial_leakages/LEAK_PATTERNS_SENSITIVITY_2y.csv"
    )

    # Enforce start_at <= stop_at at CLI level if both provided
    if args.start_at is not None and args.stop_at is not None:
        if args.start_at > args.stop_at:
            parser.error(
                f"-start_at ({args.start_at}) cannot exceed -stop_at ({args.stop_at})."
            )

    # Run
    main(
        inp_path,
        opt_path,
        senset_path,
        leak_csv_path,
        max_simulation_time=args.sim_dur,
        stop_at=args.stop_at,
        start_at=args.start_at,
    )
