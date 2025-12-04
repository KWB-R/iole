from typing import Literal, get_args, Optional

import pandas as pd
import numpy as np
from . import Options

perturbation_target = Literal["roughness", "demand"]


# OLD
# def apply_uniform_noise(
#     df: pd.DataFrame,
#     noise_pctg: float,
#     rng: np.random.Generator,
#     force_mean: bool = True,
# ) -> pd.DataFrame:
#     assert isinstance(df, pd.DataFrame), f"Only DataFrames allowed as function input."
#     assert all(
#         dt == float for dt in df.dtypes.unique()
#     ), f"Dataframe contains non-float columns."

#     factors = rng.uniform(1 - noise_pctg, 1 + noise_pctg, size=df.shape)

#     df_pert = df.mul(factors)

#     if force_mean:
#         pert_mean = df_pert.mean().mean()
#         pre_mean = df.mean().mean()

#         df_pert = df_pert * pre_mean / pert_mean

#     return df_pert


# NEW 100% equal to TUB noise
def apply_uniform_noise(
    df: pd.DataFrame,
    noise_pctg: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame), f"Only DataFrames allowed as function input."
    assert all(
        dt == float for dt in df.dtypes.unique()
    ), f"Dataframe contains non-float columns."

    if noise_pctg == 0:
        return df

    lb = df.values * (1 - noise_pctg)
    ub = df.values * (1 + noise_pctg)

    sampled = pd.DataFrame(rng.uniform(lb, ub), columns=df.columns, index=df.index)

    return sampled


def apply_uniform_noise2(
    df: pd.DataFrame,
    noise_pctg: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Needed for correct behaviour under Ubuntu
    Windows numpy allows nan values, ubuntu doesnt."""

    if noise_pctg == 0:
        return df

    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Only DataFrames allowed as function input.")
    if not all(np.issubdtype(dt, np.floating) for dt in df.dtypes):
        raise TypeError("DataFrame contains non-float columns.")
    if noise_pctg < 0:
        raise ValueError("noise_pctg must be non-negative.")

    # Compute bounds
    vals = df.values  # float ndarray
    lb = vals * (1 - noise_pctg)
    ub = vals * (1 + noise_pctg)

    # Build masks
    finite_mask = np.isfinite(lb) & np.isfinite(ub)
    proper_range_mask = ub > lb  # strict: avoid equal bounds edge-case
    valid = finite_mask & proper_range_mask

    # Prepare output
    out = vals.copy()

    # Sample only where valid
    if np.any(valid):
        out_flat = out.ravel()
        lb_flat = lb.ravel()
        ub_flat = ub.ravel()
        valid_flat = valid.ravel()

        out_flat[valid_flat] = rng.uniform(
            lb_flat[valid_flat], ub_flat[valid_flat], size=valid_flat.sum()
        )

    # Where invalid (NaNs, infs, ub<=lb), keep original values as-is (including NaNs)
    # Optionally: if you prefer to keep exactly-equal bounds (noise_pctg==0) perturbed 0,
    # you could set out[~valid & (ub==lb)] = lb[~valid & (ub==lb)]

    return pd.DataFrame(out, columns=df.columns, index=df.index)


def perturbate_df(
    df: pd.DataFrame,
    options: Options,
    target: Optional[perturbation_target] = None,
):
    # case to df if applies
    if isinstance(df, pd.Series):
        _df = df.copy().to_frame()
    elif isinstance(df, pd.DataFrame):
        _df = df.copy()
    else:
        raise Exception(f"Invalid type {type(df)} of provided arg.")

    # get rng and mode
    rng = options.default_rng()
    mode = options.perturbation_mode

    pre_mean = df.mean() if len(_df.columns) == 1 else _df.mean().mean()

    match mode:
        case "uniform":
            factors = rng.uniform(
                1 - options.perturbation_pctg,
                1 + options.perturbation_pctg,
                size=df.shape,
            )

            df_pert = _df.mul(factors)
            pert_mean = (
                df_pert.mean() if len(_df.columns) == 1 else df_pert.mean().mean()
            )

            # new mean == old mean
            df_pert = df_pert * pre_mean / pert_mean

            return df_pert

        case "multiplier":
            if target is None:
                raise Exception(f"If mode is {mode} target cannot be None.")
            match target:
                case "demand":
                    mult = options.demand_multiplier
                case "roughness":
                    mult = options.roughness_multiplier
                case _:
                    raise Exception("Invalid perturbation target.")

            return _df * mult

        case _:
            raise Exception("Invalid mode.")
