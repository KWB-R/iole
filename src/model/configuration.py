from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pprint import pformat
import json
import os

from typing import List, Dict, Optional, Literal, Iterable, Set, Tuple, Self, Any

from itertools import product, count, islice

import pandas as pd
import numpy as np


ARTIFICIAL_LEAK_PREFIX: str = "Leak_"
SPLIT_PIPE_SUFFIX: str = "_split"

PERTURBATION_MODE = Literal["uniform", "multiplier"]
PERTURBATION_TARGET = Literal["demand", "roughness", "measurement"]
DEMAND_PERTURBATION_METHOD = Literal["base", "dyn_base", "rng_patterns"]
MEASUREMENT_TYPE = Literal["head", "inflow", "pump_flow"]


@dataclass
class _ExcludedElements:
    nodes: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)


@dataclass
class Options:
    """TODO:
    - funnel the placement scenario into Options so it has a representation in each Worker
    - or better: Split Options into general Options and WorkerOptions?
    """

    # get flow from pipes:
    flow_pipe_ids: List[str] = field(default_factory=list)

    # pump ids
    pump_ids: List[str] = field(default_factory=list)

    # replacement mapping
    inflow_replacement_mapping: Dict[str, str] = field(
        default_factory=dict
    )  # {reservoir_id: connected_pipe}; if missing this step will be skipped and the network won't be split

    # Perturbation
    perturbation_mode: Dict[PERTURBATION_TARGET, PERTURBATION_MODE] = field(
        default_factory=lambda: {
            "demand": "uniform",
            "roughness": "uniform",
            "measurement": "uniform",
        },
        init=True,
    )

    # how is the demand perturbation done?
    demand_perturbation_method: DEMAND_PERTURBATION_METHOD = "base"
    rng_pattern_count: int = 25  # only demand_perturbation_mode "rng_patterns"
    rng_pattern_names: list[str] = field(
        default_factory=lambda: [
            "P-Residential",
            "P-Commercial",
        ],
        init=True,
    )  # only demand_perturbation_mode "rng_patterns"
    rng_skip_constant_patterns: bool = (
        True  # only demand_perturbation_mode "rng_patterns"
    )

    # demand perturbation update frequency in seconds (once (==0) versus dynamic(>=1))
    demand_perturbation_frequency: int = field(default=0, init=True)

    # which measurements to perturbate
    perturated_measurements: dict[MEASUREMENT_TYPE, bool] = field(
        default_factory=lambda: {
            "head": True,
            "inflow": False,
            "pump_flow": False,
        },
        init=True,
    )

    # Perturbation factors
    sensor_accuracy_multiplier: float = 1
    roughness_multiplier: float = 1
    demand_multiplier: float = 1

    #  Perturbation noise percentage
    # ε used to determine U(1-ε, 1+ε)[data.shape] * data
    sensor_accuracy_noise: float = 0
    roughness_noise: float = 0
    demand_noise: float = 0

    # cusum
    cusum_multiplier: float = 1

    # sensor decimals
    head_rounding_decimals: int = 2

    # for sensitivity
    use_dual_model_correction: bool = True
    run_cusum: bool = False
    aggregate_virtual_flows: bool = True
    save_virtual_flows: bool = False
    save_sensitivity_parameter_json: bool = True

    # default seed
    rng_seed: int = 42

    # time options
    max_simulation_time: Optional[pd.Timedelta] = None
    correction_period: pd.Timedelta = pd.Timedelta(days=7) - pd.Timedelta("1ns")
    calibration_week: int = field(default=1, init=True)

    # etc
    artificial_leak_prefix = ARTIFICIAL_LEAK_PREFIX
    split_pipe_suffix = SPLIT_PIPE_SUFFIX
    timedelta_to_datetime_conversion_date: pd.Timestamp = pd.Timestamp("2019-01-01")

    # excluded elements
    excluded_elements: _ExcludedElements = field(
        default_factory=_ExcludedElements, init=False
    )

    # debug/storage options
    keep_results: bool = False
    keep_workers: bool = False
    delete_worker_models: bool = True

    def __post_init__(self):
        if self.keep_workers:
            print("Warning! Keeping workers with all their data is memory intensive.")

        assert (
            self.demand_perturbation_frequency >= 0
        ), f"Base demand update frequency has to be greater than 0 (is {self.demand_perturbation_frequency})."

    def __repr__(self):
        return pformat(self.__dict__)

    def copy(self):
        return deepcopy(self)

    def set_perturbation_value(
        self, new_value: float, target: PERTURBATION_TARGET, mode: PERTURBATION_MODE
    ):
        _t = str(target)

        match target:
            case "demand":
                attr0 = _t
            case "measurement":
                attr0 = "sensor_accuracy"
            case "roughness":
                attr0 = _t

        match mode:
            case "multiplier":
                attr1 = "multiplier"
            case "uniform":
                attr1 = "noise"

        attr = "_".join([attr0, attr1])
        setattr(self, attr, new_value)

    @property
    def excluded_perturbation_nodes(self):
        return self.excluded_elements.nodes

    @property
    def excluded_perturbation_pipes(self):
        return self.excluded_elements.links

    @property
    def mult_map(self):
        return {
            "demand": self.demand_multiplier,
            "roughness": self.roughness_multiplier,
            "measurement": self.sensor_accuracy_multiplier,
        }

    @property
    def noise_map(self):
        return {
            "demand": self.demand_noise,
            "roughness": self.roughness_noise,
            "measurement": self.sensor_accuracy_noise,
        }

    @property
    def perturbate_head(self) -> bool:
        return self.perturated_measurements["head"]

    @property
    def perturbate_inflow(self) -> bool:
        return self.perturated_measurements["inflow"]

    @property
    def perturbate_pump_flow(self) -> bool:
        return self.perturated_measurements["pump_flow"]

    def default_rng(self) -> np.random.default_rng:
        return np.random.default_rng(self.rng_seed)

    def add_perturbation_excluded_nodes(self, excluded_nodes: List[str]):
        self.excluded_elements.nodes.extend(excluded_nodes)

    def add_perturbation_excluded_pipes(self, excluded_pipes: List[str]):
        self.excluded_elements.links.extend(excluded_pipes)

    def set_roughness_multiplier(self, new_mult):
        self.roughness_multiplier = new_mult

    def set_demand_multiplier(self, new_mult):
        self.demand_multiplier = new_mult

    def set_accuracy_multiplier(self, new_mult):
        self.sensor_accuracy_multiplier = new_mult

    def set_cusum_multipliers(self, new_mults: List[float]):
        self.cusum_multipliers = new_mults

    def get_sensitivity_parameters(self) -> dict[str, Any]:
        # 1. noise/multipliers

        out = {"perturbation_modes": self.perturbation_mode, "perturbation_values": {}}

        for k, v in self.perturbation_mode.items():
            match v:
                case "multiplier":
                    k_out = f"{k}_multiplier"
                    v_out = self.mult_map[k]
                case "uniform":
                    k_out = f"{k}_noise"
                    v_out = self.noise_map[k]
            out["perturbation_values"][k_out] = v_out

        # 2. head rounding
        out["head_rounding_decimals"] = self.head_rounding_decimals

        return out

    def to_json(self, fp: os.PathLike):
        from dataclasses import asdict

        def default_handler(obj):
            if isinstance(obj, pd.Timedelta):
                return {"__type__": "Timedelta", "value": obj.isoformat()}
            elif isinstance(obj, pd.Timestamp):
                return {"__type__": "Timestamp", "value": obj.isoformat()}
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        data = asdict(self)
        print(data)
        with open(fp, "w") as f:
            json.dump(data, f, default=default_handler, indent=2)

    @classmethod
    def from_json(cls, fp: os.PathLike):
        from dataclasses import fields

        def object_hook(dct):
            if "__type__" in dct:
                if dct["__type__"] == "Timedelta":
                    return pd.Timedelta(dct["value"])
                elif dct["__type__"] == "Timestamp":
                    return pd.Timestamp(dct["value"])
            return dct

        with open(fp, "r") as f:
            data = json.load(f, object_hook=object_hook)

        # Handle nested dataclass
        if "excluded_elements" in data:
            data["excluded_elements"] = _ExcludedElements(**data["excluded_elements"])

        # Separate init and non-init fields to handle init=False fields
        init_fields = {f.name for f in fields(cls) if f.init}
        init_data = {k: v for k, v in data.items() if k in init_fields}
        non_init_data = {k: v for k, v in data.items() if k not in init_fields}

        # Create instance with init fields
        instance = cls(**init_data)

        # Set non-init fields manually
        for k, v in non_init_data.items():
            setattr(instance, k, v)

        return instance


@dataclass
class SensitivitySetup:
    """Container for permutations of realisations to test in the sensitivity analysis
    if discrete_df is provided, realisations are ignored and the iterator is built from the dataframe
    placement_scenarios are still used to determine the parameters

    Can be iterated over to get deterministic
    for tpl in self:
        count, sensors, accuracy, roughness, demand = tpl
        <use the sensitivity parameters>
    """

    placement_scenarios: dict[int | str, dict[str, list[str]]]
    acc_realisations: float | list[float] = field(default=0, init=True)
    rgh_realisations: float | list[float] = field(default=0, init=True)
    dem_realisations: float | list[float] = field(default=0, init=True)
    discrete_df: pd.DataFrame | None = None
    cusum_realisations: float | list[float] | None = None
    resume_at: int = 1  # starts at first scenario by default
    stop_at: Optional[int] = None

    def __post_init__(self):
        self._counter = None
        self._iterator = None

        if self.discrete_df is not None:
            assert all(
                n in self.discrete_df.columns
                for n in ["demand", "roughness", "measurement"]
            ), ""

        for attr in [
            "acc_realisations",
            "rgh_realisations",
            "dem_realisations",
        ]:
            if isinstance(v := getattr(self, attr), (float, int)):
                setattr(self, attr, [v])

    def __repr__(self):
        return pformat(self.__dict__)

    def __len__(self):

        def get_len(obj):
            if isinstance(obj, (list, dict)):
                return len(obj)
            elif isinstance(obj, float):
                return 1
            elif obj is None:
                return 1
            else:
                raise ValueError(f"Invalid type {type(obj)}.")

        if self.discrete_df is not None:
            return len(self.discrete_df) * get_len(self.placement_scenarios)

        n_placement = get_len(self.placement_scenarios)
        n_acc = get_len(self.acc_realisations)
        n_rgh = get_len(self.acc_realisations)
        n_dem = get_len(self.dem_realisations)
        n_cusum = get_len(self.cusum_realisations)

        return n_placement * n_acc * n_rgh * n_dem * n_cusum

    @property
    def all_sensors(self) -> List:
        all_sensors: Set = set()
        for v in self.placement_scenarios.values():
            for slist in v.values():
                all_sensors.update(slist)
        return list(all_sensors)

    @property
    def n_scenarios(self) -> int:

        if self.discrete_df is not None:
            return len(self.discrete_df) * len(self.placement_scenarios.keys())

        l = (
            len(self.placement_scenarios.keys())
            * len(list(self.acc_realisations))
            * len(list(self.rgh_realisations))
            * len(list(self.dem_realisations))
        )

        return l

        """Not implemented logic
        out = -1

        if (self.resume_at == 1) and (self.stop_at is None):
            out = l

        elif self.resume_at == 1 and (self.stop_at is not None):
            out = self.stop_at

        elif self.resume_at != 1 and (self.stop_at is None):
            out = l - self.resume_at - 1

        else:
            out = self.stop_at - self.resume_at

        assert 0 < out < l, ""

        return out
        """

    @property
    def realisations(self) -> Dict[str, List[float]] | pd.DataFrame:
        if self.discrete_df is not None:
            return self.discrete_df

        return {
            "sensor_accuracy": self.acc_realisations,
            "demand": self.dem_realisations,
            "roughness": self.rgh_realisations,
        }

    def __iter__(self) -> Self:
        # small exercise in itertools :)
        if self.discrete_df is not None:
            acc = self.discrete_df["measurement"]
            dem = self.discrete_df["demand"]
            rgh = self.discrete_df["roughness"]

            iter_df = pd.DataFrame([], columns=["placement", "acc", "rgh", "dem"])
            for k in self.placement_scenarios.keys():

                _sub_df = pd.DataFrame(
                    {"placement": k, "acc": acc, "rgh": rgh, "dem": dem}
                )

                iter_df = pd.concat([iter_df, _sub_df])

            base_iterator = iter_df.iterrows()

        else:
            base_iterator = product(
                self.placement_scenarios.keys(),
                self.acc_realisations,
                self.rgh_realisations,
                self.dem_realisations,
            )

        self._iterator = islice(
            base_iterator, max(0, self.resume_at - 1), self.stop_at, 1
        )
        self._counter = count(max(1, self.resume_at))

        return self

    def __next__(self) -> Tuple[int, Dict[str, Iterable], float, float, float]:
        try:
            if self.discrete_df is None:
                p, a, r, d = next(self._iterator)
            else:
                _, (p, a, r, d) = next(self._iterator)

            i = next(self._counter)
            return i, self.placement_scenarios[p], a, r, d

        except (StopIteration, TypeError):
            raise StopIteration("All scenarios have been processed")

    def to_json(self, fp: os.PathLike):
        from dataclasses import asdict

        data = asdict(self)

        if self.discrete_df is not None:
            data["discrete_df"] = self.discrete_df.to_dict()
            print(data)

        with open(fp, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, fp: os.PathLike):
        with open(fp, "r") as f:
            data = json.load(f)

        # JSON converts all dict keys to strings, so convert back to int where possible
        if "placement_scenarios" in data:
            converted_scenarios = {}
            for k, v in data["placement_scenarios"].items():
                # Try to convert key to int, otherwise keep as string
                try:
                    key = int(k)
                except (ValueError, TypeError):
                    key = k
                converted_scenarios[key] = v
            data["placement_scenarios"] = converted_scenarios

        if "discrete_df" in data:
            data["discrete_df"] = pd.DataFrame.from_dict(data["discrete_df"])

        return cls(**data)

    def get_chunks(self, chunksize: int) -> list[list[tuple]]:
        chunks = []

        collector = []
        for tpl in self:
            collector.append(tpl)
            if len(collector) == chunksize:
                chunks.append(collector)
                collector = []

        if collector:
            chunks.append(collector)

        return chunks
