from __future__ import annotations
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import traceback
from typing import (
    List,
    Dict,
    Optional,
    Literal,
    Tuple,
    ClassVar,
    Any,
    Self,
    Type,
)

from pprint import pprint
import concurrent.futures as ccf
from functools import reduce, cached_property, partialmethod
import json
import gc

from time import perf_counter
from math import floor

import pandas as pd
import numpy as np
from .util import (
    FileCount,
    SimEnv,
    reindex_cyclic,
)
from .model import (
    HydraulicNetwork,
    VirtualReservoir,
    Options,
    PERTURBATION_TARGET,
    SensitivitySetup,
    apply_uniform_noise2,
)

from .model.network_simulation import (
    SimulationTargets,
    DynBaseDemandPertArgs,
    PatternPertArgs,
)

CPU_COUNT: int | None = os.cpu_count()

simulation_mode = Literal["full", "sensitivity_preparation"]

"""
    Dynamic base demand updating atm only available for flow correction mode!!
"""


# Worker classes (handle single dual model simulations after data preparation)
@dataclass
class _SimulationWorker(ABC):
    """Submitted to ccf executor like executor.submit(worker)

    Class:
        cusum_ground_truth: Dict that contains location (key) and start/end times of leakage in s (value)

    Instance:

        simulation:         _Simulation instance where worker was initialised
        wdir:               working directory of worker
        nw:                 reference to HydraulicNetwork, copy when needed!
        heads:              head measurements as dataframe
        node_demands:       dataframe with primal demand multipliers and categories
        link_roughness:     dataframe with primal pipe roughness
        pressure_sensors:   list of node ids hwre sensors are
        options:            simulation Options instance

    """

    # ground truth on (sub-)class level, set in _Simulation
    cusum_ground_truth: ClassVar[Optional[Dict[str, Tuple[float, float]]]] = None

    wdir: os.PathLike
    id: int
    pressure_sensors: List[str]
    options: Options

    heads_path: os.PathLike
    inflows_path: os.PathLike
    node_demands_path: os.PathLike
    link_roughness_path: os.PathLike

    # read when worker is run
    nw: HydraulicNetwork = field(default=None, init=False)
    heads: pd.DataFrame = field(default=None, init=False)
    inflows: pd.DataFrame = field(default=None, init=False)
    node_demands: pd.DataFrame = field(default=None, init=False)
    link_roughness: pd.DataFrame = field(default=None, init=False)

    # results
    vflows: pd.DataFrame = field(default=None, init=False)
    ttc: float = field(default=None, init=False)

    # optionally calculated after simulation if cusum is enabled
    ttd: Dict[str, pd.Timedelta] = field(default_factory=dict, init=False)

    @classmethod
    def set_cusum_ground_truth(cls, cgt: Dict[str, Tuple[float, float]]):
        cls.cusum_ground_truth = cgt
        print(f"CUSUM ground truth for {cls.__name__} set to")
        pprint(cgt)
        print("\n")

    @property
    def pm_path(self):
        return os.path.join(self.wdir, f"prepared_primal_model_copy_{self.id}.inp")

    @property
    def dm_path(self):
        """Dual Model"""
        return os.path.join(self.wdir, f"static_dual_model_{self.id}.inp")

    @property
    def dmc_path(self):
        """Corrected Dual Model"""
        return os.path.join(self.wdir, "static_dual_model_pert_corr.inp")

    @property
    def vf_path(self):
        return os.path.join(self.wdir, f"virtual_flows_{self.id}.parquet")

    @property
    def sens_param_path(self):
        return os.path.join(self.wdir, f"sensitivity_parameters_{self.id}.json")

    def __post_init__(self):
        """After initialisation, working dir is created"""
        self._create_subdir()

    def __call__(self) -> Self:
        self._save_sens_params_json()
        self._read_data()
        self._run_simulation()
        self._cleanup()
        return self

    def _read_data(self):
        self.nw = HydraulicNetwork(self.pm_path)
        self.heads = pd.read_parquet(self.heads_path, columns=self.pressure_sensors)
        self.inflows = pd.read_parquet(self.inflows_path)
        self.node_demands = pd.read_parquet(self.node_demands_path)
        self.link_roughness = pd.read_parquet(self.link_roughness_path)

    def _create_subdir(self):
        if not os.path.exists(self.wdir):
            os.mkdir(self.wdir)

    def _apply_perturbation(
        self, df: pd.DataFrame, df_type: PERTURBATION_TARGET
    ) -> pd.DataFrame:

        pmode = self.options.perturbation_mode[df_type]
        match pmode:
            case "multiplier":
                mult = self.options.mult_map[df_type]
                result = df * mult
            case "uniform":
                pctg = self.options.noise_map[df_type]
                result = apply_uniform_noise2(
                    df=df, noise_pctg=pctg, rng=self.options.default_rng()
                )
            case _:
                raise Exception(
                    f"Invalid noise mode '{self.options.perturbation_mode}."
                )

        return result

    _perturbate_measurement = partialmethod(_apply_perturbation, df_type="measurement")
    _perturbate_roughness = partialmethod(_apply_perturbation, df_type="roughness")
    _perturbate_demand = partialmethod(_apply_perturbation, df_type="demand")

    def get_perturbated_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        noisy_demands = self._perturbate_demand(df=self.node_demands)
        noisy_roughness = self._perturbate_roughness(df=self.link_roughness)
        noisy_heads = self._perturbate_measurement(df=self.heads)

        return noisy_demands, noisy_roughness, noisy_heads

    def get_unperturbated_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.node_demands, self.link_roughness, self.heads

    @abstractmethod
    def _run_simulation(self): ...

    def _process_virtual_flows(self):
        if self.options.aggregate_virtual_flows:
            self.vflows = self.vflows.sum(axis=1).to_frame()

        if self.options.save_virtual_flows:
            self.vflows.to_parquet(self.vf_path)

    def _save_sens_params_json(self):
        if self.options.save_sensitivity_parameter_json:
            with open(self.sens_param_path, "w") as file:
                json.dump(self.options.get_sensitivity_parameters(), fp=file)

    def _cleanup(self):
        if self.options.delete_worker_models:
            for file in os.listdir(self.wdir):
                name, ext = os.path.splitext(file)
                if ext == ".inp":
                    _path = os.path.join(self.wdir, file)
                    os.remove(_path)


@dataclass
class SimulationWorkerFC(_SimulationWorker):

    corr_flows: pd.DataFrame = field(default=None, init=False)

    def _run_simulation(self):
        start_time = perf_counter()

        # 1a. apply perturbation
        noisy_demands, noisy_roughness, noisy_heads = self.get_perturbated_data()

        ## only update demand now if base demand is not dynamic
        if self.options.demand_perturbation_method == "base":
            self.nw.import_base_demands(
                noisy_demands, exclude=self.options.excluded_elements.nodes
            )
        self.nw.import_roughness(
            noisy_roughness, exclude=self.options.excluded_elements.links
        )

        # 1b. convert network to static dual model
        _, vps = self.nw.add_virtual_reservoirs(*self.pressure_sensors)
        self.nw.set_patterns(patterns=noisy_heads)
        self.nw.save(self.dm_path)

        if self.options.use_dual_model_correction:
            # 2. run short simulation to get correction period (default: 1 week)
            cal_duration = min(
                floor(self.options.correction_period.total_seconds()),
                int(self.options.max_simulation_time.total_seconds()),
            )
            solver_options = {"setTimeSimulationDuration": [cal_duration]}
            self.corr_flows = self.nw.run_simulation(
                simulation_targets={"flow": vps}, solver_options=solver_options
            )["flow"]

            self.corr_flows.columns = [
                f"{c.split('_')[-1]}{VirtualReservoir._flow_corr_pattern_suffix}"
                for c in self.corr_flows.columns
            ]

            # 3. Add VN patterns as demand patterns to model
            self.nw.set_patterns(self.corr_flows)
            self.nw.save(self.dmc_path)

        # 4. Run full simulation and save vflows
        ## 4.1 demand perturbation
        dem_perturbation_settings = None

        if self.options.demand_perturbation_method == "dyn_base":
            if self.options.demand_perturbation_frequency > 0:
                dem_perturbation_settings = DynBaseDemandPertArgs(
                    base_seed=self.options.rng_seed,
                    percentage=self.options.demand_noise,
                    excluded_junctions=self.options.excluded_perturbation_nodes,
                    frequency=self.options.demand_perturbation_frequency,
                )
        if self.options.demand_perturbation_method == "rng_patterns":
            dem_perturbation_settings = PatternPertArgs(
                n=self.options.rng_pattern_count,
                noise_pctg=self.options.demand_noise,
                pattern_names=self.options.rng_pattern_names,
                seed=self.options.rng_seed,
                skip_constant_patterns=self.options.rng_skip_constant_patterns,
                skip_nodes=self.options.excluded_perturbation_nodes,
            )

        self.vflows = self.nw.run_simulation(
            simulation_targets={"flow": vps},
            demand_perturbation_settings=dem_perturbation_settings,
        )["flow"]

        self._process_virtual_flows()

        end_time = perf_counter()

        self.ttc = end_time - start_time


@dataclass
class SimulationWorkerHC(_SimulationWorker):

    corr_heads: pd.DataFrame = field(default=None, init=False)

    def __post_init__(self):
        raise Exception(
            f"Head correction correction is not up to date and probably discontinued."
        )
        return super().__post_init__()

    @property
    def pmp_path(self):
        """Perturbated primal model"""
        return os.path.join(self.wdir, "primal_model_pert.inp")

    @property
    def dmc_path(self):
        """Corrected Dual Model"""
        return os.path.join(self.wdir, "dual_model_corr.inp")

    def _run_simulation(self):
        start_time = perf_counter()

        # 1. convert to perturbated primal model
        noisy_demands, noisy_roughness, noisy_heads = self.get_perturbated_data()

        self.nw.import_base_demands(
            noisy_demands, exclude=self.options.excluded_elements.nodes
        )
        self.nw.import_roughness(
            noisy_roughness, exclude=self.options.excluded_elements.links
        )
        self.nw.save(self.pmp_path)

        # 2. simulate to get perturbated heads (1 week calibration)
        # which week is determined in options as integer (1 = first week)
        # TODO: This currently does not work after making the primal model static, change order or procedure
        if self.options.use_dual_model_correction:
            solver_options = {
                "setTimeSimulationDuration": [
                    floor(
                        self.options.calibration_week
                        * self.options.correction_period.total_seconds()
                    )
                ]
            }

            corr_start = (self.options.calibration_week - 1) * pd.Timedelta(days=7)
            corr_end = self.options.calibration_week * pd.Timedelta(
                days=7
            ) - pd.Timedelta("1ns")

            self.corr_heads = self.nw.run_simulation(
                simulation_targets={"head": self.pressure_sensors},
                solver_options=solver_options,
            )["head"].loc[corr_start:corr_end]

            if self.options.head_rounding_decimals is not None:
                self.corr_heads = self.corr_heads.round(
                    self.options.head_rounding_decimals
                )

            # 3. calculate head correction pattern
            hcp = reindex_cyclic(
                df=self.corr_heads - noisy_heads.loc[self.corr_heads.index],
                new_idx=noisy_heads.index,
            )

            _heads = self.heads.add(hcp)

        else:
            _heads = self.heads

        # 4. Convert to dual model, insert corrected heads as head patterns
        _, vps = self.nw.add_virtual_reservoirs(*self.pressure_sensors)
        self.nw.set_patterns(_heads)
        self.nw.save(self.dmc_path)

        # 5. Calculate virtual flows
        self.vflows = self.nw.run_simulation(simulation_targets={"flow": vps})

        self._process_virtual_flows()

        end_time = perf_counter()

        self.ttc = end_time - start_time


# Simulation wrappers (dispatch workers for simulation iterations)
@dataclass
class _Simulation(ABC):

    # subclass classvars
    WORKER_TYPE: ClassVar[Type[_SimulationWorker]] = None

    # global classvars
    ISOLATED_CORES: ClassVar[int] = 2  # n_workers = cpu_count - isolated_cores
    FC_INP_PREFIX: ClassVar[str] = "nw"
    FC_DAT_PREFIX: ClassVar[str] = "data"

    source_inp: os.PathLike
    artificial_leaks_path: os.PathLike
    options: Options
    sensitivity_setup: SensitivitySetup

    # in postinit
    _filecounter_inp: FileCount = field(default=None, init=False)
    _filecounter_data: FileCount = field(default=None, init=False)
    _leak_df: pd.DataFrame = field(default=None, init=False)

    # prepared in simulation
    _dir: os.PathLike = field(default=None, init=False)
    _heads: pd.DataFrame = field(default=None, init=False)
    _pressures: pd.DataFrame = field(default=None, init=False)
    _inflows: pd.DataFrame = field(default=None, init=False)
    _pump_flows: pd.DataFrame = field(default=None, init=False)
    _original_rgh: pd.DataFrame = field(default=None, init=False)
    _original_dem: pd.DataFrame = field(default=None, init=False)
    _prepared_primal_network: HydraulicNetwork = field(default=None, init=False)
    _cgt: Dict[Tuple[float, float]] = field(
        default_factory=dict, init=False
    )  # CUSUM ground truth

    primal_results: Dict[SimulationTargets, pd.DataFrame] = field(
        default_factory=dict, init=False
    )

    # pathes
    _demands_path: os.PathLike = field(default=None, init=False)
    _roughness_path: os.PathLike = field(default=None, init=False)
    _path_primal: os.PathLike = field(default=None, init=False)
    _path_primal_leaks: os.PathLike = field(default=None, init=False)
    _path_primal_prepared: os.PathLike = field(default=None, init=False)

    pathes_primal_results: dict[os.PathLike] = field(default_factory=dict, init=False)

    # result batch folders
    _batch_folders: Dict[int, os.PathLike] = field(default_factory=dict, init=False)

    # simulation results
    results: Dict[int, Any] = field(default_factory=dict, init=False)  # in _run()
    failed: Dict[int, Any] = field(default_factory=dict, init=False)

    # workers
    workers: List[_SimulationWorker] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._read_leaks()
        self._filecounter_inp = FileCount(prefix=self.FC_INP_PREFIX)
        self._filecounter_data = FileCount(prefix=self.FC_DAT_PREFIX)

        if self.WORKER_TYPE is None:
            raise Exception("ClassVar WORKER_TYPE not set in subclass.")

    @cached_property
    def ground_truth(self) -> Optional[Dict[Tuple[float, float]]]:
        self._calculate_cusum_ground_truth(
            self._get_leaks(self.options.max_simulation_time)
        )
        if len(self._cgt) > 0:
            return self._cgt
        return None

    @classmethod
    def N_WORKERS(cls):
        return os.cpu_count() - cls.ISOLATED_CORES

    def _read_leaks(self) -> None:
        """
        index: pd.Timedelta
        columns: pipe names
        dtype: float
        """
        df = pd.read_csv(self.artificial_leaks_path, index_col=0)
        df.index = pd.to_timedelta(df.index)

        self._leak_df = df

    def _get_leaks(self, ts_end: Optional[pd.Timedelta] = None) -> List[pd.Series]:
        if ts_end is not None:
            df = self._leak_df.loc[:ts_end]
        else:
            df = self._leak_df

        return [l for n, l in df.items() if l.sum() > 0]

    def _get_worker(self, kw_dict: Dict[str, Any]) -> _SimulationWorker:
        worker = self.WORKER_TYPE(**kw_dict)
        # after worker creation write network copy to worker.nw_path
        self._prepared_primal_network.save(worker.pm_path)
        return worker

    def _copy_source_network(self) -> None:
        assert self._dir is not None, ""

        self._path_primal = self._filecounter_inp.get_path(self._dir, "primal.inp")

        if self.options.max_simulation_time is None:
            shutil.copyfile(self.source_inp, self._path_primal)

        else:
            _onw = HydraulicNetwork(self.source_inp).export_simulation_slice(
                end=self.options.max_simulation_time
            )
            _onw.save(self._path_primal)

    def _create_batch_folders(self, n: int):
        assert self._dir is not None, ""

        n_scenarios = self.sensitivity_setup.n_scenarios
        n_folders = n_scenarios // n + 1

        for i in range(n_folders):
            n0 = i * n + 1
            n1 = min((i + 1) * n, n_scenarios)
            _name = f"{n0} to {n1}"
            _path = os.path.join(self._dir, _name)
            if os.path.exists(_path):
                continue
            os.mkdir(_path)
            self._batch_folders[n1] = _path

    def _get_batch_folder(self, simulation_number: int):
        assert self._dir is not None, ""

        batch_folder = None
        for k, v in self._batch_folders.items():
            if simulation_number <= k:
                batch_folder = v
                break

        return batch_folder

    def _calculate_cusum_ground_truth(self, leaks: List[pd.Series]) -> None:
        """Called from ground_truth property"""
        assert self._leak_df is not None, ""

        for l in leaks:
            _l = l.replace(0, np.nan).dropna()
            start_time_seconds = _l.first_valid_index().total_seconds()
            end_time_seconds = _l.last_valid_index().total_seconds()
            self._cgt[l.name] = (
                start_time_seconds,
                end_time_seconds,
            )

    def run(
        self,
        delete_after: bool = False,
        stdout_redirect: bool = False,
        preparation_only: bool = False,
        batch_folder_size: int = 500,
    ):
        """Run simulation preparation and iterations from setup
        Steps:
            1. Run preparation steps, write data attrs, retrieve primal HydraulicNetwork
                if preparation_only: return
            2. Create batch folders for iterations
            3. Run main simulation (defined in subclasses)
            4. clean simulation directory
        """

        with SimEnv(
            self.source_inp,
            delete_after=delete_after,
            stdout_redirect=stdout_redirect,
            cleanup=True,
        ) as _dir:
            self._dir = _dir

            # 0. hardcopy source inp in simulation directory
            self._copy_source_network()

            # 1st part of simulation
            self._prepared_primal_network = self._run_preparation()

            if preparation_only:
                return

            # 2nd part of simulation
            # create batch folders
            self._create_batch_folders(n=batch_folder_size)

            # 3nd part of simulation, dispatch workers in chunks
            # set worker CUSUM of necessary
            if self.options.run_cusum:
                self.WORKER_TYPE.set_cusum_ground_truth(self.ground_truth)

            # run simulations
            CHUNKSIZE = self.N_WORKERS() * 2
            chunks = self.sensitivity_setup.get_chunks(CHUNKSIZE)

            for chunk in chunks:
                # chunk: list[tuple[...]]
                with ccf.ProcessPoolExecutor(max_workers=self.N_WORKERS()) as executor:
                    future_to_id = {}
                    for task in chunk:
                        d = self._dispatch_worker(executor, task)
                        future_to_id.update(d)

                    self._collect_workers(future_to_id)

    def _run_preparation(self):
        """First part of simulation workflow
        1. measurement creation from primal model
        2. write data and networks to folder that can then be used by the parallel simulation
        """
        # 1. Read
        onw = HydraulicNetwork(self._path_primal)

        # 2. Export component data
        self._original_dem = onw.export_base_demands()
        self._original_rgh = onw.export_roughness()

        self._demands_path = self._filecounter_data.get_path(
            self._dir, "demands.parquet"
        )
        self._original_dem.to_parquet(self._demands_path)
        self._roughness_path = self._filecounter_data.get_path(
            self._dir, "roughness.parquet"
        )
        self._original_rgh.to_parquet(self._roughness_path)

        # 3. Insert leaks
        leaks = self._get_leaks(ts_end=self.options.max_simulation_time)
        onw.add_leaks(*leaks)

        self._path_primal_leaks = self._filecounter_inp.get_path(
            self._dir, "primal_leaks.inp"
        )
        onw.save(self._path_primal_leaks)

        # 4a. Run simulation to retrieve artificial measurements
        psensors = self.sensitivity_setup.all_sensors  # pressure sensor locations
        fpipes = self.options.flow_pipe_ids
        fpumps = self.options.pump_ids

        sim_targets = {
            "head": psensors,
            "pressure": psensors,
            "flow": fpipes,
            "pump_flow": fpumps,
        }

        self.primal_results = onw.run_simulation(simulation_targets=sim_targets)
        if (hrd := self.options.head_rounding_decimals) is not None:
            self.primal_results["head"] = self.primal_results["head"].round(hrd)
            self.primal_results["pressure"] = self.primal_results["pressure"].round(hrd)

        # save primal data
        for tgt, df in self.primal_results.items():
            _path = self._filecounter_data.get_path(self._dir, f"primal_{tgt}.parquet")
            df.to_parquet(_path)
            self.pathes_primal_results[tgt] = _path

        # 4b. if inflows and/or pump_flows get a sensor accuracy error
        ## TODO: Implement but find different logic first since this should happen in Workers and not Preparation

        # 5. Remove leakages (leakages are imprinted in the head measurements)
        onw.remove_artificial_leaks()

        # 6. Split network if pumps are specified
        if (fpumps is not None) and (len(fpumps) > 0):
            for pump_id in fpumps:
                onw.split_area_at_pump(
                    pump_id=pump_id,
                    substitute_demand_pattern=self.primal_results["pump_flow"][pump_id],
                )

        # 7. Insert surrogates for tanks/reservoirs
        # this will lead to a defective network without reservoirs
        # will only work again after converting to dual model!
        for obj_id, pipe_id in self.options.inflow_replacement_mapping.items():
            dmp_series = self.primal_results["flow"][pipe_id]

            onw.replace_reservoir_or_tank_with_emitter_node(
                reservoir_or_tank_id=obj_id,
                demand_pattern_series=dmp_series,
                pipes_to_reconnect=pipe_id,
            )

        self._path_primal_prepared = self._filecounter_inp.get_path(
            self._dir, "primal_prepared.inp"
        )

        onw.save(self._path_primal_prepared)

        return onw

    def _dispatch_worker(self, executor: ccf.ProcessPoolExecutor, sens_params: tuple):
        count, sensor_ids, acc_realisation, rgh_realisation, dem_realisation = (
            sens_params
        )

        # folder
        wd = self._get_batch_folder(simulation_number=count)
        wdir = os.path.join(wd, f"{count}")

        # altered options
        _options = self.options.copy()
        # change options for sim worker using the sensitivity settings
        for t, m in _options.perturbation_mode.items():
            new_val = {
                "demand": dem_realisation,
                "roughness": rgh_realisation,
                "measurement": acc_realisation,
            }.get(t)

            _options.set_perturbation_value(new_val, target=t, mode=m)

        # sensor list
        sensors = reduce(lambda x, y: x + y, sensor_ids.values(), [])

        # attributes
        kwargs = dict(
            wdir=wdir,
            id=count,
            heads_path=os.path.abspath(self.pathes_primal_results["head"]),
            inflows_path=os.path.abspath(self.pathes_primal_results["flow"]),
            node_demands_path=os.path.abspath(self._demands_path),
            link_roughness_path=os.path.abspath(self._roughness_path),
            pressure_sensors=sensors,
            options=_options,
        )

        # submit
        worker = self._get_worker(kwargs)

        future = executor.submit(worker)

        return {future: (count, wd)}

    def _collect_workers(
        self,
        future_to_id: Dict[ccf.Future, Tuple[int, os.PathLike]],
    ):
        for future in ccf.as_completed(future_to_id):
            count, wd = future_to_id[future]

            if future.exception() is not None:
                # Worker failed
                exc = future.exception()
                tb = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )

                self.failed[count] = {
                    "wd": wd,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": tb,
                }

                print(
                    f"Simulation {count} failed:\n{type(exc).__name__}: {str(exc)}\n{tb}"
                )
            else:
                # Worker succeeded
                worker: _SimulationWorker = future.result()

                if self.options.keep_results:
                    self.results[count] = {
                        "wd": wd,
                        "vf": worker.vflows,
                        "ttd": worker.ttd,
                        "ttc": worker.ttc,
                        "options": worker.options,
                    }

                # worker fate
                if self.options.keep_workers:
                    self.workers.append(worker)
                else:
                    # delete worker and gc
                    del worker
                    gc.collect()


# New simulation (flow correction, fixed boundary conditions)
@dataclass
class SimulationFC(_Simulation):
    """Flow corrected simulation
    Will automatically dispatch SimulationWorkerFC
    """

    WORKER_TYPE: ClassVar[Type[_SimulationWorker]] = SimulationWorkerFC


# New version of old simulation
@dataclass
class SimulationHC(_Simulation):
    """Head corrected simulation
    Will automatically dispatch SimulationWorkerHC
    """

    WORKER_TYPE: ClassVar[Type[_SimulationWorker]] = SimulationWorkerHC


def get_simulator(
    source_inp: os.PathLike,
    artificial_leaks_path: os.PathLike,
    options: Options,
    sensitivity_setup: SensitivitySetup,
    correction: Literal["head", "flow"] = "flow",
) -> _Simulation:
    match correction:
        case "flow":
            obj = SimulationFC
        case "head":
            obj = SimulationHC

    return obj(
        source_inp=source_inp,
        artificial_leaks_path=artificial_leaks_path,
        options=options,
        sensitivity_setup=sensitivity_setup,
    )
