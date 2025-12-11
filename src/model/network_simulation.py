from __future__ import annotations
import os
from contextlib import contextmanager
import multiprocessing as mp
import tempfile
from itertools import count
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any, Never, Tuple, ClassVar

import pandas as pd
import numpy as np

from epyt import epanet

from src.util import timer


SimulationKind = Literal["head", "pressure", "flow", "pump_flow"]
SimulationTargets = Dict[SimulationKind, List[str]]


VIRTUAL_FLOW_PATTERN_NAME = "virtual_flow"

__DEBUG = True

def _to_offset(s: str) -> Tuple[bool, pd.DateOffset]:
    try:
        offset = pd.tseries.frequencies.to_offset(s)
        return (True, offset)
    except:
        return (False, None)


def assign_perturbated_patterns(
    d: epanet,
    n: int,
    noise_pctg: float,
    pattern_names: str | None = None,
    seed: int = 42,
    skip_nodes: list[str] | None = None,
    skip_constant_patterns: bool = True,
):
    """
    d: epanet instance
    n: number of generated patterns
    noise_pctg: applied noise
    pattern_names: names of patterns to be perturbated
    seed: for reproducibility
    skip_node: excluded nodes (e.g. industrial)
    skip_constant_patterns: if no variability in pattern, do not touch

    performs operation on live epyt.epanet instance
    """
    global __DEBUG

    if not pattern_names:
        print(
            f"Warning! No pattern names provided. This will potentially break things if non-demand patterns exist in the network."
        )

    original_patterns = pd.DataFrame(d.getPattern().T, columns=d.getPatternNameID())
    original_patterns = original_patterns[
        [c for c in original_patterns.columns if c in (pattern_names or [])]
    ]

    # initialise pattern name containers
    valid_pattern_names = []
    for name, values in original_patterns.items():
        if skip_constant_patterns:
            if values.min() == values.max():
                continue
            valid_pattern_names.append(name)

    # seed generator
    seed_gen = iter(
        np.random.default_rng(seed).integers(
            low=1, high=2**31, size=n * len(valid_pattern_names)
        )
    )

    # generate perturbated patterns
    new_pattern_ids = defaultdict(list)
    for p in valid_pattern_names:
        for i in range(1, n + 1):
            rng = np.random.default_rng(next(seed_gen))  # unique rng
            perturbated_pattern = original_patterns[p] * rng.uniform(
                low=1 - noise_pctg, high=1 + noise_pctg, size=len(original_patterns)
            )
            name = f"{p}_{i}"
            idx = d.addPattern(name, perturbated_pattern)
            new_pattern_ids[p].append(idx)

    # iterate over nodes and randomly assign perturbated patterns
    excluded_node_indices = [
        _nidx for node in (skip_nodes or []) if (_nidx := d.getNodeIndex(node)) != 0
    ]

    assignment_rng = np.random.default_rng(seed)  # fresh RNG for new pattern assignment

    for category_index, pattern_names in d.getNodeDemandPatternNameID().items():
        if d.getPatternNameID()[category_index - 1] not in new_pattern_ids.keys():
            continue
        for node_index in d.getNodeIndex():
            if node_index in excluded_node_indices:
                continue
            pat_name = pattern_names[node_index - 1]
            if not pat_name:  # '' e.g. for reservoirs
                continue
            node_pattern = pattern_names[node_index - 1]
            pids = new_pattern_ids.get(node_pattern, None)
            if pids is None:
                # node has no pattern in this category that is relevant
                continue
            random_pattern_index = assignment_rng.choice(pids, size=1)[0]
            d.setNodeDemandPatternIndex(
                node_index, category_index, random_pattern_index
            )

    if __DEBUG:
        d.saveInputFile(f"{str(uuid4())}.inp")


@dataclass(frozen=True)
class PatternPertArgs:
    n: int
    noise_pctg: float
    pattern_names: str | None
    seed: int
    skip_nodes: list[str] | None
    skip_constant_patterns: bool


@dataclass
class DynamicBaseDemandPerturbation:

    initial_network: epanet
    perturbation_percentage: float
    exclude_junctions: list[str] = field(default_factory=list)
    seed: int = field(default_factory=int)

    update_interval: int | None = field(default=None, init=True)

    _data: pd.DataFrame = field(default=None, init=False)
    _valid_junction_indices: list[int] = field(default_factory=list, init=False)
    _counter: count = field(default=None, init=False)

    def __post_init__(self):
        self._get_valid_junction_indices()
        self._get_dataframe()

        # if no interval given
        if self.update_interval is None:
            self.update_interval = self.initial_network.getTimePatternStep()

        # detach original network, not needed anymore
        self.initial_network = None
        self._counter = count()

    def _get_valid_junction_indices(self):
        excluded_indices = [
            self.initial_network.getNodeIndex(name) for name in self.exclude_junctions
        ]
        self._valid_junction_indices = np.array(
            [
                n
                for n in self.initial_network.getNodeIndex()
                if n not in excluded_indices
            ]
        )

    def _get_dataframe(self):
        pnames = pd.DataFrame(self.initial_network.getNodeDemandPatternNameID()).stack()
        pindices = pd.DataFrame(
            self.initial_network.getNodeDemandPatternIndex()
        ).stack()
        pvalues = pd.DataFrame(self.initial_network.getNodeBaseDemands()).stack()

        self._data = (
            pd.concat([pnames, pindices, pvalues], axis=1).droplevel(1).reset_index()
        )
        self._data.columns = [
            "node_index",
            "pattern_name",
            "pattern_index",
            "original_multiplier",
        ]
        self._data["node_index"] += 1
        self._data = self._data.set_index(["pattern_index", "node_index"])

    def _get_rng(self):
        seed = self.seed + next(self._counter)
        return np.random.default_rng(seed)

    def pertube_base_demands(
        self,
        d: epanet,
        current_simulation_time: int,
    ) -> epanet:
        # guard clause
        if current_simulation_time % self.update_interval != 0:
            return d

        rng = self._get_rng()

        self._data["pert_multiplier"] = (
            rng.uniform(
                low=1 - self.perturbation_percentage,
                high=1 + self.perturbation_percentage,
                size=len(self._data),
            )
            * self._data["original_multiplier"]
        )

        for cat in self._data.index.get_level_values(0).unique():
            # exclude category == 0 (means no demand category assisnged)
            if cat == 0:
                continue

            _slice = self._data.loc[pd.IndexSlice[cat, :]]
            _slice = _slice[_slice["original_multiplier"] != 0]
            demands = np.where(
                _slice.index.isin(self._valid_junction_indices),
                _slice["pert_multiplier"],
                _slice["original_multiplier"],
            )

            # epyt fails for unknown reason
            # d.setNodeBaseDemands(_slice.index.values, cat, demands)
            for node_index, dem_mult in zip(_slice.index.values, demands):
                # set via direct API call
                d.api.ENsetbasedemand(index=node_index, demandIdx=cat, value=dem_mult)

        return d


@dataclass(frozen=True)
class DynBaseDemandPertArgs:
    percentage: float
    excluded_junctions: list[str]
    base_seed: int
    frequency: int


@dataclass
class Simulator:
    """
    solver_options: Dict[method, [args]] that will be used to configure the simulation on class level
    e.g. {"setOptionsAccuracyValue": [0.0001]} will be interpreted as:
        getattr(d, "setOptionsAccuracyValue")(*[0.0001])
    """

    EN_USE_TEMP_FILES: ClassVar[bool] = False  # set True to slow down simulation :)

    solver_options: Optional[Dict[str, List[Any]]] = None

    @property
    def _temp_files_flag(self):
        if self.EN_USE_TEMP_FILES:
            return 1
        return 0

    @timer
    def run_simulation(
        self,
        simulation_targets: SimulationTargets,
        cleanup: bool = False,
        inp_path: os.PathLike | None = None,
        epyt_nw: None | epanet = None,
        demand_perturbation_settings: (
            DynBaseDemandPertArgs | PatternPertArgs | None
        ) = None,
    ) -> dict[str, pd.DataFrame]:
        """
        if epyt_nw is provided, skips reading the network
        """
        assert (inp_path is not None) != (
            epyt_nw is not None
        ), "One of path or network has to be provided."
        # load network
        if epyt_nw is None:
            d = epanet(inp_path, display_msg=False, display_warnings=False)
        else:
            d = epyt_nw

        if self.solver_options is not None:
            for method, args in self.solver_options.items():
                try:
                    getattr(d, method)(*args)
                    print(f"Applied {method}({*args,}) option.")
                except Exception as e:
                    print(
                        f"Unable to apply option {method} with args {args} to solver.\n{e}"
                    )

        if inp_path is not None:
            print(f"Loaded epyt instance of: {os.path.basename(inp_path)}")

        # get compound identifiers
        flow_pipe_ids = simulation_targets.get("flow", None)
        if flow_pipe_ids is not None:
            flow_pipe_ids = [p for p in flow_pipe_ids if p in d.getLinkNameID()]

        flow_pump_ids = simulation_targets.get("pump_flow", None)
        if flow_pump_ids is not None:
            flow_pump_ids = [p for p in flow_pump_ids if p in d.getLinkPumpNameID()]

        head_node_ids = simulation_targets.get("head", None)
        if head_node_ids is not None:
            head_node_ids = [n for n in head_node_ids if n in d.getNodeNameID()]

        pressure_node_ids = simulation_targets.get("pressure", None)
        if pressure_node_ids is not None:
            pressure_node_ids = [n for n in pressure_node_ids if n in d.getNodeNameID()]

        # indices
        flow_pipe_indices = (
            d.getLinkIndex(flow_pipe_ids) if flow_pipe_ids is not None else None
        )
        flow_pump_indices = (
            d.getLinkIndex(flow_pump_ids) if flow_pump_ids is not None else None
        )
        head_node_indices = (
            d.getNodeIndex(head_node_ids) if head_node_ids is not None else None
        )
        pressure_node_indices = (
            d.getNodeIndex(pressure_node_ids) if pressure_node_ids is not None else None
        )

        # initialize output
        output = {}
        for k in simulation_targets.keys():
            output[k] = []

        # create storage functions to prevent truth checks for output categories in every simulation loop
        def __store_flow(*args):
            output["flow"].append(d.getLinkFlows(flow_pipe_indices))

        def __store_pump_flow(*args):
            output["pump_flow"].append(d.getLinkFlows(flow_pump_indices))

        def __store_head(*args):
            output["head"].append(d.getNodeHydraulicHead(head_node_indices))

        def __store_pressure(*args):
            output["pressure"].append(d.getNodePressure(pressure_node_indices))

        # create operations queue
        operations_queue = []
        for k in simulation_targets.keys():
            if k == "flow":
                operations_queue.append(__store_flow)

            if k == "pump_flow":
                operations_queue.append(__store_pump_flow)

            if k == "head":
                operations_queue.append(__store_head)

            if k == "pressure":
                operations_queue.append(__store_pressure)

            # if Dynamic base demand
            if isinstance(demand_perturbation_settings, DynBaseDemandPertArgs):
                dbdp = DynamicBaseDemandPerturbation(
                    initial_network=d,
                    perturbation_percentage=demand_perturbation_settings.percentage,
                    exclude_junctions=demand_perturbation_settings.excluded_junctions,
                    seed=demand_perturbation_settings.base_seed,
                    update_interval=demand_perturbation_settings.frequency,
                )
                operations_queue.append(dbdp.pertube_base_demands)

        # if Pattern perturbation
        if isinstance(demand_perturbation_settings, PatternPertArgs):
            assign_perturbated_patterns(
                d=d,
                n=demand_perturbation_settings.n,
                noise_pctg=demand_perturbation_settings.noise_pctg,
                pattern_names=demand_perturbation_settings.pattern_names,
                seed=demand_perturbation_settings.seed,
                skip_constant_patterns=demand_perturbation_settings.skip_constant_patterns,
                skip_nodes=demand_perturbation_settings.skip_nodes,
            )

        # run simulation
        d.openHydraulicAnalysis()
        d.initializeHydraulicAnalysis(self._temp_files_flag)

        tl = []
        pstep = d.getTimePatternStep()

        tstep = 1
        while tstep > 0:

            t = d.runHydraulicAnalysis()

            if t % pstep == 0:
                for op in operations_queue:
                    op(d, t)

                tl.append(t)

            tstep = d.nextHydraulicAnalysisStep()

        d.closeHydraulicAnalysis()

        # only close if loaded from file
        if epyt_nw is None:
            d.closeNetwork()

        # transform output
        for k in output:
            match k:
                case "flow":
                    cols = flow_pipe_ids
                case "pump_flow":
                    cols = flow_pump_ids
                case "head":
                    cols = head_node_ids
                case "pressure":
                    cols = pressure_node_ids

            output[k] = pd.DataFrame(
                output[k], columns=cols, index=pd.Series(pd.to_timedelta(tl, unit="s"))
            )

        if cleanup:
            self._cleanup_sim_dir(inp_path=inp_path)

        return output

    @staticmethod
    def _cleanup_sim_dir(inp_path):
        filename = os.path.splitext(os.path.basename(inp_path))[0]
        directory = os.path.dirname(inp_path)

        for file in os.listdir(directory):
            if file.startswith(filename) and (
                os.path.basename(file) != os.path.basename(inp_path)
            ):
                _path = os.path.join(directory, file)
                os.remove(_path)


@dataclass
class Localiser:
    """
    Args:
        dual_model_inp_path: source path
        patterns: DataFrame with all patterns that are used in the model
        virtual_flow: presumed timeseries for leakage flow; resampled
        virtual_pipes: list of virtual pipes
        pipes_to_test: pipes to test
        temporal_resolution: resolution used for resampling

    Class level:
        N_PROCESSES: number of CPU cores used for parallel localisation

    """

    N_PROCESSES: ClassVar[int] = mp.cpu_count() - 1

    dual_model_inp_path: os.PathLike
    patterns: pd.DataFrame
    virtual_flow: pd.Series
    virtual_pipes: list[str]
    pipes_to_test: list[str] | None = field(default=None, init=True)

    aggregation: Literal["none", "partial", "total", "abs_total"] = "partial"

    # prepared in workflow
    _prepared_network: epanet = field(default=None, init=False)
    _prepared_patterns: pd.DataFrame = field(default=None, init=False)
    _prepared_vf: pd.Series = field(default=None, init=False)

    # result container
    _results: dict[str, Any] = field(default=None, init=False)

    def __post_init__(self):
        self._validate_timeseries_data()

    @property
    def results(self):
        if self._results is None:
            print("No localisation performed yet.")
            return
        return self._results

    def _validate_timeseries_data(self) -> None | Never:
        """Aligning the patterns will be very complex if these do not pass,
        so enforcing them for now is the best choice

        TODO: Find better validation.
        """
        # 1. timedelta or datetime validation
        for attr in ["patterns", "virtual_flow"]:
            df = getattr(self, attr)
            if not isinstance(df.index, (pd.TimedeltaIndex, pd.DatetimeIndex)):
                raise Exception(f"Invalid index, use timedelta or datetime.")
            if isinstance(df, pd.DatetimeIndex):
                df.index = df.index - df.last_valid_index()
                setattr(self, attr, df)

        # 2. data matches each other?
        # either
        check0 = self.patterns.index.equals(self.virtual_flow.index)

        # index is equal after resampling
        try:
            check1 = self.patterns.resample(
                rule=self.virtual_flow.index.inferred_freq
            ).index.equals(self.patterns.index)
        except:
            check1 = False

        if not any(check for check in [check0, check1]):
            raise Exception(f"Invalid timeseries indices provided.")

    def run(self, temporal_resolution: str = "1 h") -> dict[str, Any]:
        """
        Launches parallel localisation simulations

        Problem:
            EPyT expects a file path, that's why copies of the dual model have to be created.
            The pipes_to_test will be distributed evenly to the process workers.
            Once a network is loaded by a process, the simulation doesnt have to reload
            for each iteration and the leak node to test is inserted and removed in a persistent
            epyt.epanet instance.

        Args:
            temporal_resolution: valid string for pd.DateOffset conversion

        """
        d = self._prepare_network(temporal_resolution)

        if self.pipes_to_test is None:
            self.pipes_to_test = [p for p in d.getLinkPipeNameID()]

        with tempfile.TemporaryDirectory() as tdir:
            # copy model
            inp_pathes = self._multiply_inp_file(tdir, d)

            # setup source inp <-> worker mapping
            manager = mp.Manager()
            path_queue = manager.Queue()
            for path in inp_pathes:
                path_queue.put(path)

            # launch execution
            with mp.Pool(
                processes=self.N_PROCESSES,
                initializer=self._initialise_worker,
                initargs=(
                    path_queue,
                    self.virtual_pipes,
                    self.aggregation,
                ),
            ) as pool:
                results = pool.imap_unordered(
                    self._run_worker, self.pipes_to_test, chunksize=16
                )

                result_dict = {name: result for name, result in results}

        self._results = result_dict

        return self._results

    def _prepare_network(self, tr: pd.DateOffset):
        # Open Network
        d: epanet = epanet(self.dual_model_inp_path)

        pattern_ids = d.getPatternNameID()
        pattern_ids_set = set(pattern_ids)
        pattern_df_ids = set(self.patterns.columns.to_list())

        self._validate_pattern_ids(
            network_patterns=pattern_ids_set, dataframe_patterns=pattern_df_ids
        )

        # Prepare data
        tr = self._determine_tr(tr)  # == resampling rule
        self._prepared_patterns = (
            self.patterns.resample(rule=tr).mean().loc[:, pattern_ids]
        )  # network pattern id order
        self._prepared_vf = self.virtual_flow.resample(rule=tr).mean()

        # Set patterns
        d.setPatternMatrix(self._prepared_patterns.T.values)
        d.addPattern(VIRTUAL_FLOW_PATTERN_NAME, self._prepared_vf.values)

        # Adjust timesteps
        tstep = self._tdidx_frq_as_seconds(self._prepared_patterns.index)
        d.setTimeHydraulicStep(tstep)
        d.setTimePatternStep(tstep)
        d.setTimeQualityStep(tstep)
        d.setTimeReportingStep(tstep)

        # adjust duration
        d.setTimeSimulationDuration(
            self._tdidx_duration_as_seconds(self._prepared_patterns.index)
        )

        return d

    def _determine_tr(self, tres: str) -> pd.DateOffset | Never:
        """Retrieve DateOffset for resampling
        If provided frequency is lower than native, using maximum native frequency.
        """
        is_valid_offset, tr = _to_offset(tres)

        if not is_valid_offset:
            raise Exception(f"Invalid temporal resolution '{tres}'.")

        _tr_adjusted = False

        for df in [self.patterns, self.virtual_flow]:
            inferred_frq = df.index.inferred_freq
            _valid, inferred_offset = _to_offset(inferred_frq)
            if not _valid:
                raise Exception(f"Invalid offset string {inferred_frq}.")

            # if native resolution is already longer than provided resolution, use that one
            if inferred_offset > tr:
                tr = inferred_offset

        if _tr_adjusted:
            print(
                f"Native frequency of either patterns or virtual flows is higher than provided temporal resolution, falling back to '{tr}'."
            )

        return tr

    def _get_tempdir_inp_pathes(self, tdir: os.PathLike) -> List[os.PathLike]:
        names = []
        for i in range(self.N_PROCESSES):
            _path = os.path.join(tdir, f"{i}.inp")
            names.append(_path)
        return names

    def _multiply_inp_file(self, tdir: os.PathLike, nw: epanet) -> list[os.PathLike]:
        pathes = self._get_tempdir_inp_pathes(tdir)
        for path in pathes:
            nw.saveInputFile(path)
        return pathes

    @staticmethod
    def _validate_pattern_ids(
        network_patterns: set[str], dataframe_patterns: set[str]
    ) -> None | Never:
        if network_patterns != dataframe_patterns:
            print(
                f"Pattern names in network patterns and provided dataframe must match."
            )
            _in_nw_not_in_df = network_patterns - dataframe_patterns
            _in_df_not_in_nw = dataframe_patterns - network_patterns
            if _in_nw_not_in_df:
                print(f"Missing in provided pattern data: {*_in_nw_not_in_df,}")
            if _in_df_not_in_nw:
                print(f"Missing in provided network patterns: {*_in_df_not_in_nw,}")
            raise Exception(f"Pattern IDs do not match.")

    @staticmethod
    def _tdidx_frq_as_seconds(td_index: pd.TimedeltaIndex) -> int:
        _, offset = _to_offset(td_index.inferred_freq)
        return int(pd.Timedelta(offset).total_seconds())

    @staticmethod
    def _tdidx_duration_as_seconds(td_index: pd.TimedeltaIndex) -> int:
        return int((td_index[-1] - td_index[0]).total_seconds())

    @staticmethod
    def _initialise_worker(
        path_queue: mp.Queue,
        virtual_pipes: List[str],
        aggregation: str,
    ) -> _LocalisationWorker:
        global _WORKER
        source_file = path_queue.get()
        _WORKER = _LocalisationWorker(
            inp_path=source_file, virtual_pipes=virtual_pipes, aggregation=aggregation
        )
        print(f"Worker initialized with {source_file}")
        return _WORKER

    @staticmethod
    def _run_worker(pipe_id: str) -> pd.Series:
        global _WORKER
        return pipe_id, _WORKER.run(pipe_id)


# Process workers
_WORKER = None  # global container for worker persistence

_DEBUG_INP_FOLDER = os.path.normpath(
    r"C:\Users\jkoslo\Documents\iOLE (lok)\programming\iole_kwb\iole\data\temp\debug_inp"
)


@dataclass
class _LocalisationWorker:

    debug_folder: ClassVar[os.PathLike] = _DEBUG_INP_FOLDER
    debug: ClassVar[bool] = False

    inp_path: os.PathLike
    virtual_pipes: List[str]
    aggregation: Literal["none", "partial", "total", "abs_total"]

    _d: epanet = field(default=None, init=False)
    _simulator: Simulator = field(default_factory=Simulator, init=False)
    _pipes: list[str] = field(default=None, init=False)

    def __post_init__(self):
        self._open_network()

    @property
    def pressure_nodes(self):
        return [vp.replace("vp_", "") for vp in self.virtual_pipes]

    def _open_network(self):
        self._d = epanet(self.inp_path)

    def run(self, pipe_id: str) -> pd.DataFrame | pd.Series | float:
        """Run single simulation with vf as demandpattern inserted in a pipe
        and return aggregated virtual flow (axis=1 only)
        """

        if self.debug:
            saveat = os.path.join(self.debug_folder, f"{pipe_id}.inp")
        else:
            saveat = None

        with self._live_insert_leak_node_epyt(
            self._d, pipe_id=pipe_id, save_location=saveat
        ) as nw:
            sim_result = self._simulator.run_simulation(
                epyt_nw=nw,
                simulation_targets={
                    "flow": self.virtual_pipes,
                    "head": self.pressure_nodes,
                },
            )

            vf = sim_result["flow"]
            heads = sim_result["head"]

            match self.aggregation:
                case "none":
                    pass
                case "partial":
                    vf = vf.sum(axis=1)
                case "total":
                    vf = vf.sum(axis=1).sum()
                case "abs_total":
                    vf = vf.sum(axis=1).abs().sum()

        return vf, heads

    @contextmanager
    def _live_insert_leak_node_epyt(
        self, nw: epanet, pipe_id: str, save_location: Optional[os.PathLike] = None
    ):
        # get index
        idx = nw.getLinkIndex(pipe_id)

        # initially connected nodes
        node_ids = nw.getNodesConnectingLinksID(idx)[0]

        # grab original pipe properties
        length = nw.getLinkLength(idx)
        diameter = nw.getLinkDiameter(idx)
        roughness = nw.getLinkRoughnessCoeff(idx)
        minor_loss = nw.getLinkMinorLossCoeff(idx)

        # split pipe
        temporary_node_name = f"{pipe_id}_TEST_NODE"
        nw.splitPipe(pipe_id, f"{pipe_id}_TEMP", temporary_node_name)
        pipenode_idx = nw.getNodeIndex(temporary_node_name)

        # set demands
        pidx = nw.getPatternIndex(VIRTUAL_FLOW_PATTERN_NAME)
        nw.setNodeDemandPatternIndex(pipenode_idx, pidx)
        nw.setNodeBaseDemands(pipenode_idx, 1)

        try:
            yield nw

        finally:
            if save_location is not None:
                nw.saveInputFile(save_location)

            nw.deleteNode(pipenode_idx)
            nw.addLinkPipe(
                pipe_id,
                *node_ids,
                length,
                diameter,
                roughness,
                minor_loss,
            )
