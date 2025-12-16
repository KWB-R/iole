from __future__ import annotations
import os
from copy import deepcopy

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, ClassVar, Literal, Self, Dict, Any
import tempfile

import pandas as pd
import numpy as np
import oopnet as on

from .configuration import (
    ARTIFICIAL_LEAK_PREFIX,
    SPLIT_PIPE_SUFFIX,
    SUBSTITUTE_PUMP_LINK_PREFIX,
    SUBSTITUTE_INFLOW_PATTERN_SUFFIX,
)
from .network_simulation import (
    Simulator,
    SimulationTargets,
)

from ..util.data_processing import wrap_cyclic_dataframe

## SUPER IMPORTANT MONKEY PATCH INCLUDED AT MODULE LEVEL ##############
import src.util.oopnet_patch

#######################################################################


@dataclass
class VirtualReservoir:
    """TODO: Needs to be rewritten, I had a different idea in mind earlier...
    on.Network = VirtualReservoir.add(on.Network, *vr_ids) works as is but it's weird syntax
    """

    vr_prefix: ClassVar[str] = "vr_"  # virtual reservoirs
    vp_prefix: ClassVar[str] = "vp_"  # virtual pipes from sensor to VN
    flow_corr_pattern_suffix: ClassVar[str] = "_flow_corr"  # flow corr patterns

    REGISTRY: ClassVar[dict[str, VirtualReservoir]] = {}

    # new in nw
    connected_node_id: str

    # new elements
    reservoir: on.Reservoir = field(default=None, init=False)
    pipe: on.Pipe = field(default=None, init=False)
    in_nw: on.Network = field(default=None, init=False)

    @property
    def vr_id(self):
        return f"{self.vr_prefix}{self.connected_node_id}"

    @property
    def vp_id(self):
        return f"{self.vp_prefix}{self.connected_node_id}"

    @property
    def flow_corr_pat_id(self):
        return (
            f"{self.vr_prefix}{self.connected_node_id}{self.flow_corr_pattern_suffix}"
        )

    @property
    def head_pat_id(self):
        return f"{self.connected_node_id}"

    @classmethod
    def add(cls, nw: on.Network, *connected_nodes: str) -> Tuple[List[str], List[str]]:

        vrids = []
        vpids = []

        _nw_nodes = on.get_node_ids(nw)

        for nid in connected_nodes:

            new_vr = cls(connected_node_id=nid)

            if new_vr.connected_node_id not in _nw_nodes:
                print(f"No node {new_vr.connected_node_id} found in network.")
                return nw

            if new_vr.vr_id in _nw_nodes:
                print(f"Node {new_vr.vr_id} already exists in network.")
                return nw

            # Get sensor node
            connected_node: on.Junction = on.get_node(nw, new_vr.connected_node_id)

            # Create VR
            new_vr.reservoir = on.Reservoir(
                id=new_vr.vr_id,
                head=1,
                xcoordinate=connected_node.xcoordinate + 50 * np.sin(np.radians(45)),
                ycoordinate=connected_node.ycoordinate + 50 * np.cos(np.radians(45)),
                elevation=connected_node.elevation,
                headpattern=on.Pattern(id=new_vr.vr_id),
            )

            # Insert VR
            on.add_node(nw, new_vr.reservoir)

            # Create VP
            # Connected Node <VP> VR
            new_vr.pipe = on.Pipe(
                id=new_vr.vp_id,
                diameter=100,
                length=1,
                roughness=130,
                startnode=connected_node,
                endnode=new_vr.reservoir,
            )

            # Connect VR
            on.add_pipe(nw, new_vr.pipe)

            # Create VR Head Pattern
            if new_vr.connected_node_id in on.get_pattern_ids(nw):
                print(f"Pattern {new_vr.connected_node_id} already exists in network.")
                return nw

            # Create connected Node flow correction pattern
            flowpat = on.Pattern(id=new_vr.flow_corr_pat_id, multipliers=[0])
            on.add_pattern(nw, flowpat)

            connected_node.demand = (
                connected_node.demand + [1]
                if isinstance(connected_node.demand, list)
                else [connected_node.demand, 1]
            )
            connected_node.demandpattern = (
                connected_node.demandpattern + [flowpat]
                if isinstance(connected_node.demandpattern, list)
                else [connected_node.demandpattern, flowpat]
            )

            # connect nw to instance
            new_vr.in_nw = nw

            vrids.append(new_vr.vr_id)
            vpids.append(new_vr.vp_id)

            cls.REGISTRY[connected_node.id] = new_vr

        return vrids, vpids


@dataclass
class HydraulicNetwork:

    """Class that holds the on.Network
    contains methods that change topology to
    convert into _DualModel

    Args:
        source_path: path to inp file
        nw: on.Network container
        base_patterns: if not specified, all patterns in inp
        inflow_pipes: pipes that connect to a tank or reservoir

    """

    source_path: os.PathLike | None = field(default=None)
    nw: on.Network | None = field(default=None)

    base_patterns: pd.DataFrame | None = field(default=None)
    pump_demands: list[str] = field(default_factory=list, init=False)
    inflow_pipes: list[str] = field(default_factory=list, init=True)

    def __post_init__(self):
        if (self.source_path is None) and (self.nw is None):
            raise ValueError(
                "One of on.Network or a path to an inp-file has to be provided."
            )

        if self.source_path:
            self.nw = on.Network.read(self.source_path)

        if self.base_patterns is None:
            self.base_patterns = self.get_pattern_df()

    def __repr__(self):
        return f"{type(self).__name__}: {self.nw.title or 'unnamed network'}"

    def __str__(self):
        return self.__repr__()

    def _repr_pretty_(self):
        return self.__repr__()

    @property
    def max_pattern_steps(self):
        return np.max([len(p.multipliers) for p in on.get_patterns(self.nw)])

    @property
    def pattern_index(self):
        """The pattern index is the timedelta index
        that is made from the longest pattern
        so that after that time the simulation
        patterns are cyclic.
        """

        lmax = self.max_pattern_steps
        ptimestep = pd.Timedelta(self.nw.times.patterntimestep)
        pstarttime = pd.Timedelta(self.nw.times.patternstart)

        return pd.timedelta_range(pstarttime, periods=lmax, freq=ptimestep)

    def check_pattern_compatibility(self, duration_assumption: pd.Timedelta):
        dur = self.max_pattern_steps * pd.Timedelta(self.nw.times.patterntimestep)
        if dur != duration_assumption:
            raise ValueError(f"Pattern duration does not match assumed cycle length.")

    def copy(self) -> HydraulicNetwork:
        return deepcopy(self)

    def save(self, path):
        self.nw.write(path)

    def export_simulation_slice(
        self, start: Optional[pd.Timedelta] = None, end: Optional[pd.Timedelta] = None
    ) -> HydraulicNetwork:
        assert (start is not None) or (end is not None), "Provide at least one."

        patterns = self.get_pattern_df().copy()
        new_patterns = patterns.loc[pd.IndexSlice[start:end]]

        new_instance = self.copy()
        new_instance.set_pattern_df(new_patterns)

        return new_instance

    def set_pattern_df(
        self, df: pd.DataFrame, mode: Literal["full", "update"] = "full"
    ):
        """
        full: uses pattern df as new patterns, df needs to include all pattern names
        update: updates patterns with provided timeseries (resampled if no match with simulation timestep)
        """
        match mode:
            case "full":
                return self._set_full_patterns(df=df)
            case "update":
                return self._update_patterns_from_df(df=df)

    def _set_full_patterns(self, df: pd.DataFrame) -> Self:
        assert isinstance(df.index, pd.TimedeltaIndex), "Wrong index format."

        pattern_ids = on.get_pattern_ids(self.nw)

        assert all(
            c in df.columns for c in pattern_ids
        ), "Not all patterns found in provided dataframe."

        df_freq = (
            pd.Timedelta(pd.infer_freq(df.index)).total_seconds()
            if df.index.freq is None
            else pd.Timedelta(df.index.freq).total_seconds()
        )

        df_duration = (df.last_valid_index() - df.first_valid_index()).total_seconds()

        for pattern in on.get_patterns(self.nw):
            vals = df[pattern.id].values
            pattern.multipliers = vals

        self.nw.times.patterntimestep = pd.Timedelta(seconds=df_freq)
        self.nw.times.duration = pd.Timedelta(seconds=df_duration)

        return self

    def _update_patterns_from_df(self, df: pd.DataFrame) -> Self:
        """This will automatically resample the df if needed"""
        ptstep = self.nw.times.patterntimestep
        df_freq = (
            pd.Timedelta(pd.infer_freq(df.index)).total_seconds()
            if df.index.freq is None
            else pd.Timedelta(df.index.freq).total_seconds()
        )

        if ptstep.total_seconds() != df_freq:
            _df = df.resample(rule=ptstep).mean()
            print(
                f"Warning! Simulation timestep is {ptstep.total_seconds():.0f}s, resampling provided patterns before insertion."
            )
        else:
            _df = df.copy()

        pnames = on.get_pattern_ids(self.nw)
        for name, timeseries in _df.items():
            if name in pnames:
                pattern = on.get_pattern(self.nw, id=name)
                pattern.multipliers = timeseries.values
            else:
                pattern = on.Pattern(id=name, multipliers=timeseries.values)
                on.add_pattern(self.nw, pattern)

        return self

    def get_pattern_df(self) -> pd.DataFrame:
        patterns = on.get_patterns(self.nw)

        pindex = self.pattern_index

        lmax = len(pindex)

        pattern_dict = {}

        for pattern in patterns:
            l = len(pattern.multipliers)
            n_reps = lmax // l + 1

            if isinstance(pattern.multipliers, list):
                pmults = pattern.multipliers
            elif isinstance(pattern.multipliers, np.ndarray):
                pmults = pattern.multipliers.tolist()
            elif isinstance(pattern.multipliers, pd.Series):
                pmults = pattern.multipliers.to_list()
            else:
                raise TypeError(
                    f"Unexpected type {type(pattern.multipliers).__name__} for pattern multipliers in pattern."
                )

            p_series = pd.Series(n_reps * pmults).iloc[:lmax]
            pattern_dict[pattern.id] = p_series

        df = pd.DataFrame.from_dict(pattern_dict)
        df.index = pindex

        return df

    def replace_reservoir_or_tank_with_emitter_node(
        self,
        reservoir_or_tank_id: str,
        demand_pattern: pd.Series | str,
        pipes_to_reconnect: Optional[List[str]] = None,
    ) -> None:
        """
        reservoir_or_tank_id: str = id of reservoir or tank
        demand_pattern_series: pd.Series = new demand pattern, idx has to match self.pattern_index
        pipes_to_reconnect: if provided, will only reconnect the named pipe to the new emitter node
                            if blank, will reconnect all original components to the new node
        delete_unconnected_components: bool will delete all components that are connected to the
                                            pipes that werent specified in pipes_to_reconnect
        """
        if isinstance(demand_pattern, pd.Series):
            assert len(demand_pattern) == len(
                self.pattern_index
            ), "Index of provided timeseries does not match general network pattern index."
            values = demand_pattern.values
        elif isinstance(demand_pattern, str):
            values = [0]

        _r = on.get_node(network=self.nw, id=reservoir_or_tank_id)

        # make pattern
        dempat = on.Pattern(
            id=f"{reservoir_or_tank_id}{SUBSTITUTE_INFLOW_PATTERN_SUFFIX}",
            multipliers=values,
        )

        on.add_pattern(self.nw, dempat)

        new_node = on.Junction(
            id=f"{reservoir_or_tank_id}_SURROGATE",
            demand=-1,
            demandpattern=dempat,
            elevation=_r.elevation,
            comment=_r.comment,
            tag=_r.tag,
            xcoordinate=_r.xcoordinate,
            ycoordinate=_r.ycoordinate,
        )
        _r_pipes = on.get_adjacent_links(self.nw, query_node=_r)

        # 1. delete reservoir or tank
        on.remove_node(self.nw, reservoir_or_tank_id)

        # 2. place new node
        on.add_junction(self.nw, new_node)

        # DRY func
        def __set_new_connected_node(p, rtid):
            if p.startnode.id == rtid:
                setattr(p, "startnode", new_node)
            elif p.endnode.id == rtid:
                setattr(p, "endnode", new_node)
            else:
                raise Exception(
                    f"Something went wrong, queried pipe {p.id} not connected to {rtid}."
                )

        # 3. replace connections
        if pipes_to_reconnect is None:
            for p in _r_pipes:
                __set_new_connected_node(p, reservoir_or_tank_id)

        else:
            _removed_links = []  # contains IDs
            for p in _r_pipes:
                if p.id in pipes_to_reconnect:
                    __set_new_connected_node(p, reservoir_or_tank_id)

                else:
                    on.remove_link(self.nw, p.id)
                    _removed_links.append(p.id)

            # EPANET fails e.g. if control rules exist that reference nonexistant elements
            # sanitize controls
            if (self.nw.__dict__["controls"] is not None) or (
                len(self.nw.__dict__["controls"]) > 0
            ):
                _del_control_indices = []
                for i, ctrl in enumerate(on.get_controls(self.nw)):
                    aid = ctrl.action.object.id
                    cid = ctrl.condition.object.id
                    if any(o in _removed_links for o in [aid, cid]):
                        _del_control_indices.append(i)

                # remove flagged controls
                self.nw.__dict__["controls"] = [
                    c
                    for i, c in enumerate(self.nw.__dict__["controls"])
                    if i not in _del_control_indices
                ]

    def _clean_network(self, patterns: bool = False, controls: bool = False):
        if patterns:
            # 1. find unused patterns
            _pids = set()
            for node in on.get_junctions(self.nw):
                if node.demandpattern is not None:
                    if isinstance(node.demandpattern, list):
                        for p in node.demandpattern:
                            _pids.add(p.id)
                    if isinstance(node.demandpattern, on.Pattern):
                        _pids.add(node.demandpattern.id)

            for pattern in on.get_pattern_ids(self.nw):
                if pattern not in _pids:
                    on.remove_pattern(self.nw, id=pattern)

        if controls:
            # 2. find undefined objects in control section
            nids = on.get_node_ids(self.nw)
            lids = on.get_link_ids(self.nw)

            ctrls = on.get_controls(self.nw)

            controls_to_keep = []

            for control in ctrls:
                _keep = True
                for j in [control.action, control.condition]:
                    obj = j.object

                    if not any(obj.id in l for l in [nids, lids]):
                        _keep = False

                if _keep:
                    controls_to_keep.append(control)

            self.nw.controls = controls_to_keep

    def split_area_at_pump(self, pump_id, substitute_demand_pattern: pd.Series | str):
        # sets empty pattern multipliers if pattern is str
        # get pump

        if isinstance(substitute_demand_pattern, pd.Series):
            pname = substitute_demand_pattern.name
            pvalues = substitute_demand_pattern.values
        elif isinstance(substitute_demand_pattern, str):
            pname = substitute_demand_pattern
            pvalues = [1]
        else:
            raise TypeError(f"Invalid argument {substitute_demand_pattern}.")

        pump = on.get_pump(self.nw, pump_id)

        pump_x = (pump.startnode.xcoordinate + pump.endnode.xcoordinate) / 2
        pump_y = (pump.startnode.ycoordinate + pump.endnode.ycoordinate) / 2

        # add Reservoir
        new_reservoir = on.Reservoir(
            id=f"Reservoir_{pump_id}",
            xcoordinate=pump_x,
            ycoordinate=pump_y + 5,
            elevation=pump.endnode.elevation,
        )

        on.add_reservoir(self.nw, new_reservoir)

        # add substitute demand pattern
        new_pattern = on.Pattern(
            id=pname,
            multipliers=pvalues,
        )

        on.add_pattern(self.nw, new_pattern)

        # add substitute node
        new_node = on.Junction(
            id=pname,
            elevation=pump.startnode.elevation,
            xcoordinate=pump_x,
            ycoordinate=pump_y - 5,
            demand=1,
            demandpattern=new_pattern,
        )

        on.add_junction(
            self.nw,
            new_node,
        )

        # add substitute pipe
        old_pump_startnode = pump.startnode
        pump.startnode = new_reservoir

        new_pipe = on.Pipe(
            id=f"{SUBSTITUTE_PUMP_LINK_PREFIX}{pname}",
            startnode=old_pump_startnode,
            endnode=new_node,
            length=1,
            roughness=150,
            diameter=1000,
        )

        on.add_pipe(self.nw, new_pipe)
        self.pump_demands.append(pump_id)

    def to_dual_model(self, node_names: List[str]) -> _DualModel:

        nw_copy = deepcopy(self.nw)

        vrs, vps = VirtualReservoir.add(nw_copy, *node_names)

        dm = _DualModel(
            source_path=None,
            nw=nw_copy,
            virtual_pipes=vps,
            virtual_reservoirs=vrs,
            base_patterns=self.base_patterns,
        )

        return dm

    def set_pattern_series(self, pattern_series: pd.Series):
        name = pattern_series.name
        if name in on.get_pattern_ids(self.nw):
            pattern = on.get_pattern(self.nw, name)
            pattern.multipliers = pattern_series.values
        else:
            pattern = on.Pattern(id=name, multipliers=pattern_series.values)
            on.add_pattern(self.nw, pattern)

    def set_patterns(self, patterns: pd.DataFrame, overwrite: bool = True):
        """Set and add or just add patterns"""
        _patterns = on.get_pattern_ids(self.nw)

        for name, data in patterns.items():
            if name in _patterns:
                if overwrite:
                    _pattern = on.get_pattern(self.nw, name)
                    _pattern.multipliers = data
                else:
                    print(f"Pattern {name} already in network.")

            else:
                _pattern = on.Pattern(name, multipliers=data)
                on.add_pattern(self.nw, _pattern)

    def get_pattern_dict(self) -> dict[str, pd.Series]:
        return {
            p.id: pd.Series(p.multipliers).rename(p.id)
            for p in on.get_patterns(self.nw)
        }

    def wrap_base_patterns(
        self, start_timestamp: pd.Timestamp, pattern_ids: list[str], start_dow: int = 0
    ):
        """Overwrites network patterns with wrapped base patterns
        Does not affect base_pattern attr.
        """
        _df = wrap_cyclic_dataframe(
            pattern_df=self.base_patterns[pattern_ids].copy(),
            pattern_start_dayofweek=start_dow,
            start_timestamp=start_timestamp,
        )

        self.set_patterns(_df, overwrite=True)

    def run_simulation(
        self,
        simulation_targets: SimulationTargets,
        solver_options: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[SimulationTargets, pd.DataFrame]:
        """Run simulation using EpytSimulation class
        - slices result if indices do not match
        """

        simulator = Simulator(solver_options=solver_options)

        with tempfile.TemporaryDirectory() as tdir:
            tpath = os.path.join(tdir, "dual_model.inp")
            self.save(tpath)
            result = simulator.run_simulation(
                inp_path=tpath,
                simulation_targets=simulation_targets,
                cleanup=True,
            )

        return result


@dataclass
class _DualModel(HydraulicNetwork):

    virtual_reservoirs: list[str] = field(default_factory=list)
    virtual_pipes: list[str] = field(default_factory=list)

    correction_pattern_ids: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.correction_pattern_ids = [p for p in on.get_pattern_ids(self.nw) if VirtualReservoir.flow_corr_pattern_suffix in p]
        return super().__post_init__()

    def run_localisation(
        self,
        leak_flow: pd.Series,
        heads: pd.DataFrame,
        inflows: pd.DataFrame,
        temporal_resolution: Optional[str | pd.DateOffset] = None,
        pipe_list: Optional[list[str]] = None,
    ) -> dict[str, pd.Series | pd.DataFrame]:
        raise NotImplementedError("Localisation not yet implemented.")



