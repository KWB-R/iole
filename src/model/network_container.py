from __future__ import annotations
import os
from copy import deepcopy

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, ClassVar, Callable, Literal, Self, Dict, Any
from functools import wraps

import pandas as pd
import numpy as np
import oopnet as on
import networkx as nx

from .configuration import ARTIFICIAL_LEAK_PREFIX, SPLIT_PIPE_SUFFIX
from .network_simulation import (
    Simulator,
    SimulationTargets,
    DynBaseDemandPertArgs,
    PatternPertArgs,
)

## SUPER IMPORTANT MONKEY PATCH INCLUDED AT MODULE LEVEL ##############
import src.util.oopnet_patch
#######################################################################


# decorators to track if changes to network were saved
def set_unsaved(func):
    @wraps(func)
    def inner(self: HydraulicNetwork, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._changes_saved = False
        return result

    return inner


def set_saved(func):
    @wraps(func)
    def inner(self: HydraulicNetwork, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._changes_saved = True
        return result

    return inner


@dataclass
class VirtualReservoir:
    """TODO: Needs to be rewritten, I had a different idea in mind earlier...
    on.Network = VirtualReservoir.add(on.Network, *vr_ids) works as is but it's weird syntax
    """

    _vr_prefix: ClassVar[str] = "vr_"  # virtual reservoirs
    _vp_prefix: ClassVar[str] = "vp_"  # virtual pipes from sensor to VN
    _vn_prefix: ClassVar[str] = "vn_"  # virtual nodes
    _flow_corr_pattern_suffix: ClassVar[str] = "_flow_corr"  # flow corr patterns

    REGISTRY: ClassVar[List[VirtualReservoir]] = {}

    # new in nw
    connected_node_id: str

    # new elements
    reservoir: on.Reservoir = field(default=None, init=False)
    pipe: on.Pipe = field(default=None, init=False)
    in_nw: on.Network = field(default=None, init=False)

    @property
    def vr_id(self):
        return f"{self._vr_prefix}{self.connected_node_id}"

    @property
    def vp_id(self):
        return f"{self._vp_prefix}{self.connected_node_id}"

    @property
    def flow_corr_pat_id(self):
        return f"{self.connected_node_id}{self._flow_corr_pattern_suffix}"

    @classmethod
    def add(
        cls, nw: on.Network, *connected_nodes: str
    ) -> Tuple[on.Network, List[str], List[str]]:

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
                headpattern=on.Pattern(id=new_vr.connected_node_id),
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

    source_path: os.PathLike

    _leak_nodes: List[str] = field(default_factory=list, init=False)

    _changes_saved: bool = field(default=False, init=False)  # toggled by decorator
    _most_recent_path: os.PathLike = field(
        default=None, init=False
    )  # set in .save method

    def __post_init__(self):
        self.nw = on.Network.read(self.source_path)
        self._most_recent_path = self.source_path

    @property
    def inp_path(self):
        """Returns path to most recently saved inp file"""
        return self._most_recent_path

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

    def copy(self) -> HydraulicNetwork:
        return deepcopy(self)

    @set_saved
    def save(self, path) -> None:
        self.nw.write(path)
        self._most_recent_path = path
        return self._most_recent_path

    def export_simulation_slice(
        self, start: Optional[pd.Timedelta] = None, end: Optional[pd.Timedelta] = None
    ) -> HydraulicNetwork:
        assert (start is not None) or (end is not None), "Provide at least one."

        patterns = self.get_pattern_df().copy()
        new_patterns = patterns.loc[pd.IndexSlice[start:end]]

        new_instance = self.copy()
        new_instance.set_pattern_df(new_patterns)

        return new_instance

    @set_unsaved
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

    @set_unsaved
    def transform_base_demands(
        self, func: Callable, node_list: Optional[List[str]] = None
    ) -> None:
        """Transforms all base demands with the provided function
        func: f(x) -> x*
        """

        _nl = on.get_junction_ids(self.nw) if node_list is None else node_list

        for node in on.get_junctions(self.nw):
            if node.id in _nl:
                if isinstance(node.demand, list):
                    new_demands = [func(d) for d in node.demand]
                    node.demand = new_demands
                elif isinstance(node.demand, float):
                    node.demand = func(node.demand)
                else:
                    print(f"No base demand information for {node.id}.")

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

    @set_unsaved
    def replace_reservoir_or_tank_with_emitter_node(
        self,
        reservoir_or_tank_id: str,
        demand_pattern_series: pd.Series,
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
        assert len(demand_pattern_series) == len(
            self.pattern_index
        ), "Index of provided timeseries does not match general network pattern index."

        _r = on.get_node(network=self.nw, id=reservoir_or_tank_id)

        # make pattern
        dempat = on.Pattern(
            id=f"{reservoir_or_tank_id}_SURROGATE_PATTERN",
            multipliers=demand_pattern_series.values,
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

    @set_unsaved
    def add_leaks(self, *leaks: pd.Series):
        if not leaks:
            return None

        pipe_ids = on.get_pipe_ids(self.nw)
        pidx = self.pattern_index
        pfreq = pd.Timedelta(pidx.freq)

        for l in leaks:
            if l.name not in pipe_ids:
                print(f"{l.name} not found in network pipe ids, skipping.")
                continue

            if not isinstance(l.index, pd.TimedeltaIndex):
                print(f"Index of {l.name} is not pd.TimedeltaIndex, skipping.")
                continue

            lfreq = (
                pd.Timedelta(l.index.freq)
                if l.index.freq is not None
                else pd.Timedelta(pd.infer_freq(l.index))
            )
            if lfreq is None:
                print(f"Freq of {l.name} is not None, skipping.")
                continue

            if lfreq != pfreq:
                print(
                    f"Freq of {l.name} ({lfreq}) does not match pattern freq ({pfreq}), skipping."
                )
                continue

            if len(pidx) > len(l):
                _l = l.copy().reindex(pidx).fillna(0)
            else:
                _l = l.copy()

            self._insert_leak(_l)

        # adjust sim time
        new_sim_duration = self.max_pattern_steps * pfreq
        self.nw.times.duration = new_sim_duration

        return self

    def _insert_leak(self, leak: pd.Series):

        leak_node_id = f"{ARTIFICIAL_LEAK_PREFIX}{leak.name}"
        split_pipe_name = f"{leak.name}{SPLIT_PIPE_SUFFIX}"

        if leak_node_id not in on.get_node_ids(self.nw):
            try:

                leak_pipe = on.get_pipe(self.nw, leak.name)
                sn, en = leak_pipe.startnode, leak_pipe.endnode

                # add pattern
                new_pattern = on.Pattern(id=leak_node_id, multipliers=leak.values)
                on.add_pattern(self.nw, new_pattern)

                # define leak node
                leak_node = on.Junction(
                    id=leak_node_id,
                    elevation=(sn.elevation + en.elevation) / 2,
                    demand=1,
                    demandpattern=on.Pattern(leak_node_id),
                    xcoordinate=(sn.xcoordinate + en.xcoordinate) / 2,
                    ycoordinate=(sn.ycoordinate + en.ycoordinate) / 2,
                )
                on.add_junction(self.nw, leak_node)

                # reconnect old pipe
                leak_pipe.endnode = leak_node
                leak_pipe.length /= 2

                # define new pipe
                new_pipe = on.Pipe(
                    id=split_pipe_name,
                    length=leak_pipe.length,
                    diameter=leak_pipe.diameter,
                    roughness=leak_pipe.roughness,
                    minorloss=leak_pipe.minorloss,
                    startnode=leak_node,
                    endnode=en,
                )

                # add new pipe
                on.add_pipe(self.nw, new_pipe)

                # save leak_node props
                self._leak_nodes.append(
                    {
                        "node_id": leak_node.id,
                        "original_pipe": {
                            "id": leak.name,
                            "start_node": sn,
                            "end_node": en,
                        },
                        "split_pipe": {"id": split_pipe_name},
                    }
                )

                print(f"Inserted leak at node id '{leak_node_id}'.")

            except Exception as e:
                raise e

        else:
            print(f"Node '{leak_node_id}' already in network, no leak inserted.")

    @set_unsaved
    def remove_artificial_leaks(self):
        for ln in self._leak_nodes:
            # remove elements
            on.remove_node(self.nw, ln["node_id"])
            on.remove_pipe(self.nw, ln["split_pipe"]["id"])

            # reconnect original pipe
            original_pipe = on.get_pipe(self.nw, ln["original_pipe"]["id"])
            original_pipe.length = original_pipe.length * 2
            original_pipe.startnode = ln["original_pipe"]["start_node"]
            original_pipe.endnode = ln["original_pipe"]["end_node"]

            # remove leakage pattern
            self.nw._patterns.pop(
                f"{ARTIFICIAL_LEAK_PREFIX}{ln["original_pipe"]["id"]}", None
            )

        self._leak_nodes = []

    @set_unsaved
    def split_area_at_pump(self, pump_id, substitute_demand_pattern: pd.Series):
        # get pump
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
            id=substitute_demand_pattern.name,
            multipliers=substitute_demand_pattern.values,
        )

        on.add_pattern(self.nw, new_pattern)

        # add substitute node
        new_node = on.Junction(
            id=substitute_demand_pattern.name,
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
            id=f"p_{substitute_demand_pattern.name}",
            startnode=old_pump_startnode,
            endnode=new_node,
            length=1,
            roughness=150,
            diameter=1000,
        )

        on.add_pipe(self.nw, new_pipe)

    @set_unsaved
    def add_virtual_reservoirs(
        self, *node_names: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Returns ids of
            <VP> VN <VP2>
        """

        vrids, vpids = VirtualReservoir.add(self.nw, *node_names)

        return vrids, vpids

    @set_unsaved
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

    def export_base_demands(self) -> pd.DataFrame:
        demands = {}

        for node in on.get_junctions(self.nw):
            demands[node.id] = np.array([node.demand]).flatten()

        return pd.DataFrame.from_dict(demands, orient="index")

    @set_unsaved
    def import_base_demands(
        self, base_demands: pd.DataFrame, exclude: List[str] = None
    ) -> None:
        if exclude is None:
            exclude = []
        for nid, bdm in base_demands.replace(np.nan, 0).iterrows():
            if nid not in exclude:
                on.get_node(self.nw, nid).demand = bdm.to_list()

    def export_roughness(self) -> pd.DataFrame:
        roughness = {}

        for pipe in on.get_pipes(self.nw):
            roughness[pipe.id] = pipe.roughness

        return pd.DataFrame.from_dict(roughness, orient="index")

    @set_unsaved
    def import_roughness(
        self, roughness_df: pd.DataFrame, exclude: List[str] = None
    ) -> None:
        if exclude is None:
            exclude = []
        for pid, rgh in roughness_df.iterrows():
            if pid not in exclude:
                on.get_pipe(self.nw, pid).roughness = rgh.values[0]

    def to_networkx(self) -> nx.Graph:
        """
        Returns undirected Graph with nodes and edges of network

        node attrs:
        - coordinates, node_type, is_virtual

        edge attrs:
        - name, link_type, is_virtual

        """
        _nodes = on.get_nodes(self.nw)
        _links = on.get_links(self.nw)

        # create nx nodes
        nodes = []
        coords = {}
        nkind = {}
        is_node_virtual = {}

        for node in _nodes:
            nodes.append(node.id)
            coords[node.id] = (node.xcoordinate, node.ycoordinate)
            nkind[node.id] = type(node).__name__
            is_node_virtual[node.id] = (
                True if node.id.startswith(VirtualReservoir._vr_prefix) else False
            )

        # create nx edges
        edges = []
        edge_names = {}
        ekind = {}
        is_link_virtual = {}

        for edge in _links:
            tpl = (edge.startnode.id, edge.endnode.id)
            edges.append(tpl)
            edge_names[tpl] = edge.id
            ekind[tpl] = type(edge).__name__
            is_link_virtual[tpl] = (
                True if edge.id.startswith(VirtualReservoir._vp_prefix) else False
            )

        # # create graph
        G = nx.Graph()

        # nodes
        G.add_nodes_from(nodes)
        nx.set_node_attributes(G, coords, "coordinates")
        nx.set_node_attributes(G, nkind, "node_type")
        nx.set_node_attributes(G, is_node_virtual, "is_virtual")

        # edges
        G.add_edges_from(edges)
        nx.set_edge_attributes(G, edge_names, "name")
        nx.set_edge_attributes(G, ekind, "link_type")
        nx.set_edge_attributes(G, is_link_virtual, "is_virtual")

        return G

    def plot_topology(self):
        # G = self.to_networkx()
        raise NotImplementedError()

    def get_sample_pipes(
        self,
        pctg: float,
        min_pipes: int,
        seed: int = 42,
        exclude_funcs: list[Callable] | None = None,
    ):
        """Return random pipe ids e.g. for localisation testing,
        if no specific exclude_funcs are provided, defaults to:
            1. no pipes that start with the vp prefix
            2. no pipes that contain "PUMP" (L-Town-specific probably)
        """

        if exclude_funcs is None:
            exclude_funcs = [
                lambda x: x.startswith((VirtualReservoir._vp_prefix)),
                lambda x: "PUMP" in x,
            ]

        pipes = [
            p
            for p in on.get_pipe_ids(self.nw)
            if not any(func(p) for func in exclude_funcs)
        ]

        n = max(min_pipes, int(pctg * len(pipes)))
        return np.random.default_rng(seed=seed).choice(pipes, n).tolist()

    def run_simulation(
        self,
        simulation_targets: SimulationTargets,
        solver_options: Optional[Dict[str, List[Any]]] = None,
        cleanup_sim_dir: bool = True,
        demand_perturbation_settings: (
            DynBaseDemandPertArgs | PatternPertArgs | None
        ) = None,
    ) -> Dict[SimulationTargets, pd.DataFrame]:
        """Run simulation using EpytSimulation class
        - retrieves most recent network inp
        - slices result if indices do not match
        """
        path = self.inp_path

        if not self._changes_saved:
            print(
                f"Warning! Recent changes not saved to inp-file, running simulation of latest saved state\n\t{path}."
            )

        simulator = Simulator(solver_options=solver_options)

        result = simulator.run_simulation(
            inp_path=path,
            simulation_targets=simulation_targets,
            cleanup=cleanup_sim_dir,
            demand_perturbation_settings=demand_perturbation_settings,
        )

        if len(result) != self.max_pattern_steps:
            pidx = self.pattern_index
            for k, v in result.items():
                r = result[k]

                # workaround if duration is shortened in simulator
                if (solver_options is None) or (
                    not "setTimeSimulationDuration" in solver_options.keys()
                ):
                    r = v.loc[pidx]
                else:
                    r = v.loc[
                        : pd.Timedelta(
                            seconds=solver_options["setTimeSimulationDuration"][0]
                        )
                    ]

                result[k] = r

        return result

    def run_localisation(
        self,
        leak_flow: pd.Series,
        temporal_resolution: Optional[str | pd.DateOffset] = None,
        pipe_list: Optional[list[str]] = None,
    ) -> dict[str, pd.Series | pd.DataFrame]: ...
