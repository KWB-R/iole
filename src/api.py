"""Contains functions to create HydraulicModel/DualModel instances and perform simulations"""

from typing import Never, TYPE_CHECKING, Literal
from dataclasses import dataclass, field
from pathlib import Path
from functools import cached_property

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    import oopnet as on

if __name__ != "__main__":
    from .model.network_container import _DualModel, HydraulicNetwork, VirtualReservoir
    from .model.configuration import SUBSTITUTE_INFLOW_PATTERN_SUFFIX
else:
    from model.network_container import _DualModel, HydraulicNetwork, VirtualReservoir
    from model.configuration import SUBSTITUTE_INFLOW_PATTERN_SUFFIX

MeasurementType = Literal["head", "flow", "pump_flow"]


class DualModelOptions:

    base_pattern_start_day: int = (
        0  # 0=Monday, 6=Sunday; patterns have to start at 00:00
    )
    base_pattern_length: pd.Timedelta = pd.Timedelta(days=7)
    cyclic_base_patterns: list[str] = ["P-Residential", "P-Commercial"]
    cyclic_pattern_wrapping: bool = True

    # Options
    # if True, model can use leak-free residual flows at virtual reservoirs
    use_virtual_flow_correction_patterns: bool = False

    # if True replaces Reservoirs and Tanks in the DualModel with negative demand nodes for fixed boundary conditions
    use_inflow_replacement: bool = False
    use_pump_replacement: bool = True

    skip_data_validation: bool = False

    correction_pattern_start_day: int = (
        0  # 0=Monday, 6=Sunday; patterns have to start at 00:00
    )


@dataclass
class DualModel:
    """Class to access the DualModel simulation

    Args:
        base_inp: path to inp-file to be used as a basis
        pressure_sensor_node_ids: nodes where pressure sensors is available
        pump_ids: pumps in the system
        inflow_mapping: dictionary of <tank or reservoir id>:<connected pipe> to insert surrogate negative demands

    Post init args:
        nw: _DualModel(HydraulicNetwork) instance that wraps around on.Network
        correction_flows: cyclic pattern of virtual pipe residual flwos in leak free period (currently: 7d only)
    """

    base_inp: Path
    pressure_sensor_node_ids: list[str]
    pump_ids: list[str] = field(default_factory=list)
    inflow_mapping: dict[str, str] = field(default_factory=dict)

    # post_init
    nw: _DualModel | None = field(default=None, init=False)

    # optional/dynamically updated
    correction_flows: pd.DataFrame | None = field(default=None, init=False)

    def __post_init__(self):
        self._build_dual_model()

    @cached_property
    def _reverse_inflow_mapping(self):
        return dict(zip(self.inflow_mapping.values(), self.inflow_mapping.keys()))

    def _build_dual_model(self):
        hm = HydraulicNetwork(
            source_path=self.base_inp, inflow_pipes=list(self.inflow_mapping.values())
        )

        hm.check_pattern_compatibility(DualModelOptions.base_pattern_length)

        if self.pump_ids and DualModelOptions.use_pump_replacement:
            for pump in self.pump_ids:
                hm.split_area_at_pump(pump_id=pump, substitute_demand_pattern=pump)

        if self.inflow_mapping and DualModelOptions.use_inflow_replacement:
            for k, v in self.inflow_mapping.items():
                hm.replace_reservoir_or_tank_with_emitter_node(
                    reservoir_or_tank_id=k,
                    demand_pattern=v,
                    pipes_to_reconnect=v,
                )

        self.nw = hm.to_dual_model(self.pressure_sensor_node_ids)

    def set_correction_flows(self, correction_flows: pd.DataFrame) -> Never | None:
        """This would be a 7-day DataFrame that contains
        a column for every pressure sensor with a pattern built
        from the virtual flows in a leak-free period.

        As this is extracted from earlier simulated virtual flows
        the data has a timestamp that is used to create a pattern
        that fits the cyclic base pattern start of the original
        network file.

        Names of columns have to be vp_<node_id>.
        (prefix is specified in network_container.VirtualReservoir._vr_prefix)
        """

        if not DualModelOptions.use_virtual_flow_correction_patterns:
            raise ValueError("Correction will only be used when option is toggled.")

        assert isinstance(
            correction_flows.index, pd.DatetimeIndex
        ), "Index needs to be Datetime."

        assert (
            correction_flows.columns.str.contains(
                vr.replace(VirtualReservoir.vp_prefix, "")
            ).any()
            for vr in self.nw.virtual_reservoirs
        )

        self.correction_flows = correction_flows
        # sanitisize names
        self.correction_flows.columns = [
            f"{c.replace(VirtualReservoir.vp_prefix, VirtualReservoir.vr_prefix)}{VirtualReservoir.flow_corr_pattern_suffix}"
            for c in self.correction_flows.columns
        ]

    def run_simulation(
        self,
        heads: pd.DataFrame,
        inflows: pd.DataFrame | None = None,
        pump_flows: pd.DataFrame | None = None,
        aggregate: bool = True,
    ) -> pd.DataFrame | pd.Series | Never:

        _heads = heads.copy()

        if DualModelOptions.use_inflow_replacement:
            if inflows is None:
                raise ValueError("No inflows provided.")

        # data validation
        _heads.columns = [
            f"{VirtualReservoir.vr_prefix}{c}" for c in _heads.columns
        ]  # e.g. n123 -> vr_n123 (head pattern is named like the VR it's assigned to)
        self._validate_data("head", _heads, heads.index)
        patterns = [_heads]

        if pump_flows is not None:
            _pump_flows = pump_flows.copy()
            self._validate_data("pump_flow", _pump_flows, _heads.index)
            patterns.append(_pump_flows)

        if inflows is not None:
            _inflows = inflows.copy()
            self._validate_data("flow", _inflows, _heads.index)
            # fix name to fit patterns
            _inflows.columns = [
                f"{self._reverse_inflow_mapping[c]}{SUBSTITUTE_INFLOW_PATTERN_SUFFIX}"
                for c in _inflows.columns
            ]
            patterns.append(_inflows)

        # write correction patterns
        ## TODO: make sure these are correctly aligned, compare base pattern wrapping
        if (
            self.correction_flows is not None
            and DualModelOptions.use_virtual_flow_correction_patterns
        ):
            self.nw.set_patterns(self.correction_flows, overwrite=True)
            if DualModelOptions.cyclic_pattern_wrapping:
                self.nw.wrap_correction_patterns(
                    start_timestamp=_heads.first_valid_index(),
                    start_dow=DualModelOptions.correction_pattern_start_day,
                )

        # concat, assign
        data_patterns = pd.concat(patterns, axis=1)
        self.nw.set_patterns(data_patterns, overwrite=True)

        # make native network patterns start at the same dayofweek and time as the data
        if DualModelOptions.cyclic_pattern_wrapping:
            self.nw.wrap_base_patterns(
                pattern_ids=DualModelOptions.cyclic_base_patterns,
                start_dow=DualModelOptions.base_pattern_start_day,
                start_timestamp=_heads.first_valid_index(),
            )

        # simulation duration/length of data
        simulation_duration_seconds = (
            _heads.last_valid_index() - _heads.first_valid_index()
        ).total_seconds()

        # run simulation
        vflow = self.nw.run_simulation(
            simulation_targets={"flow": self.nw.virtual_pipes},
            solver_options={"setTimeSimulationDuration": [simulation_duration_seconds]},
        )["flow"]

        # reindex to original timestamps
        vflow.index = _heads.index

        if aggregate:
            return vflow.sum(axis=1)
        else:
            return vflow

    def run_localisation(self): ...

    def _validate_data(
        self, kind: MeasurementType, data: pd.DataFrame, target_index: pd.Index
    ):
        if DualModelOptions.skip_data_validation:
            return

        # nan check
        if data.isna().values.any():
            raise ValueError(f"NANs in {kind} data detected.")

        # indices align?
        if not data.index.equals(target_index):
            raise ValueError(f"Index mismatch: {kind}.")

        # get names
        data_names = set(data.columns.to_list())

        match kind:
            case "flow":
                network_names = set(self.nw.inflow_pipes)

            case "head":
                network_names = set(self.nw.virtual_reservoirs)

            case "pump_flow":
                network_names = set(self.nw.pump_demands)

        # any missing?
        if bool(_diff := (network_names - data_names)):
            raise ValueError(f"No patterns for {kind}: {*_diff,} in provided data.")


if __name__ == "__main__":
    ...
