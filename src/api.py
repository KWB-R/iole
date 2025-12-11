"""Contains functions to create HydraulicModel/DualModel instances and perform simulations"""

from typing import Never
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from inspect import ismethod

import pandas as pd
import numpy as np

if __name__ != "__main__":
    from .model.network_container import _DualModel, HydraulicNetwork, VirtualReservoir
else:
    from model.network_container import _DualModel, HydraulicNetwork, VirtualReservoir


class DualModelOptions:

    # Base pattern
    # 0=Monday, 6=Sunday; patterns have to start at 00:00
    base_pattern_start_day: int = 0
    base_pattern_length: pd.Timedelta = pd.Timedelta(days=7)
    cyclic_base_patterns: list[str] = ["P-Residential", "P-Commercial"]
    wrap_patterns: bool = True

    # Options
    # if True, model can use leak-free residual flows at virtual reservoirs
    use_virtual_flow_correction_patterns: bool = False

    # if True replaces Reservoirs and Tanks in the DualModel with negative demand nodes for fixed boundary conditions
    use_inflow_replacement: bool = False

    use_pump_replacement: bool = True


@dataclass
class DualModel:
    """Class to access the DualModel simulation

    Args:
        base_inp: path to inp-file to be used as a basis
        pressure_sensor_node_ids: nodes where pressure sensors is available
        pump_ids: pumps in tee system

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
                    demand_pattern_series=v,
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
                vr.replace(VirtualReservoir._vp_prefix, "")
            ).any()
            for vr in self.nw.virtual_reservoirs
        )

        self.correction_flows = correction_flows
        # sanitisize names
        self.correction_flows.columns = [
            f"{c.replace(VirtualReservoir._vp_prefix, VirtualReservoir._vr_prefix)}{VirtualReservoir._flow_corr_pattern_suffix}"
            for c in self.correction_flows.columns
        ]

    def run_simulation(
        self,
        heads: pd.DataFrame,
        inflows: pd.DataFrame | None = None,
        pump_flows: pd.DataFrame | None = None,
        aggregate: bool = True,
    ) -> pd.DataFrame | pd.Series | Never:

        if DualModelOptions.use_inflow_replacement:
            if inflows is None:
                raise ValueError("No inflows provided.")

        head_start_index = heads.index[0]
        for df in [inflows, pump_flows]:
            if df is not None:
                assert (
                    df.index[0] == head_start_index
                ), "Start indices of provided data do not match."

        # all VRs?
        assert all(
            vr_id.replace("vr_", "") in heads.columns
            for vr_id in self.nw.virtual_reservoirs
        ), f"Missing VR headpatterns in head dataframe: {*[vr_id for vr_id in self.nw.virtual_reservoirs if vr_id not in heads.columns],}"

        patterns = [heads]

        # all surrogate pump flows?
        if pump_flows is not None:
            assert all(
                pump_id in pump_flows.columns for pump_id in self.nw.pump_demands
            ), "Missing pump demand surrogates."
            patterns.append(pump_flows)

        # all inflow pipes?
        if inflows is not None:
            assert all(p_id in inflows.columns for p_id in self.nw.inflow_pipes)
            patterns.append(inflows)

        # write correction patterns
        ## TODO: make sure these are correctly aligned, compare base pattern wrapping
        if (
            self.correction_flows is not None
            and DualModelOptions.use_virtual_flow_correction_patterns
        ):
            self.nw.set_patterns(self.correction_flows, overwrite=True)

        data_patterns = pd.concat(patterns, axis=1)

        assert data_patterns.notna().values.all(), "NAN values in data detected."

        self.nw.set_patterns(data_patterns, overwrite=True)

        if DualModelOptions.wrap_patterns:
            self.nw.wrap_base_patterns(
                pattern_ids=DualModelOptions.cyclic_base_patterns,
                start_dow=DualModelOptions.base_pattern_start_day,
                start_timestamp=heads.first_valid_index(),
            )

        simulation_duration_seconds = (
            heads.last_valid_index() - heads.first_valid_index()
        ).total_seconds()

        vflow = self.nw.run_simulation(
            simulation_targets={"flow": self.nw.virtual_pipes},
            solver_options={"setTimeSimulationDuration": [simulation_duration_seconds]},
        )["flow"]

        vflow.index = heads.index

        if aggregate:
            return vflow.sum(axis=1)
        else:
            return vflow

    def run_localisation(self): ...


if __name__ == "__main__":
    ...
