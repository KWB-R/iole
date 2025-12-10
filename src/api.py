""" Contains functions to create HydraulicModel/DualModel instances and perform simulations """

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .model.network_container import _DualModel, HydraulicNetwork


class DualModelOptions:

    pattern_start_day: int = 0 #0=Monday, 6=Sunday


@dataclass
class DualModel:

    base_inp: Path
    pressure_sensor_node_ids: list[str]
    inflow_pipe_ids: list[str]
    pump_ids: list[str] = field(default_factory=list)

    # post_init
    _nw: _DualModel | None = field(default=None, init=False)

    def __post_init__(self):
        self._build_dual_model()

    def _build_dual_model(self):
        hm = HydraulicNetwork(self.base_inp)

        if self.pump_ids:
            for pump in self.pump_ids:
                hm.split_area_at_pump(pump_id=pump)

        self._nw = hm.to_dual_model(self.pressure_sensor_node_ids)

    def run_detection(
        self,
        heads: pd.DataFrame,
        inflows: pd.DataFrame,
        pump_flows: pd.DataFrame | None,
        aggregate: bool = True
        ) -> pd.DataFrame | pd.Series:
        ...

    def run_localisation(
            self
        ):
        ...



if __name__ == "__main__":
    # Usage
