from __future__ import annotations
import os
from abc import ABC
from dataclasses import dataclass, field

import pandas as pd

from .model import (
    HydraulicNetwork,
    Options,
)

# Simulation wrappers (dispatch workers for simulation iterations)
@dataclass
class Simulation(ABC):

    dual_model_inp: os.PathLike
    options: Options

    heads: pd.DataFrame
    inflows: pd.DataFrame
    pump_flows: pd.DataFrame

    corr_flows: pd.DataFrame = field(default=None)

    nw: HydraulicNetwork = field(default=None, init=False)

    dm_path: os.PathLike = field(default=None, init=False)

    def __post_init__(self):
        self.nw = HydraulicNetwork(self.dual_model_inp)

    def run(
        self,
    ):

        self._add_measurements()
        self._write()

        return self.nw.run_simulation(
            simulation_targets={"flow": vps},
        )["flow"]

    def _add_measurements(self):
        self.nw.set_patterns(patterns=self.heads)
        self.nw.set_patterns(self.corr_flows)

    def _write(self):
        self.nw.save(self.dm_path)


