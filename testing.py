import json
from functools import reduce
from pathlib import Path

import pandas as pd


from src.api import DualModel, DualModelOptions

with open(
    r"data\scenario_configurations\all_sensor_scenarios_0_weighted.json", "r"
) as file:
    psensors = reduce(lambda x, y: x + y, json.load(file)["33"].values(), [])

inp = Path(r"data\networks\L-TOWN.inp")

dm = DualModel(
    base_inp=inp,
    pressure_sensor_node_ids=psensors,
    pump_ids=["PUMP_1"],
    inflow_mapping={"T1": "p239", "R2": "p235", "R1": "p227"},
)

data = pd.read_parquet(r"data\dummy_data\example_data.parquet")
head_data = data[psensors]
inflows = data[["p239", "p235", "p227"]]
pump_flow = data["PUMP_1"].to_frame()


indexer = pd.IndexSlice["2019-02-04 03:00:00":"2019-02-05 16:00:00"]

vf = dm.run_simulation(
    heads=head_data[indexer],
    inflows=inflows[indexer],
    pump_flows=pump_flow[indexer],
)

dm.nw.save(r"data\output\temp\dm.inp")


# Usage
