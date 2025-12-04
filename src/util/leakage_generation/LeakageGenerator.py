__doc__ = ""

import os
from typing import Optional
import pandas as pd
import numpy as np

class LeakageGenerator:

    def __init__(
        self,
        simulation_time_stamp: pd.DatetimeIndex,
        leak_start_time: str = "2018-01-08 00:00",
        leak_peak_time: str = "2018-01-22 23:55",
        leak_end_time: str = "2018-01-31 23:55",
        leak_diameter: float = 0.01,
        leak_name: Optional[str] = None,
    ):

        self.leak_start_time = leak_start_time
        self.leak_peak_time = leak_peak_time
        self.leak_end_time = leak_end_time
        self.leak_diameter = leak_diameter
        self.leak_name = leak_name
        self.simulation_time_stamp = simulation_time_stamp

    @property  # position of the leak start in the simulation time stamp.
    def ST(self):
        return self.simulation_time_stamp.get_indexer(
            [self.leak_start_time], method="nearest"
        )[0]

    @property  # position of the leak peak in the simulation time stamp.
    def PT(self):
        return (
            self.simulation_time_stamp.get_indexer(
                [self.leak_peak_time], method="nearest"
            )[0]
            + 1
        )

    @property  # position of the leak end in the simulation time stamp.
    def ET(self):
        try:
            return (
                self.simulation_time_stamp.get_indexer(
                    [self.leak_end_time], method="nearest"
                )[0]
                + 1
            )
        except:
            return len(self.simulation_time_stamp)

    def calculate_array(self):

        leak_area = np.pi * ((self.leak_diameter / 2) ** 2)

        increment_leak_diameter = np.arange(
            self.leak_diameter / (self.PT - self.ST),
            self.leak_diameter,
            self.leak_diameter / (self.PT - self.ST),
        )

        increment_leak_area = (
            0.75
            * ((2 / 1000) ** 0.5)
            * 990.27
            * np.pi
            * ((increment_leak_diameter / 2) ** 2)
        )

        leak_magnitude = 0.75 * ((2 / 1000) ** 0.5) * 990.27 * leak_area

        pattern_array = (
            [0] * (self.ST)
            + increment_leak_area.tolist()
            + [leak_magnitude] * (self.ET - self.PT + 1)
            + [0] * (len(self.simulation_time_stamp) - self.ET)
        )

        # Transform the resulting pattern_array to pandas data frame.
        pattern_array = pd.DataFrame(
            pattern_array[: len(self.simulation_time_stamp)],
            columns=[f"{self.leak_name}"],
            index=self.simulation_time_stamp,
        )

        self.pattern_array = pattern_array

    def get_array(self):
        if not hasattr(self, "pattern_array"):
            self.calculate_array()
        return self.pattern_array

    def save_parquet(self, folder_path) -> os.PathLike:
        if not hasattr(self, "pattern_array"):
            self.calculate_array()

        path = os.path.join(
            folder_path, f"{self.leak_name if self.leak_name is not None else "unnamed"}_leak_pattern_array.parquet"
        )

        self.pattern_array.to_parquet(path, index=True)

        return path


class LeakageGenerator2:

    def __init__(self):
        ...

    def get_timedelta_series(
            self,
            leak_name: str,
            leak_magnitude: float,
            leak_duration_total: pd.Timedelta,
            timestep: pd.Timedelta = pd.Timedelta(minutes=5),
            leak_growth_duration: pd.Timedelta = pd.Timedelta(minutes=0),
            leak_start_offset: Optional[pd.Timedelta] = pd.Timedelta(days=0),
            ):

        assert leak_growth_duration <= leak_duration_total, "Leak growth cannot be longer than total leak duration."

        timestep_seconds = timestep.total_seconds()

        leak_idx = pd.to_timedelta(np.arange(
            0,
            leak_duration_total.total_seconds(),
            timestep_seconds),
            unit="s")

        growth_end_idx = len(leak_idx[:int(leak_growth_duration.total_seconds() / timestep_seconds)])

        increment_leak_diameter = np.concat(
            [np.linspace(0, 1, growth_end_idx),
                np.repeat(1, len(leak_idx) - growth_end_idx)],
            axis=0)

        increment_leak_area = (
            0.75
            * ((2 / 1000) ** 0.5)
            * 990.27
            * np.pi
            * ((increment_leak_diameter / 2) ** 2)
        )

        # make series and scale
        pattern = pd.Series(increment_leak_area / np.max(increment_leak_area) * leak_magnitude,
                            index=leak_idx)

        pattern.index = pattern.index + leak_start_offset
        pattern.index.freq = timestep

        if leak_name is not None:
            pattern.name = leak_name
        else:
            pattern.name = "leak"

        return pattern



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lg = LeakageGenerator2()

    leak1_td = lg.get_timedelta_series(
        leak_magnitude=1.8,
        leak_duration_total=pd.Timedelta(days=25),
        leak_growth_duration=pd.Timedelta(days=12),
        timestep=pd.Timedelta(minutes=5),
        leak_start_offset= pd.Timedelta(days=5),
        leak_name="p426"
    )

    leak2_td = lg.get_timedelta_series(
        leak_magnitude=2.5,
        leak_duration_total=pd.Timedelta(days=14),
        timestep=pd.Timedelta(minutes=5),
        leak_start_offset= pd.Timedelta(days=10),
        leak_name="p426"
    )


    leak1_td.plot()
    leak2_td.plot()

    plt.show()

