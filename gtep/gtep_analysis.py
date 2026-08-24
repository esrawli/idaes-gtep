#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################


import os
import json

import pandas as pd


class ExpansionPlanningAnalysis:
    """Analyze input data for an expansion planning case.

    This class reads the standard GTEP input CSV files and reports
    basic system counts, such as the number of buses, branches,
    generators, storage units, and loads.
    """

    def __init__(self, data_path):
        """Initialize the analysis object.

        :param data_path: Directory containing GTEP input data files.
        """
        self.data_path = data_path

    def _get_file_path(self, filename):
        """Return the full path for a file in the data directory."""
        return os.path.join(self.data_path, filename)

    def count_buses(self):
        """Read bus.csv and return the total number of unique buses."""
        bus_file = self._get_file_path("bus.csv")

        if not os.path.exists(bus_file):
            raise FileNotFoundError(f"Could not find bus.csv at: {bus_file}")

        bus_df = pd.read_csv(bus_file)
        return bus_df["Bus ID"].nunique()

    def count_storage(self):
        """Read storage.csv and return the total number of storage units.

        If storage.csv is not available, return 0.
        """
        storage_file = self._get_file_path("storage.csv")

        if not os.path.exists(storage_file):
            return 0

        storage_df = pd.read_csv(storage_file)
        return storage_df["name"].nunique()

    def count_branches(self):
        """Read branch.csv and return the total number of unique branches."""
        branch_file = self._get_file_path("branch.csv")

        if not os.path.exists(branch_file):
            raise FileNotFoundError(f"Could not find branch.csv at: {branch_file}")

        branch_df = pd.read_csv(branch_file)
        return branch_df["UID"].nunique()

    def count_generators(self):
        """Read gen.csv and return the total number of unique generators."""
        gen_file = self._get_file_path("gen.csv")

        if not os.path.exists(gen_file):
            raise FileNotFoundError(f"Could not find gen.csv at: {gen_file}")

        gen_df = pd.read_csv(gen_file)
        return gen_df["GEN UID"].nunique()

    def count_loads(self):
        """Read DAY_AHEAD_load.csv and return the number of load columns.

        The first four columns are assumed to be Year, Month, Day, and
        Period. All remaining columns are treated as load IDs.
        """
        load_file = self._get_file_path("DAY_AHEAD_load.csv")

        if not os.path.exists(load_file):
            raise FileNotFoundError(
                f"Could not find DAY_AHEAD_load.csv at: {load_file}"
            )

        load_df = pd.read_csv(load_file, nrows=1)

        time_columns = {"Year", "Month", "Day", "Period"}
        load_columns = [col for col in load_df.columns if col not in time_columns]

        return len(load_columns)

    def count_load_days(self):
        """Read DAY_AHEAD_load.csv and return the number of unique days."""
        load_file = self._get_file_path("DAY_AHEAD_load.csv")

        if not os.path.exists(load_file):
            raise FileNotFoundError(
                f"Could not find DAY_AHEAD_load.csv at: {load_file}"
            )

        load_df = pd.read_csv(load_file, usecols=["Year", "Month", "Day"])

        return load_df[["Year", "Month", "Day"]].drop_duplicates().shape[0]

    def count_load_hours(self):
        """Read DAY_AHEAD_load.csv and return the number of time periods/hours."""
        load_file = self._get_file_path("DAY_AHEAD_load.csv")

        if not os.path.exists(load_file):
            raise FileNotFoundError(
                f"Could not find DAY_AHEAD_load.csv at: {load_file}"
            )

        load_df = pd.read_csv(load_file, usecols=["Year", "Month", "Day", "Period"])

        return load_df[["Year", "Month", "Day", "Period"]].drop_duplicates().shape[0]
    
    def count_system_assets(self):
        """Return counts for buses, branches, generators, storage, and loads."""
        return {
            "buses": self.count_buses(),
            "branches": self.count_branches(),
            "generators": self.count_generators(),
            "storage": self.count_storage(),
            "loads": self.count_loads(),
            "load_days": self.count_load_days(),
            "load_hours": self.count_load_hours(),
        }

    def calculate_total_load_shed(self, results_path):
        """Read load_shed.json and return total load shed.

        The JSON keys are expected to look like:
        investmentStage[1].representativePeriod[1].commitmentPeriod[1].dispatchPeriod[1].loadShed.<load_name>

        All load-shed values are summed.
        """
        load_shed_file = os.path.join(results_path, "load_shed.json")

        if not os.path.exists(load_shed_file):
            raise FileNotFoundError(
                f"Could not find load_shed.json at: {load_shed_file}"
            )

        with open(load_shed_file, "r") as f:
            load_shed_data = json.load(f)

        return sum(float(value) for value in load_shed_data.values())


    def calculate_total_generation_gwh(self, results_path, hours_per_period=1.0):
        """Read generation.json and return total generation in GWh.

        The JSON values are assumed to be MW for each time period.
        The total is converted using:

            GWh = sum(MW) * hours_per_period / 1000

        :param results_path: Directory containing generation.json.
        :param hours_per_period: Duration of each period in hours.
            Defaults to 1.0.
        :return: Total generation in GWh.
        """
        generation_file = os.path.join(results_path, "generation.json")

        if not os.path.exists(generation_file):
            raise FileNotFoundError(
                f"Could not find generation.json at: {generation_file}"
            )

        with open(generation_file, "r") as f:
            generation_data = json.load(f)

        total_mw = sum(float(value) for value in generation_data.values())
        total_gwh = total_mw * hours_per_period / 1000

        return total_gwh


    def calculate_total_generation_gw(self, results_path):
        """Read generation.json and return total generation in GW."""
        generation_file = os.path.join(results_path, "generation.json")

        if not os.path.exists(generation_file):
            raise FileNotFoundError(
                f"Could not find generation.json at: {generation_file}"
            )

        with open(generation_file, "r") as f:
            generation_data = json.load(f)

        total_mw = sum(float(value) for value in generation_data.values())
        total_gw = total_mw / 1000

        return total_gw

    def calculate_total_load_gw(self, results_path):
        """Read loads.json and return total load in GW.

        The JSON keys are expected to include load parameters such as:
        investmentStage[1].representativePeriod[1].commitmentPeriod[1].loads.<load_name>

        The JSON values are assumed to be MW. The total is converted
        using:

            GW = sum(MW) / 1000

        :param results_path: Directory containing loads.json.
        :return: Total load in GW.
        """
        loads_file = os.path.join(results_path, "loads.json")

        if not os.path.exists(loads_file):
            raise FileNotFoundError(f"Could not find loads.json at: {loads_file}")

        with open(loads_file, "r") as f:
            loads_data = json.load(f)

        total_mw = sum(float(value) for value in loads_data.values())
        total_gw = total_mw / 1000

        return total_gw
    
    
    def print_system_asset_counts(self):
        """Print counts for buses, branches, generators, storage, and loads."""
        asset_counts = self.count_system_assets()

        for asset_type, count in asset_counts.items():
            print(f"{asset_type}: {count}")


if __name__ == "__main__":
    data_path = "./data/base_case_pcm_2030"
    results_path = "./this_is_too_big"
    
    analysis = ExpansionPlanningAnalysis(data_path)
    analysis.print_system_asset_counts()
    total_load_shed = analysis.calculate_total_load_shed(results_path)
    total_generation_gwh = analysis.calculate_total_generation_gwh(results_path, hours_per_period=1)
    total_generation_gw = analysis.calculate_total_generation_gw(results_path)
    total_load_gw = analysis.calculate_total_load_gw(results_path)
    
    print(f"Total load (GW): {total_load_gw}")
    print(f"Total load shed: {total_load_shed}")
    print(f"Total generation (GWh): {total_generation_gwh}")
    
