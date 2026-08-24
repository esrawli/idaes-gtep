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


import json
import os

import pandas as pd


class ExpansionPlanningAnalysis:
    """Analyze input data and saved results for an expansion planning case.

    This class reads standard GTEP input CSV files and result JSON files
    to report basic system counts and aggregate solution metrics.
    """

    def __init__(self, data_path, print_results=True, save_csv=False, csv_path=None):
        """Initialize the analysis object.

        :param data_path: Directory containing GTEP input data files.
        :param print_results: If True, print calculated metrics.
            Defaults to True.
        :param save_csv: If True, save calculated metrics to a CSV file.
            Defaults to False.
        :param csv_path: Optional path to the output metrics CSV file.
        """
        self.data_path = data_path
        self.print_results = print_results
        self.save_csv = save_csv
        self.csv_path = csv_path
        self.metrics_rows = []

    def _get_file_path(self, filename):
        """Return the full path for a file in the data directory."""
        return os.path.join(self.data_path, filename)

    def _record_metric(self, metric, value, units="", category=""):
        """Record, optionally print, and later save a metric.

        :param metric: Metric name.
        :param value: Metric value.
        :param units: Metric units.
        :param category: Metric category.
        """
        row = {
            "metric": metric,
            "category": category,
            "value": value,
            "units": units,
        }

        self.metrics_rows.append(row)

        if self.print_results:
            unit_text = f" {units}" if units else ""
            print(f"{metric}: {value}{unit_text}")

    def save_metrics_csv(self, csv_path=None):
        """Save recorded metrics to a CSV file.

        :param csv_path: Optional path to the output CSV file. If not
            provided, uses ``self.csv_path``.
        :return: DataFrame containing recorded metrics.
        """
        output_path = csv_path or self.csv_path

        if output_path is None:
            raise ValueError(
                "csv_path must be provided to save metrics to a CSV file."
            )

        metrics_df = pd.DataFrame(self.metrics_rows)

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        metrics_df.to_csv(output_path, index=False)

        if self.print_results:
            print(f"\nSaved analysis metrics to: {output_path}")

        return metrics_df

    def finalize(self, csv_path=None):
        """Save metrics to CSV if ``save_csv`` is enabled.

        :param csv_path: Optional path to the output CSV file.
        :return: DataFrame if metrics are saved, otherwise None.
        """
        if self.save_csv:
            return self.save_metrics_csv(csv_path)

        return None

    def count_buses(self):
        """Read bus.csv and return the total number of unique buses."""
        bus_file = self._get_file_path("bus.csv")

        if not os.path.exists(bus_file):
            raise FileNotFoundError(f"Could not find bus.csv at: {bus_file}")

        bus_df = pd.read_csv(bus_file)
        count = bus_df["Bus ID"].nunique()

        self._record_metric(
            metric="buses",
            value=count,
            units="count",
            category="system_count",
        )

        return count

    def count_storage(self):
        """Read storage.csv and return the total number of storage units.

        If storage.csv is not available, return 0.
        """
        storage_file = self._get_file_path("storage.csv")

        if not os.path.exists(storage_file):
            count = 0
        else:
            storage_df = pd.read_csv(storage_file)
            count = storage_df["name"].nunique()

        self._record_metric(
            metric="storage",
            value=count,
            units="count",
            category="system_count",
        )

        return count

    def count_branches(self):
        """Read branch.csv and return the total number of unique branches."""
        branch_file = self._get_file_path("branch.csv")

        if not os.path.exists(branch_file):
            raise FileNotFoundError(f"Could not find branch.csv at: {branch_file}")

        branch_df = pd.read_csv(branch_file)
        count = branch_df["UID"].nunique()

        self._record_metric(
            metric="branches",
            value=count,
            units="count",
            category="system_count",
        )

        return count

    def count_generators(self):
        """Read gen.csv and return the total number of unique generators."""
        gen_file = self._get_file_path("gen.csv")

        if not os.path.exists(gen_file):
            raise FileNotFoundError(f"Could not find gen.csv at: {gen_file}")

        gen_df = pd.read_csv(gen_file)
        count = gen_df["GEN UID"].nunique()

        self._record_metric(
            metric="generators",
            value=count,
            units="count",
            category="system_count",
        )

        return count

    def count_loads(self):
        """Read DAY_AHEAD_load.csv and return the number of load columns.

        The first four columns are assumed to be Year, Month, Day, and
        Period. All remaining columns are treated as load IDs.
        """
        load_file = self._get_file_path("DAY_AHEAD_load.csv")

        if not os.path.exists(load_file):
            raise FileNotFoundError(f"Could not find DAY_AHEAD_load.csv at: {load_file}")

        load_df = pd.read_csv(load_file, nrows=1)

        time_columns = {"Year", "Month", "Day", "Period"}
        load_columns = [col for col in load_df.columns if col not in time_columns]
        count = len(load_columns)

        self._record_metric(
            metric="loads",
            value=count,
            units="count",
            category="system_count",
        )

        return count

    def count_load_days(self):
        """Read DAY_AHEAD_load.csv and return the number of unique load days."""
        load_file = self._get_file_path("DAY_AHEAD_load.csv")

        if not os.path.exists(load_file):
            raise FileNotFoundError(f"Could not find DAY_AHEAD_load.csv at: {load_file}")

        load_df = pd.read_csv(load_file, usecols=["Year", "Month", "Day"])
        count = load_df[["Year", "Month", "Day"]].drop_duplicates().shape[0]

        self._record_metric(
            metric="load_days",
            value=count,
            units="count",
            category="system_count",
        )

        return count

    def count_load_hours(self):
        """Read DAY_AHEAD_load.csv and return the number of load time periods.

        If Period is hourly, this is the number of load hours.
        """
        load_file = self._get_file_path("DAY_AHEAD_load.csv")

        if not os.path.exists(load_file):
            raise FileNotFoundError(f"Could not find DAY_AHEAD_load.csv at: {load_file}")

        load_df = pd.read_csv(load_file, usecols=["Year", "Month", "Day", "Period"])
        count = (
            load_df[["Year", "Month", "Day", "Period"]]
            .drop_duplicates()
            .shape[0]
        )

        self._record_metric(
            metric="load_time_periods",
            value=count,
            units="count",
            category="system_count",
        )

        return count

    def count_system_assets(self):
        """Return counts for buses, branches, generators, storage, and loads."""
        return {
            "buses": self.count_buses(),
            "branches": self.count_branches(),
            "generators": self.count_generators(),
            "storage": self.count_storage(),
            "loads": self.count_loads(),
            "load_days": self.count_load_days(),
            "load_time_periods": self.count_load_hours(),
        }

    def calculate_total_load_shed(self, results_path):
        """Read load_shed.json and return total load shed.

        The JSON keys are expected to include load-shed variables such as:
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

        total_load_shed = sum(float(value) for value in load_shed_data.values())

        self._record_metric(
            metric="total_load_shed",
            value=total_load_shed,
            units="MW",
            category="reliability",
        )

        return total_load_shed

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

        self._record_metric(
            metric="total_generation",
            value=total_gwh,
            units="GWh",
            category="system_energy",
        )

        return total_gwh

    def calculate_total_generation_gw(self, results_path):
        """Read generation.json and return total generation in GW.

        The JSON values are assumed to be MW. The total is converted
        using:

            GW = sum(MW) / 1000
        """
        generation_file = os.path.join(results_path, "generation.json")

        if not os.path.exists(generation_file):
            raise FileNotFoundError(
                f"Could not find generation.json at: {generation_file}"
            )

        with open(generation_file, "r") as f:
            generation_data = json.load(f)

        total_mw = sum(float(value) for value in generation_data.values())
        total_gw = total_mw / 1000

        self._record_metric(
            metric="total_generation",
            value=total_gw,
            units="GW",
            category="system_power",
        )

        return total_gw

    def calculate_total_load_gw(self, results_path):
        """Read loads.json and return total load in GW.

        The JSON values are assumed to be MW. The total is converted
        to GW using:

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

        self._record_metric(
            metric="total_load",
            value=total_gw,
            units="GW",
            category="system_power",
        )

        return total_gw

    def calculate_generation_by_unit_type_gwh(self, results_path, hours_per_period=1.0):
        """Read generation.json and return total generation by unit type in GWh.

        Generator names are extracted from generation.json keys and
        mapped to unit types using gen.csv. The JSON values are assumed
        to be MW for each time period.

        GWh = sum(MW) * hours_per_period / 1000

        :param results_path: Directory containing generation.json.
        :param hours_per_period: Duration of each period in hours.
            Defaults to 1.0.
        :return: Dictionary with total generation by unit type in GWh.
        """
        gen_file = self._get_file_path("gen.csv")
        generation_file = os.path.join(results_path, "generation.json")

        if not os.path.exists(gen_file):
            raise FileNotFoundError(f"Could not find gen.csv at: {gen_file}")

        if not os.path.exists(generation_file):
            raise FileNotFoundError(
                f"Could not find generation.json at: {generation_file}"
            )

        gen_df = pd.read_csv(gen_file)

        gen_uid_col = "GEN UID" if "GEN UID" in gen_df.columns else "genUID"
        unit_type_col = "Unit Type" if "Unit Type" in gen_df.columns else "unit_type"

        gen_uid_to_type = {
            str(row[gen_uid_col]): str(row[unit_type_col]).upper()
            for _, row in gen_df.iterrows()
        }

        with open(generation_file, "r") as f:
            generation_data = json.load(f)

        generation_by_type_mw = {}

        for key, value in generation_data.items():
            gen_name = key.rsplit(".", 1)[-1]
            unit_type = gen_uid_to_type.get(gen_name, "UNKNOWN")

            generation_by_type_mw[unit_type] = (
                generation_by_type_mw.get(unit_type, 0.0) + float(value)
            )

        generation_by_type_gwh = {
            unit_type: total_mw * hours_per_period / 1000
            for unit_type, total_mw in generation_by_type_mw.items()
        }

        for unit_type, total_gwh in sorted(generation_by_type_gwh.items()):
            self._record_metric(
                metric=f"generation_{unit_type}",
                value=total_gwh,
                units="GWh",
                category="generation_by_unit_type",
            )

        return generation_by_type_gwh

    def run_all_metrics(self, results_path, hours_per_period=1.0, csv_path=None):
        """Calculate all supported metrics and optionally save them.

        :param results_path: Directory containing saved result JSON files.
        :param hours_per_period: Duration of each time period in hours.
            Defaults to 1.0.
        :param csv_path: Optional output CSV path. If provided and
            ``save_csv`` is enabled, metrics are saved to this path.
        :return: Dictionary of selected aggregate metric outputs.
        """
        outputs = {}

        outputs["system_assets"] = self.count_system_assets()
        outputs["total_generation_gwh"] = self.calculate_total_generation_gwh(
            results_path,
            hours_per_period=hours_per_period,
        )
        outputs["total_load_gw"] = self.calculate_total_load_gw(results_path)
        outputs["total_load_shed_mw"] = self.calculate_total_load_shed(results_path)
        outputs["generation_by_unit_type_gwh"] = (
            self.calculate_generation_by_unit_type_gwh(
                results_path,
                hours_per_period=hours_per_period,
            )
        )

        self.finalize(csv_path)

        return outputs
