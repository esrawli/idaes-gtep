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

# Generation and Transmission Expansion Planning
# IDAES project
# author: Kyle Skolfield, Thom R. Edwards
# date: 01/04/2024
# Model available at http://www.optimization-online.org/DB_FILE/2017/08/6162.pdf

import re
import os
import csv
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go

from collections import namedtuple, defaultdict
from pathlib import Path

import pyomo.environ as pyo
import pyomo.gdp as gdp
from pyomo.environ import units as u
from pyomo.core.base.param import IndexedParam
from pyomo.core.base.expression import ScalarExpression, IndexedExpression

from gtep.gtep_model import ExpansionPlanningModel

logger = logging.getLogger(__name__)


class ExpansionPlanningSolution:
    """This class stores the solution to the ExpansionPlanningModel
    class for writing and visualization.

    """

    def __init__(self, data_path):
        self.gen_df = pd.read_csv(f"{data_path}/gen.csv")
        self.storage_csv_path = os.path.join(data_path, "storage.csv")
        if os.path.exists(self.storage_csv_path):
            self.storage_df = pd.read_csv(self.storage_csv_path)

    def _get_generation_types(self):
        """This method returns generation type labels and colors used
        in plots and stackgraphs.

        """

        GenerationType = namedtuple("GenerationType", ["label", "color"])
        tab20 = plt.get_cmap("tab20")

        def darken_color(hex_color, percent=0.2):
            hex_color = hex_color.lstrip("#")
            rgb = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
            darker_rgb = [max(0, int(c * (1 - percent))) for c in rgb]
            return "#" + "".join(f"{c:02x}" for c in darker_rgb)

        gas_cc = mcolors.to_hex(tab20(1))
        gas_ct = mcolors.to_hex(tab20(3))
        coal = mcolors.to_hex(tab20(5))
        nuclear = mcolors.to_hex(tab20(2))
        solar = mcolors.to_hex(tab20(9))
        rt_solar = mcolors.to_hex(tab20(8))
        wind = mcolors.to_hex(tab20(11))
        thermal = mcolors.to_hex(tab20(13))
        steam = mcolors.to_hex(tab20(14))
        hydro = mcolors.to_hex(tab20(19))
        storage = mcolors.to_hex(tab20(15))
        other = mcolors.to_hex(tab20(0))

        return {
            # Unit-type names used by create_plots()
            "CC": GenerationType("Gas CC", gas_cc),
            "CT": GenerationType("Gas CT", gas_ct),
            "COAL": GenerationType("Coal", coal),
            "NUC": GenerationType("Nuclear", nuclear),
            "PV": GenerationType("Solar", solar),
            "RTPV": GenerationType("RT Solar", rt_solar),
            "WIND": GenerationType("Wind", wind),
            "THERMAL": GenerationType("Thermal", thermal),
            "THERM": GenerationType("Therm", thermal),
            "STEAM": GenerationType("Steam", steam),
            "HYDRO": GenerationType("Hydro", hydro),
            "BATTERY": GenerationType("Storage", storage),
            "PS": GenerationType("Pumped Storage", storage),
            "ST": GenerationType("Storage Turbine", storage),
            "OTHER": GenerationType("Other", other),
            "GEO": GenerationType("Geothermal", other),
            # Generator-name suffixes used by stackgraph/metrics
            "cc_gas": GenerationType("Gas CC", gas_cc),
            "ct_gas": GenerationType("Gas CT", gas_ct),
            "coal": GenerationType("Coal", coal),
            "nuclear": GenerationType("Nuclear", nuclear),
            "solar": GenerationType("Solar", solar),
            "wind": GenerationType("Wind", wind),
            "hydro": GenerationType("Hydro", hydro),
            "thermal_other": GenerationType("Thermal", thermal),
            "therm": GenerationType("Therm", thermal),
            "steam": GenerationType("Steam", steam),
            "battery_charge": GenerationType("Battery Charging", storage),
            "battery_discharge": GenerationType("Battery Discharging", storage),
            "ps": GenerationType("PS", storage),
            "other": GenerationType("Other", other),
            "geo": GenerationType("Geo", other),
            # Candidate suffixes
            "gas_cc-c": GenerationType("Gas CC Candidate", darken_color(gas_cc)),
            "gas_ct-c": GenerationType("Gas CT Candidate", darken_color(gas_ct)),
            "steam-c": GenerationType("Steam Candidate", darken_color(steam)),
            "pv-c": GenerationType("Solar Candidate", darken_color(solar)),
            "wind-c": GenerationType("Wind Candidate", darken_color(wind)),
            "hydro-c": GenerationType("Hydro Candidate", darken_color(hydro)),
            "battery-c": GenerationType("Storage Candidate", darken_color(storage)),
            "ps-c": GenerationType("PS Candidate", darken_color(storage)),
            "other-c": GenerationType("Other Candidate", darken_color(other)),
        }

    def load_from_model(self, gtep_model):
        """This method loads the results from the solved model
        and the metadata into the solution object.

        This method stores solver results, model dimensions, input
        data, and selected commitment/investment expression values
        from a solved ExpansionPlanningModel.

        """
        # Check that the input is a GTEP model object.
        if type(gtep_model) is not ExpansionPlanningModel:
            logger.warning(
                f"Solutions must be loaded from ExpansionPlanningModel objects, not %s"
                % type(gtep_model)
            )
            raise ValueError

        # Check that the model has solver results.
        if gtep_model.results is None:
            raise ValueError(
                "ExpansionPlanningSolution objects loaded from model must have a results component."
            )

        # Store solver results.
        self.results = gtep_model.results

        # Store model dimensions and formulation metadata.
        self.stages = gtep_model.stages
        self.formulation = gtep_model.formulation
        self.data = gtep_model.data
        self.num_reps = gtep_model.num_reps
        self.len_reps = gtep_model.len_reps
        self.num_commit = gtep_model.num_commit
        self.num_dispatch = gtep_model.num_dispatch

        # Store selected expression values for validation/reporting.
        self.expressions = {
            expr.name: pyo.value(expr)
            for expr in gtep_model.model.component_data_objects(pyo.Expression)
            if ("Commitment" in expr.name) or ("Investment" in expr.name)
        }

    def _to_dict(self) -> dict:
        """This method converts solution results into a
        JSON-friendly dictionary.

        This method stores solver summary information, primal variable
        values, selected expression values, and nested result trees
        for downstream validation, reporting, or serialization.

        """

        # Store top-level solver results and expression values.
        results_dict = {
            "solution_loader": self.results.solution_loader,
            "termination_condition": self.results.termination_condition,
            "best_feasible_objective": self.results.best_feasible_objective,
            "best_objective_bound": self.results.best_objective_bound,
            "wallclock_time": self.results.wallclock_time,
            "expressions": self.expressions,
        }

        # Convert termination condition to a JSON-friendly dictionary.
        results_dict["termination_condition"] = {
            "value": self.results.termination_condition.value,
            "name": self.results.termination_condition.name,
        }

        # Store flat primal variable values.
        results_dict["solution_loader"] = {"primals": {}}
        for key, val in self.results.solution_loader.get_primals()._dict.items():
            tmp_key = key

            results_dict["solution_loader"]["primals"][tmp_key] = {
                "name": val[0].name,
                "value": val[0].value,
                "bounds": val[0].bounds,
            }

            # Add binary flag when applicable.
            if val[0].is_binary():
                results_dict["solution_loader"]["primals"][tmp_key]["is_binary"] = val[
                    0
                ].is_binary()

            # Add units when available.
            if val[0].get_units() is not None:
                results_dict["solution_loader"]["primals"][tmp_key]["units"] = (
                    val[0].get_units().name
                )
            else:
                results_dict["solution_loader"]["primals"][tmp_key]["units"] = val[
                    0
                ].get_units()

        # Initialize nested result trees.
        results_dict["primals_tree"] = {}
        results_dict["expressions_tree"] = {}
        for key, val in self.results.solution_loader.get_primals()._dict.items():
            # Split variable name to define nesting depth.
            split_name = val[0].name.split(".")

            tmp_dict = {
                "name": val[0].name,
                "value": val[0].value,
                "bounds": val[0].bounds,
            }

            # Add binary flag when applicable.
            if val[0].is_binary():
                tmp_dict["is_binary"] = val[0].is_binary()

            # Add units when available.
            if val[0].get_units() is not None:
                tmp_dict["units"] = val[0].get_units().name
            else:
                tmp_dict["units"] = val[0].get_units()

            # Add primal variable to nested dictionary.
            def nested_set(this_dict, key, val):
                if len(key) > 1:
                    if key[1] == "binary_indicator_var":
                        this_dict[key[0]] = val
                    else:
                        this_dict.setdefault(key[0], {})
                        nested_set(this_dict[key[0]], key[1:], val)
                else:
                    this_dict[key[0]] = val

            nested_set(results_dict["primals_tree"], split_name, tmp_dict)

        for key, val in self.expressions.items():
            # Split expression name to define nesting depth.
            split_name = key.split(".")

            tmp_dict = {"value": val}

            # Add expression to nested dictionary.
            def nested_set(this_dict, key, val):
                if len(key) > 1:
                    this_dict.setdefault(key[0], {})
                    nested_set(this_dict[key[0]], key[1:], val)
                else:
                    this_dict[key[0]] = val

            nested_set(results_dict["expressions_tree"], split_name, tmp_dict)

        # Store nested expression and primal trees on the solution
        # object.
        self.expressions_tree = results_dict["expressions_tree"]
        self.primals_tree = results_dict["primals_tree"]

        # Build final output dictionary.
        out_dict = {
            "data": self.data.representative_data[0].data,
            "results": results_dict,
        }

        return out_dict

    def read_json(self, filepath):
        # Read a json file
        json_filepath = Path(filepath)
        with open(json_filepath, "r") as fobj:
            json_read = json.loads(fobj.read())

        return json_read

    def to_nested_dict(self, dict_in):
        """Converts a flat dictionary with dot-separated keys into a
        nested dictionary: {time_key: {state: {gen_key: value}}}

        Ignores entries where the second part of the key is 'branch'.

        """

        ignore_this = "branch"
        out_dict = {}

        for key, val in dict_in.items():
            # split the name to figure out depth
            split_name = key.split(".")

            if ignore_this not in split_name[1]:
                # set toplevel defaults
                out_dict.setdefault(split_name[0], {})

                # split things by a predefined prefix
                out_dict[split_name[0]].setdefault(split_name[1], {})

                # specific_key = split_name[1].split(subsplit_key, 1)[1]
                specific_key = "".join(split_name[2:])
                out_dict[split_name[0]][split_name[1]][specific_key] = val

        return out_dict

    def create_results_directory(self, dir_name):
        """This method creates a directory to save model results.

        :param dir_name: Name or path of the directory where results
                         will be saved. Defaults to "results".
        :return: Directory name/path.

        """
        os.makedirs(dir_name, exist_ok=True)
        print(f"\nCreating the directory '{dir_name}' to save the results. ")

        return dir_name

    def save_model_config_to_csv(self, gtep_model, dir_name):
        """Save model configuration settings to a CSV file.

        :param gtep_model: Expansion planning model object.
        :param dir_name: Directory where the CSV file is written.
        """
        config_csv_path = f"{dir_name}/model_config.csv"

        with open(config_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["config_key", "config_value", "value_type"])

            for key, value in sorted(gtep_model.config.items()):
                writer.writerow([key, repr(value), type(value).__name__])

        print(f">>>Saved model configuration to: {config_csv_path}")

    def save_results_in_json_files(self, gtep_model, dir_name, value_threshold=1e-3):
        """This method saves the model results to JSON files.

        Outputs include investments, load shed, costs, flows,
        generation, curtailment, loads, reserves, and storage
        charge/discharge. Creates the results directory if needed.

        Only variable values greater than or equal to the argument
        `value_threshold` are saved to avoid writing near-zero
        numerical values. The default threshold is 1e-3.

        :param gtep_model: Solved expansion planning model object.
        :param dir_name: Directory where JSON files are written.
        :param value_threshold: Minimum variable value to save in the
                                JSON outputs. Defaults to ``1e-3``.

        """

        folder_name = dir_name
        m = gtep_model.model

        valid_names = ["Inst", "Oper", "Disa", "Ext", "Ret"]
        renewable_investments = {}
        dispatchable_investments = {}
        load_shed = {}
        power_flow = {}
        generation = {}
        curtailment = {}
        reserves = {}
        charging = {}
        discharging = {}
        for var in m.component_objects(pyo.Var, descend_into=True):
            for index in var:
                # Save only values above value_threshold to avoid
                # writing near-zero numerical values. The threshold is
                # configurable with a default value of 1e-3.
                if "Shed" in var.name:
                    if pyo.value(var[index]) >= value_threshold:
                        load_shed[var.name + "." + str(index)] = pyo.value(var[index])
                elif "Reserve" in var.name:
                    if pyo.value(var[index]) >= value_threshold:
                        reserves[var.name + "." + str(index)] = pyo.value(var[index])
                elif "Flow" in var.name:
                    if pyo.value(var[index]) >= value_threshold:
                        power_flow[var.name + "." + str(index)] = pyo.value(var[index])
                elif "Generation" in var.name:
                    if pyo.value(var[index]) >= value_threshold:
                        generation[var.name + "." + str(index)] = pyo.value(var[index])
                elif "Curtailment" in var.name:
                    if pyo.value(var[index]) >= value_threshold:
                        curtailment[var.name + "." + str(index)] = pyo.value(var[index])
                elif "storageCharged" in var.name:
                    if pyo.value(var[index]) >= value_threshold:
                        charging[var.name + "." + str(index)] = pyo.value(var[index])
                elif "storageDischarge" in var.name:
                    if pyo.value(var[index]) >= value_threshold:
                        discharging[var.name + "." + str(index)] = pyo.value(var[index])
                for name in valid_names:
                    if name in var.name:
                        if pyo.value(var[index]) >= value_threshold:
                            renewable_investments[var.name + "." + str(index)] = (
                                pyo.value(var[index])
                            )
        for var in m.component_objects(gdp.Disjunct, descend_into=True):
            for index in var:
                for name in valid_names:
                    if name in var.name:
                        if pyo.value(var[index].indicator_var) == True:
                            dispatchable_investments[var.name + "." + str(index)] = (
                                pyo.value(var[index].indicator_var)
                            )

        costs = {}
        for exp in m.component_objects(pyo.Expression, descend_into=True):
            if "Cost" in exp.name or "cost" in exp.name:
                if type(exp) is ScalarExpression:
                    costs[exp.name] = pyo.value(exp)
            if type(exp) is IndexedExpression:
                for e in exp:
                    costs[exp[e].name] = pyo.value(exp[e])

        # Loads are currently read through Prescient, which maps them
        # to buses and stores them as indexed parameters. If a future
        # workflow loads non-Prescient scalar load parameters, this
        # logic may need to be updated.
        loads = {}
        for param in m.component_objects(pyo.Param, descend_into=True):
            if "commitment" in param.name and "loads" in param.name:
                if type(param) is IndexedParam:
                    for p in param:
                        loads[param[p].name] = pyo.value(param[p])

        # Output file names
        output_files = {
            "renewable_investments": renewable_investments,
            "dispatchable_investments": dispatchable_investments,
            "load_shed": load_shed,
            "costs": costs,
            "flows": power_flow,
            "generation": generation,
            "curtailment": curtailment,
            "loads": loads,
            "reserves": reserves,
            "charging": charging,
            "discharging": discharging,
        }

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for name, data in output_files.items():
            filename = f"{folder_name}/{name}.json"
            with open(filename, "w") as fil:
                json.dump(data, fil)

        print(
            f"The following files have been created in the directory '{folder_name}':"
        )
        for name in output_files:
            print(f" - {folder_name}/{name}.json")

    def create_plots(self, case_json, results_path, data_path, plot_type="all"):
        """This method creates generation-mix plots from saved
        solution JSON files.

        It reads investment results for renewable, dispatchable, or
        combined generation and maps generator/storage IDs to unit
        types using `gen.csv` and, when available, `storage.csv`. It
        creates interactive Plotly treemap and/or pie chart HTML files
        in the results `plots` directory.

        :param case_json: Results group to plot. Options are
                          "renewables", "dispatchables", or
                          "combined".
        :param results_path: Directory containing saved JSON result
                             files.
        :param data_path: Directory containing input data files.
        :param plot_type: Plot type to generate. Options are
                          "treemap", "piechart", or "all".
                          Defaults to "all".

        """

        plots_dir = os.path.join(results_path, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            print(f"\nCreated the subdirectory '{plots_dir}' to save the plots.")

        # Get colors for each generation type.
        gen_types = self._get_generation_types()

        def get_gen_arrays(gen_case_json, results_path, data_path, gen_types):
            """This function builds generation-mix dictionaries used
            by plotting functions.

            It reads generator and optional storage data, loads the
            selected investment JSON file, maps assets to generation
            types, and returns generation capacity by time period and
            unit type.

            :param gen_case_json: Results group to process, either
                                  "renewables" or "dispatchables".
            :param results_path: Directory containing saved JSON result
                                 files.
            :param data_path: Directory containing input data files.
            :param gen_types: Mapping of supported unit types to
                                     plot labels and colors.
            :return: Tuple containing gen_mix, gen_mix_arrays,
                     and time_periods.

            """

            # Map generators IDs to Unit Type and PMax
            gen_uid_to_type = {
                row["GEN UID"]: row["Unit Type"].upper()
                for _, row in self.gen_df.iterrows()
            }
            gen_uid_to_pmax = {
                row["GEN UID"]: float(row["PMax MW"])
                for _, row in self.gen_df.iterrows()
            }

            # If storage.csv is available, map storage IDs to storage
            # type and energy capacity.
            storage_uid_to_type = {}
            storage_uid_to_pmax = {}
            if os.path.exists(self.storage_csv_path):
                storage_uid_to_type = {
                    row["name"]: row["storage_type"].upper()
                    for _, row in self.storage_df.iterrows()
                }
                storage_uid_to_pmax = {
                    row["name"]: float(row.get("energy_capacity", 0))
                    for _, row in self.storage_df.iterrows()
                }

            # Read and process saved JSON files for renewables and
            # dispatchables units.
            json_files = {
                "renewables": os.path.join(results_path, "renewable_investments.json"),
                "dispatchables": os.path.join(
                    results_path, "dispatchable_investments.json"
                ),
            }

            if gen_case_json not in json_files:
                raise ValueError(
                    f"Unsupported gen_case_json '{gen_case_json}'. "
                    "Choose 'renewables' or 'dispatchables'."
                )

            dict_in = self.to_nested_dict(self.read_json(json_files[gen_case_json]))

            # Collect all generator/storage keys that appear in the
            # investment results. For this, we loop over investment
            # stages/time keys, over investment-state dictionaries,
            # and at the end collect each generator/storage ID. Use
            # set() to remove duplicates.
            time_keys = list(dict_in.keys())
            keys_set = set()
            for time_key in time_keys:
                for state_dict in dict_in[time_key].values():
                    for asset in state_dict:
                        keys_set.add(asset)

            # Map generator keys to unit types and PMax MW
            gens_keys_to_type = {}
            gens_keys_to_pmax = {}
            for this_key in list(keys_set):
                if this_key in gen_uid_to_type:
                    unit_type = gen_uid_to_type.get(this_key)
                    pmax = gen_uid_to_pmax.get(this_key)
                elif this_key in storage_uid_to_type:
                    unit_type = storage_uid_to_type[this_key]
                    pmax = storage_uid_to_pmax[this_key]
                else:
                    unit_type = None
                    pmax = 0

                # Make unit_type uppercase to ensure case-insensitive
                # matching
                unit_type_upper = unit_type.upper() if unit_type else None

                # Only use if in gen_types dictionary.
                if unit_type_upper and unit_type_upper in gen_types:
                    gens_keys_to_type[this_key] = unit_type_upper
                    gens_keys_to_pmax[this_key] = pmax
                else:
                    raise ValueError(
                        f"[ERROR] Generator or storage '{this_key}' has unknown or unsupported unit type '{unit_type}'."
                    )

            # After building gens_keys_to_type
            unique_types = set(gens_keys_to_type.values())
            gen_types_sorted = sorted(unique_types)  # Alphabetical order

            # Get the modeled year(s) (in order of appearance) from
            # the DAY_AHEAD_renewables.csv time-series.
            time_periods_df = pd.read_csv(f"{data_path}/DAY_AHEAD_renewables.csv")
            time_periods = (
                time_periods_df["Year"].drop_duplicates().astype(str).tolist()
            )

            # Build generation mix by time period and generation type
            gen_mix = {tp: {gt: 0.0 for gt in gen_types_sorted} for tp in time_periods}
            for tp in time_periods:
                for k, val in dict_in.items():
                    for state, gen_dict in val.items():
                        for gen, value in gen_dict.items():
                            unit_type = gens_keys_to_type.get(gen)
                            if gen_case_json == "dispatchables":
                                pmax = gens_keys_to_pmax.get(gen, 0.0)
                                if unit_type in gen_mix[tp]:
                                    gen_mix[tp][unit_type] += pmax * value
                            elif gen_case_json == "renewables":
                                if unit_type in gen_mix[tp]:
                                    gen_mix[tp][unit_type] += value

            gen_mix_arrays = {
                gen_type: np.array(
                    [gen_mix[tp].get(gen_type, 0.0) for tp in time_periods]
                )
                for gen_type in gen_types_sorted
            }

            return gen_mix, gen_mix_arrays, time_periods

        # Define functions to create a treemap and pie chart
        # interactive Plotly plots for the generation mix. The user
        # can select which one to use by setting up the plot_type
        # option.
        def plotly_treemap_gen_mix(
            gen_mix, gen_types, results_path, case_json, small_pct_threshold=5
        ):
            """This function creates interactive Plotly treemap plots
            of generation mix.

            One HTML treemap is created and shows the share of
            generation capacity by the unit type defined in case_json.

            :param gen_mix: Dictionary of generation mix by time
                            period and unit type.
            :param gen_types: Mapping of unit types to plot labels and
                              colors.
            :param results_path: Directory where plot files are saved.
            :param case_json: Results group name used in output file
                              names. Options are "renewables",
                              "dispatchables", or "combined".
            :param small_pct_threshold: Minimum percentage used for
                                        displaying labels. Defaults to
                                        5.

            """

            for tp, mix in gen_mix.items():
                filtered_mix = {
                    k: v for k, v in mix.items() if v > 0 and k in gen_types
                }
                if not filtered_mix:
                    continue

                total = sum(filtered_mix.values())
                sorted_items = sorted(filtered_mix.items(), key=lambda x: -x[1])
                sizes = [v for k, v in sorted_items]
                pcts = [v / total * 100 for k, v in sorted_items]
                labels = []
                colors = []
                customdata = []

                for (k, v), pct in zip(sorted_items, pcts):
                    label = f"{gen_types[k].label}<br>{int(v)} MW<br>{pct:.1f}%"
                    labels.append(label if pct >= small_pct_threshold else "")
                    colors.append(gen_types[k].color)
                    customdata.append(f"{gen_types[k].label} ({int(v)} MW, {pct:.1f}%)")

                fig = go.Figure(
                    go.Treemap(
                        labels=[gen_types[k].label for k, v in sorted_items],
                        parents=[""] * len(sorted_items),
                        values=sizes,
                        marker=dict(colors=colors),
                        textinfo="label+value+percent entry",
                        hovertext=customdata,
                        hoverinfo="text",
                    )
                )

                fig.update_layout(
                    title=f"Generation Mix Treemap - {tp}",
                    width=900,
                    height=600,
                    margin=dict(t=50, l=25, r=25, b=25),
                )

                # Save as an interactive HTML
                plot_path = f"{results_path}/plots/treemap_{case_json}_{tp}.html"
                fig.write_html(f"{plot_path}")
                print(f" -> Saved interactive treemap for {tp} to {plot_path}")

        def plotly_pie_gen_mix(gen_mix, gen_types, results_path, case_json):
            """This function creates interactive Plotly pie charts of
            generation mix.

            One HTML pie chart is created for each unit type and shows
            the share of generation capacity by unit type.

            :param gen_mix: Dictionary of generation mix by time period
                            and unit type.
            :param gen_types: Mapping of unit types to plot labels and
                              colors.
            :param results_path: Directory where plot files are saved.
            :param case_json: Results group name used in output file
                              names.

            """

            for tp, mix in gen_mix.items():
                filtered_mix = {
                    k: v for k, v in mix.items() if v > 0 and k in gen_types
                }
                if not filtered_mix:
                    continue

                sizes = [filtered_mix[k] for k in filtered_mix]
                total = sum(sizes)
                labels = [
                    f"{gen_types[k].label} ({int(filtered_mix[k])} MW, {filtered_mix[k]/total*100:.1f}%)"
                    for k in filtered_mix
                ]
                colors = [gen_types[k].color for k in filtered_mix]

                # Plotly pie chart
                fig = go.Figure(
                    go.Pie(
                        labels=[gen_types[k].label for k in filtered_mix],
                        values=sizes,
                        marker=dict(colors=colors, line=dict(color="white", width=1)),
                        textinfo="label+percent",
                        hoverinfo="label+value+percent",
                        pull=[0.05]
                        * len(sizes),  # Slightly "explode" all slices for separation
                        hole=0,  # 0 for pie, >0 for donut
                    )
                )

                fig.update_layout(
                    title=f"Generation Mix Pie Chart - {tp}",
                    width=700,
                    height=700,
                    margin=dict(t=50, l=25, r=25, b=25),
                    showlegend=True,
                )

                # Save as an interactive HTML
                plot_path = f"{results_path}/plots/piechart_{case_json}_{tp}.html"
                fig.write_html(f"{plot_path}")
                print(f" -> Saved interactive pie chart for {tp} to {plot_path}")

        # ------------------------------------------------------------
        # The creation of plots under create_plots starts here, after
        # the plotting helper functions.

        if case_json != "combined":
            # Create gen_mix dictionary and arrays needed to plot
            # renewables and dispatchables types in separate plots.
            gen_mix, gen_mix_arrays, time_periods = get_gen_arrays(
                case_json, results_path, data_path, gen_types
            )
        else:
            # Combine gen_mix dictionary and arrays needed to plot
            # renewables and dispatchables types in the same plot.
            gen_mix_ren, gen_mix_arrays_ren, time_periods_ren = get_gen_arrays(
                "renewables", results_path, data_path, gen_types
            )
            gen_mix_disp, gen_mix_arrays_disp, time_periods_disp = get_gen_arrays(
                "dispatchables", results_path, data_path, gen_types
            )

            # Check that time_periods are the same
            if time_periods_ren != time_periods_disp:
                raise ValueError(
                    "Time periods for renewables and dispatchables do not match!"
                )
            time_periods = time_periods_ren  # or use time_periods_disp too

            # Get the union of all time periods and all types
            all_time_periods = sorted(
                set(gen_mix_ren.keys()) | set(gen_mix_disp.keys())
            )
            all_types = sorted(
                set(
                    t
                    for mix in [gen_mix_ren, gen_mix_disp]
                    for v in mix.values()
                    for t in v
                )
            )

            # Merge gen_mix and gen_mix arrays
            gen_mix = {}
            for tp in all_time_periods:
                gen_mix[tp] = {}
                for gt in all_types:
                    val_ren = gen_mix_ren.get(tp, {}).get(gt, 0.0)
                    val_disp = gen_mix_disp.get(tp, {}).get(gt, 0.0)
                    gen_mix[tp][gt] = val_ren + val_disp

            gen_mix_arrays = {
                gt: np.array([gen_mix[tp].get(gt, 0.0) for tp in all_time_periods])
                for gt in all_types
            }

        if plot_type == "treemap":
            plotly_treemap_gen_mix(gen_mix, gen_types, results_path, case_json)
        elif plot_type == "piechart":
            plotly_pie_gen_mix(gen_mix, gen_types, results_path, case_json)
        elif plot_type == "all":
            plotly_treemap_gen_mix(gen_mix, gen_types, results_path, case_json)
            plotly_pie_gen_mix(gen_mix, gen_types, results_path, case_json)
        else:
            raise ValueError(
                f"Plot type '{plot_type}' is not supported. Please choose between 'treemap' or 'piechart'."
            )

    def create_stackgraph_and_metrics(self, results_path, rep_days, day_hour_list):
        """Create and save an interactive stackgraph of dispatch results.

        This method reads saved JSON result files, organizes generation,
        storage charge/discharge if enabled, load, and load-shed values by
        stage, representative period, commitment period, and dispatch
        period, and creates a Plotly stacked bar chart with total load.

        Generator plotting categories are determined using ``gen.csv`` via
        ``GEN UID`` and ``Unit Type`` instead of relying on generator names
        ending with a unit-type suffix.

        :param results_path: Directory containing saved JSON result files
                             and where the plot will be saved.
        :param rep_days: List of representative day labels used for
                         formatting the x-axis.
        :param day_hour_list: Representative day/hour metadata used by
                              downstream metrics workflows.
        """

        try:
            import ujson as json
        except ImportError:
            import json

        plots_dir = os.path.join(results_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        def load_json(name):
            """Load a saved JSON results file."""
            with open(f"{results_path}/{name}.json", "r") as f:
                return json.load(f)

        def parse_result_key(key, dispatch_required=False):
            """Parse saved result key into time indices and asset name.

            This avoids using ``pyo.ComponentUID`` because saved JSON keys
            are constructed strings and may contain asset names that are
            not valid ComponentUID strings.
            """

            def get_index(component_name, required=True):
                match = re.search(rf"{component_name}\[([^\]]+)\]", key)

                if match is None:
                    if required:
                        raise RuntimeError(
                            f"Could not find {component_name} index in key: {key}"
                        )
                    return None

                value = match.group(1).strip("'\"")

                try:
                    return int(value)
                except ValueError:
                    return value

            stage = get_index("investmentStage")
            period = get_index("representativePeriod")
            commitment = get_index("commitmentPeriod")
            dispatch = get_index("dispatchPeriod", required=dispatch_required)

            # Support old manually constructed keys like:
            #   component.asset_name
            # and Pyomo-like keys like:
            #   component[asset_name]
            last_part = key.rsplit(".", 1)[-1]

            if "[" in last_part and last_part.endswith("]"):
                asset_name = last_part.split("[", 1)[1][:-1].strip("'\"")
            else:
                asset_name = last_part.strip("'\"")

            return stage, period, commitment, dispatch, asset_name

        gen_data = load_json("generation")
        loads_data = load_json("loads")
        load_shed_data = load_json("load_shed")
        reserves_data = load_json("reserves")
        charging_data = load_json("charging")
        discharging_data = load_json("discharging")

        # Note that these are the same colors used in the stack plots
        # and pie charts above.
        def darken_color(hex_color, percent=0.2):
            """Darken a hex color by a given percent."""
            hex_color = hex_color.lstrip("#")
            rgb = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
            darker_rgb = [max(0, int(c * (1 - percent))) for c in rgb]
            return "#" + "".join(f"{c:02x}" for c in darker_rgb)

        tab20 = plt.get_cmap("tab20")
        GEN_TYPES = {
            "nuclear": mcolors.to_hex(tab20(2)),
            "coal": mcolors.to_hex(tab20(5)),
            "cc_gas": mcolors.to_hex(tab20(1)),
            "ct_gas": mcolors.to_hex(tab20(3)),
            "thermal_other": mcolors.to_hex(tab20(13)),
            "steam": mcolors.to_hex(tab20(14)),
            "solar": mcolors.to_hex(tab20(9)),
            "wind": mcolors.to_hex(tab20(11)),
            "hydro": mcolors.to_hex(tab20(19)),
            "battery_discharge": mcolors.to_hex(tab20(15)),
            "battery_charge": mcolors.to_hex(tab20(15)),
            "ES4": mcolors.to_hex(tab20(17)),
            "ps": mcolors.to_hex(tab20(19)),
            "dr": mcolors.to_hex(tab20(18)),
            "other": mcolors.to_hex(tab20(0)),
            # Candidates: darker than original.
            "gas_cc-c": darken_color(mcolors.to_hex(tab20(1))),
            "gas_ct-c": darken_color(mcolors.to_hex(tab20(3))),
            "steam-c": darken_color(mcolors.to_hex(tab20(14))),
            "pv-c": darken_color(mcolors.to_hex(tab20(9))),
            "wind-c": darken_color(mcolors.to_hex(tab20(11))),
            "hydro-c": darken_color(mcolors.to_hex(tab20(19))),
            "battery-c": darken_color(mcolors.to_hex(tab20(15))),
            "ES4-c": darken_color(mcolors.to_hex(tab20(17))),
            "ps-c": darken_color(mcolors.to_hex(tab20(19))),
            "other-c": darken_color(mcolors.to_hex(tab20(0))),
        }

        GEN_TYPE_HATCHES = {
            # No hatch pattern for original types.
            "nuclear": "",
            "coal": "",
            "cc_gas": "",
            "ct_gas": "",
            "thermal_other": "",
            "steam": "",
            "solar": "",
            "wind": "",
            "hydro": "",
            "battery_discharge": "",
            "battery_charge": "",
            "ES4": "",
            "ps": "",
            "dr": "",
            "other": "",
            # Candidates get a hatch pattern.
            "gas_cc-c": "////",
            "gas_ct-c": "////",
            "steam-c": "////",
            "pv-c": "////",
            "wind-c": "////",
            "hydro-c": "////",
            "battery-c": "////",
            "ES4-c": "////",
            "ps-c": "////",
            "other-c": "////",
        }

        GEN_TYPE_ALIASES = {
            "nuclear": "Nuclear",
            "coal": "Coal",
            "cc_gas": "CC",
            "ct_gas": "CT",
            "thermal_other": "Thermal",
            "steam": "Steam",
            "solar": "Solar",
            "wind": "Wind",
            "hydro": "Hydro",
            "battery_charge": "Battery Charging",
            "battery_discharge": "Battery Discharging",
            "ES4": "ES4",
            "ps": "Pumped Storage",
            "dr": "DR",
            "other": "Other",
            "steam-c": "Steam (Candidate)",
            "gas_cc-c": "CC (Candidate)",
            "gas_ct-c": "CT (Candidate)",
            "pv-c": "Solar (Candidate)",
            "wind-c": "Wind (Candidate)",
            "hydro-c": "Hydro (Candidate)",
            "battery-c": "Battery (Candidate)",
            "ES4-c": "ES4 (Candidate)",
            "ps-c": "Pumped Storage (Candidate)",
            "other-c": "Other (Candidate)",
        }

        # Map generator IDs to Unit Type using gen.csv. This avoids
        # relying on the generator name ending with the unit type.
        gen_uid_to_type = {
            str(row["GEN UID"]): str(row["Unit Type"]).upper()
            for _, row in self.gen_df.iterrows()
        }

        # Map storage IDs to storage type using storage.csv, if available.
        if hasattr(self, "storage_csv_path") and os.path.exists(self.storage_csv_path):
            storage_uid_to_type = {
                str(row["name"]): str(row["storage_type"]).upper()
                for _, row in self.storage_df.iterrows()
            }
        else:
            storage_uid_to_type = {}

        # Map Unit Type values from gen.csv to stackgraph categories.
        unit_type_to_plot_type = {
            "CC": "cc_gas",
            "CT": "ct_gas",
            "COAL": "coal",
            "NUC": "nuclear",
            "NUCLEAR": "nuclear",
            "PV": "solar",
            "RTPV": "solar",
            "SOLAR": "solar",
            "WIND": "wind",
            "HYDRO": "hydro",
            "THERM": "thermal_other",
            "THERMAL": "thermal_other",
            "STEAM": "steam",
            "ST": "steam",
            "GEO": "geothermal" if "geothermal" in GEN_TYPES else "other",
            "DR": "dr",
            "ES4": "ES4",
            "PS": "ps",
        }

        # Candidate resources can be plotted separately when a matching
        # candidate plotting category exists.
        candidate_unit_type_to_plot_type = {
            "CC": "gas_cc-c",
            "CT": "gas_ct-c",
            "STEAM": "steam-c",
            "ST": "steam-c",
            "PV": "pv-c",
            "RTPV": "pv-c",
            "SOLAR": "pv-c",
            "WIND": "wind-c",
            "HYDRO": "hydro-c",
            "BATTERY": "battery-c",
            "PS": "ps-c",
            "ES4": "ES4-c",
        }

        def get_generator_plot_type(gen_name):
            """Map generator name to stackgraph plotting type using gen.csv."""
            gen_name = str(gen_name)

            if gen_name not in gen_uid_to_type:
                raise RuntimeError(
                    f"Cannot map generator '{gen_name}' to a Unit Type. "
                    "The generator was not found in gen.csv."
                )

            unit_type = gen_uid_to_type[gen_name]

            if gen_name.endswith("-c"):
                candidate_type = candidate_unit_type_to_plot_type.get(unit_type)
                if candidate_type in GEN_TYPES:
                    return candidate_type

            plot_type = unit_type_to_plot_type.get(unit_type)

            if plot_type in GEN_TYPES:
                return plot_type

            raise RuntimeError(
                f"Cannot map generator '{gen_name}' with Unit Type "
                f"'{unit_type}' to a supported stackgraph type."
            )

        def validate_storage_name(storage_name):
            """Validate storage name using storage.csv when available."""
            if not storage_uid_to_type:
                return

            storage_name = str(storage_name)

            if storage_name not in storage_uid_to_type:
                raise RuntimeError(
                    f"Cannot map storage asset '{storage_name}' to a storage type. "
                    "The storage asset was not found in storage.csv."
                )

        generation = {}

        for g, val in gen_data.items():
            stage, period, commitment, dispatch, gen_name = parse_result_key(
                g,
                dispatch_required=True,
            )

            if stage not in generation:
                generation[stage] = {}
            stage_dict = generation[stage]

            if period not in stage_dict:
                stage_dict[period] = {}
            period_dict = stage_dict[period]

            if commitment not in period_dict:
                commitment_dict = period_dict[commitment] = {}
            else:
                commitment_dict = period_dict[commitment]

            if dispatch not in commitment_dict:
                commitment_dict[dispatch] = dict.fromkeys(GEN_TYPES, 0)
            dispatch_dict = commitment_dict[dispatch]

            gen_type = get_generator_plot_type(gen_name)
            dispatch_dict[gen_type] += val

        # Add battery charging data to generation structure.
        for g, val in charging_data.items():
            stage, period, commitment, dispatch, storage_name = parse_result_key(
                g,
                dispatch_required=True,
            )

            if stage not in generation:
                generation[stage] = {}
            stage_dict = generation[stage]

            if period not in stage_dict:
                stage_dict[period] = {}
            period_dict = stage_dict[period]

            if commitment not in period_dict:
                commitment_dict = period_dict[commitment] = {}
            else:
                commitment_dict = period_dict[commitment]

            if dispatch not in commitment_dict:
                commitment_dict[dispatch] = dict.fromkeys(GEN_TYPES, 0)
            dispatch_dict = commitment_dict[dispatch]

            validate_storage_name(storage_name)
            dispatch_dict["battery_charge"] -= val

        # Add battery discharging data to generation structure.
        for g, val in discharging_data.items():
            stage, period, commitment, dispatch, storage_name = parse_result_key(
                g,
                dispatch_required=True,
            )

            if stage not in generation:
                generation[stage] = {}
            stage_dict = generation[stage]

            if period not in stage_dict:
                stage_dict[period] = {}
            period_dict = stage_dict[period]

            if commitment not in period_dict:
                commitment_dict = period_dict[commitment] = {}
            else:
                commitment_dict = period_dict[commitment]

            if dispatch not in commitment_dict:
                commitment_dict[dispatch] = dict.fromkeys(GEN_TYPES, 0)
            dispatch_dict = commitment_dict[dispatch]

            validate_storage_name(storage_name)
            dispatch_dict["battery_discharge"] += val

        total_charging = sum(charging_data.values())
        total_discharging = sum(discharging_data.values())

        charging_by_suffix = defaultdict(float)
        for g, val in charging_data.items():
            name = g.split(".")[-1]
            if name.endswith("_battery"):
                charging_by_suffix["battery"] += val
            elif name.endswith("_ps"):
                charging_by_suffix["ps"] += val
            else:
                charging_by_suffix["other"] += val

        discharging_by_suffix = defaultdict(float)
        for g, val in discharging_data.items():
            name = g.split(".")[-1]
            if name.endswith("_battery"):
                discharging_by_suffix["battery"] += val
            elif name.endswith("_ps"):
                discharging_by_suffix["ps"] += val
            else:
                discharging_by_suffix["other"] += val

        time_periods = [
            (s, p, c, d)
            for s in generation
            for p in generation[s]
            for c in generation[s][p]
            for d in generation[s][p][c]
        ]

        times = list(range(len(time_periods)))

        loads = {}
        for g, val in loads_data.items():
            stage, period, commitment, _, _ = parse_result_key(
                g,
                dispatch_required=False,
            )

            if stage not in loads:
                loads[stage] = {}
            stage_dict = loads[stage]

            if period not in stage_dict:
                stage_dict[period] = {}
            period_dict = stage_dict[period]

            if commitment not in period_dict:
                period_dict[commitment] = 0

            period_dict[commitment] += val

        loads_trace = []
        for s, p, c, d in time_periods:
            try:
                total_load = loads[s][p][c]
            except KeyError:
                total_load = 0
            loads_trace.append(total_load)

        load_shed = {}
        for g, val in load_shed_data.items():
            stage, period, commitment, _, _ = parse_result_key(
                g,
                dispatch_required=False,
            )

            if stage not in load_shed:
                load_shed[stage] = {}
            stage_dict = load_shed[stage]

            if period not in stage_dict:
                stage_dict[period] = {}
            period_dict = stage_dict[period]

            if commitment not in period_dict:
                period_dict[commitment] = 0

            period_dict[commitment] += val

        load_shed_trace = []
        for s, p, c, d in time_periods:
            try:
                total_shed = load_shed[s][p][c]
            except KeyError:
                total_shed = 0
            load_shed_trace.append(total_shed)

        HATCH_TO_PATTERN = {
            "": "",
            "....": ".",
            "////": "/",
            "xxxx": "x",
        }

        def plotly_stackgraph(
            times,
            time_periods,
            generation,
            GEN_TYPES,
            GEN_TYPE_ALIASES,
            GEN_TYPE_HATCHES,
            HATCH_TO_PATTERN,
            results_path,
        ):
            """Create an interactive Plotly stackgraph for representative days."""

            n_hours_per_day = 24
            n_rep_days = len(rep_days)
            n_points = n_hours_per_day * n_rep_days

            rep_days_dt = [pd.to_datetime(d) for d in rep_days]

            x_labels = []
            tickvals = []
            ticktext = []

            for i, day in enumerate(rep_days_dt):
                for h in range(n_hours_per_day):
                    idx = i * n_hours_per_day + h

                    if h == 0:
                        label = day.strftime("%b-%d 00:00")
                        x_labels.append(label)
                        tickvals.append(idx)
                        ticktext.append(label)
                    elif h == 12:
                        label = day.strftime("%b-%d 12:00")
                        x_labels.append(label)
                        tickvals.append(idx)
                        ticktext.append(label)
                    else:
                        x_labels.append("")

            times = list(range(n_points))

            traces = []
            for name, color in GEN_TYPES.items():
                label = GEN_TYPE_ALIASES.get(name, name)

                values = np.array(
                    [generation[s][p][c][d][name] for s, p, c, d in time_periods]
                )

                hatch = GEN_TYPE_HATCHES.get(name, "")
                pattern_shape = HATCH_TO_PATTERN.get(hatch, "")
                opacity = 0.7 if hatch else 1.0

                traces.append(
                    go.Bar(
                        x=times,
                        y=values,
                        name=label,
                        marker_color=color,
                        marker_pattern_shape=pattern_shape,
                        opacity=opacity,
                        marker_line_width=0,
                    )
                )

            traces.append(
                go.Bar(
                    x=times,
                    y=load_shed_trace,
                    name="Load Shed",
                    marker_color=mcolors.to_hex(tab20(7)),
                    opacity=0.7,
                    marker_line_width=0,
                )
            )

            fig = go.Figure(data=traces)

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=loads_trace,
                    mode="lines+markers",
                    name="Total Load",
                    line=dict(color="black", width=3, dash="dash"),
                    marker=dict(size=4, color="black"),
                    showlegend=True,
                )
            )

            fig.update_layout(
                barmode="relative",
                bargap=0,
                title="Generation Mix (Representative Days)",
                xaxis=dict(
                    title="Representative Days (labeled every 12 hours)",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    showgrid=True,
                    gridcolor="gray",
                    gridwidth=0.7,
                    linecolor="black",
                    mirror=True,
                ),
                yaxis=dict(
                    title="Nameplate Capacity [MW]",
                    showgrid=True,
                    gridcolor="gray",
                    gridwidth=0.7,
                    linecolor="black",
                    mirror=True,
                ),
                legend=dict(
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=14),
                    title="Generation Type",
                ),
                width=1200,
                height=600,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            for i in range(1, n_rep_days):
                fig.add_vline(
                    x=i * n_hours_per_day,
                    line=dict(color="gray", width=1, dash="dot"),
                    opacity=0.5,
                )

            all_series = {
                name: np.array(
                    [generation[s][p][c][d][name] for s, p, c, d in time_periods]
                )
                for name in GEN_TYPES
            }

            positive_stack = np.sum(
                [np.clip(vals, 0, None) for vals in all_series.values()],
                axis=0,
            )
            negative_stack = np.sum(
                [np.clip(vals, None, 0) for vals in all_series.values()],
                axis=0,
            )

            ymin = negative_stack.min() if len(negative_stack) else 0
            ymax = positive_stack.max() if len(positive_stack) else 0

            if loads_trace:
                ymax = max(ymax, max(loads_trace))

            lower = ymin * 1.25 if ymin < 0 else -1
            upper = ymax * 1.25 if ymax > 0 else 1

            fig.update_yaxes(
                range=[lower, upper],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="black",
            )

            plot_path = f"{results_path}/plots/stackgraph_generators_interactive.html"
            fig.write_html(f"{plot_path}")
            print(f" -> Saved interactive stackgraph to {plot_path}")

        # def create_stackgraph_and_metrics(self, results_path, rep_days, day_hour_list):

        #     try:
        #         import ujson as json
        #     except ImportError:
        #         import json

        #     with open(f"{results_path}/generation.json", "r") as F:
        #         gen_data = json.load(F)

        #     with open(f"{results_path}/loads.json", "r") as f:
        #         loads_data = json.load(f)

        #     with open(f"{results_path}/load_shed.json", "r") as f:
        #         load_shed_data = json.load(f)

        #     with open(f"{results_path}/reserves.json", "r") as f:
        #         reserves_data = json.load(f)

        #     with open(f"{results_path}/charging.json", "r") as f:
        #         charging_data = json.load(f)

        #     with open(f"{results_path}/discharging.json", "r") as f:
        #         discharging_data = json.load(f)

        #     def parse_result_key(key, dispatch_required=False):
        #         """Parse saved result key into time indices and asset name.

        #         This avoids using pyo.ComponentUID because saved JSON keys
        #         are constructed strings and may contain asset names that are
        #         not valid ComponentUID strings.
        #         """

        #         def get_index(component_name, required=True):
        #             match = re.search(rf"{component_name}\[([^\]]+)\]", key)

        #             if match is None:
        #                 if required:
        #                     raise RuntimeError(
        #                         f"Could not find {component_name} index in key: {key}"
        #                     )
        #                 return None

        #             value = match.group(1).strip("'\"")

        #             try:
        #                 return int(value)
        #             except ValueError:
        #                 return value

        #         stage = get_index("investmentStage")
        #         period = get_index("representativePeriod")
        #         commitment = get_index("commitmentPeriod")
        #         dispatch = get_index("dispatchPeriod", required=dispatch_required)

        #         # Support keys saved as:
        #         #   component.asset_name
        #         # and keys saved as:
        #         #   component[asset_name]
        #         last_part = key.rsplit(".", 1)[-1]

        #         if "[" in last_part and last_part.endswith("]"):
        #             asset_name = last_part.split("[", 1)[1][:-1].strip("'\"")
        #         else:
        #             asset_name = last_part.strip("'\"")

        #         return stage, period, commitment, dispatch, asset_name

        #     # Note that these are the same colors used in the stack plots
        #     # and pie charts above
        #     def darken_color(hex_color, percent=0.2):
        #         """Darken a hex color by a given percent (0.2 = 20%)"""
        #         hex_color = hex_color.lstrip("#")
        #         rgb = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
        #         darker_rgb = [max(0, int(c * (1 - percent))) for c in rgb]
        #         return "#" + "".join(f"{c:02x}" for c in darker_rgb)

        #     tab20 = plt.get_cmap("tab20")
        #     GEN_TYPES = {
        #         "nuclear": mcolors.to_hex(tab20(2)),
        #         "coal": mcolors.to_hex(tab20(5)),
        #         "cc_gas": mcolors.to_hex(tab20(1)),
        #         "ct_gas": mcolors.to_hex(tab20(3)),
        #         "thermal_other": mcolors.to_hex(tab20(13)),
        #         "steam": mcolors.to_hex(tab20(14)),
        #         "solar": mcolors.to_hex(tab20(9)),
        #         "wind": mcolors.to_hex(tab20(11)),
        #         "hydro": mcolors.to_hex(tab20(19)),
        #         "battery_discharge": mcolors.to_hex(tab20(15)),
        #         "battery_charge": mcolors.to_hex(tab20(15)),
        #         "ES4": mcolors.to_hex(tab20(17)),
        #         "ps": mcolors.to_hex(tab20(19)),
        #         "dr": mcolors.to_hex(tab20(18)),
        #         # Candidates: 20% darker than original
        #         "gas_cc-c": darken_color(mcolors.to_hex(tab20(1))),
        #         "gas_ct-c": darken_color(mcolors.to_hex(tab20(3))),
        #         "pv-c": darken_color(mcolors.to_hex(tab20(9))),
        #         "wind-c": darken_color(mcolors.to_hex(tab20(11))),
        #         "hydro-c": darken_color(mcolors.to_hex(tab20(19))),
        #         "battery-c": darken_color(mcolors.to_hex(tab20(15))),
        #         "ES4-c": darken_color(mcolors.to_hex(tab20(17))),
        #         "ps-c": darken_color(mcolors.to_hex(tab20(19))),
        #         "steam-c": darken_color(mcolors.to_hex(tab20(14))),
        #     }
        #     GEN_TYPE_HATCHES = {
        #         # No hatch pattern for "original" types
        #         "nuclear": "",
        #         "coal": "",
        #         "cc_gas": "",
        #         "ct_gas": "",
        #         "thermal_other": "",
        #         "steam": "",
        #         "solar": "",
        #         "wind": "",
        #         "hydro": "",
        #         "battery_discharge": "",
        #         "battery_charge": "",
        #         "ES4": "",
        #         "dr": "",
        #         # Candidates get a hatch pattern
        #         "gas_cc-c": "////",
        #         "gas_ct-c": "////",
        #         "steam-c": "////",
        #         "pv-c": "////",
        #         "wind-c": "////",
        #         "hydro-c": "////",
        #         "battery-c": "////",
        #         "ES4-c": "////",
        #     }
        #     GEN_TYPE_ALIASES = {
        #         "nuclear": "Nuclear",
        #         "coal": "Coal",
        #         "cc_gas": "CC",
        #         "ct_gas": "CT",
        #         "thermal_other": "Thermal",
        #         "steam": "Steam",
        #         "solar": "Solar",
        #         "wind": "Wind",
        #         "hydro": "Hydro",
        #         "battery_charge": "Battery Charging",
        #         "battery_discharge": "Battery Discharging",
        #         "ES4": "ES4",
        #         "dr": "DR",
        #         "steam-c": "Steam (Candidate)",
        #         "gas_cc-c": "CC (Candidate)",
        #         "gas_ct-c": "CT (Candidate)",
        #         "pv-c": "Solar (Candidate)",
        #         "wind-c": "Wind (Candidate)",
        #         "hydro-c": "Hydro (Candidate)",
        #         "ES4-c": "ES4 (Candidate)",
        #     }

        #     # Map generator IDs to Unit Type using gen.csv. This avoids
        #     # relying on the generator name ending with the unit type.
        #     gen_uid_to_type = {
        #         str(row["GEN UID"]): str(row["Unit Type"]).upper()
        #         for _, row in self.gen_df.iterrows()
        #     }

        #     # Map storage IDs to storage type using storage.csv, if
        #     # available. Storage charging/discharging is plotted using the
        #     # artificial battery_charge and battery_discharge categories.
        #     if hasattr(self, "storage_csv_path") and os.path.exists(self.storage_csv_path):
        #         storage_uid_to_type = {
        #             str(row["name"]): str(row["storage_type"]).upper()
        #             for _, row in self.storage_df.iterrows()
        #         }
        #     else:
        #         storage_uid_to_type = {}

        #     # Map Unit Type values from gen.csv to the stackgraph plotting
        #     # categories used in GEN_TYPES.
        #     unit_type_to_plot_type = {
        #         "CC": "cc_gas",
        #         "CT": "ct_gas",
        #         "COAL": "coal",
        #         "NUC": "nuclear",
        #         "NUCLEAR": "nuclear",
        #         "PV": "solar",
        #         "RTPV": "solar",
        #         "SOLAR": "solar",
        #         "WIND": "wind",
        #         "HYDRO": "hydro",
        #         "THERM": "thermal_other",
        #         "THERMAL": "thermal_other",
        #         "STEAM": "steam",
        #         "ST": "steam",
        #         "ES4": "ES4",
        #         "DR": "dr",
        #     }

        #     # Candidate resources can be plotted separately when a
        #     # matching candidate plotting category exists.
        #     candidate_unit_type_to_plot_type = {
        #         "CC": "gas_cc-c",
        #         "CT": "gas_ct-c",
        #         "PV": "pv-c",
        #         "RTPV": "pv-c",
        #         "SOLAR": "pv-c",
        #         "WIND": "wind-c",
        #         "HYDRO": "hydro-c",
        #         "STEAM": "steam-c",
        #         "ST": "steam-c",
        #         "ES4": "ES4-c",
        #         "BATTERY": "battery-c",
        #         "PS": "ps-c",
        #     }

        #     def get_generator_plot_type(gen_name):
        #         """Map generator name to stackgraph plotting type using gen.csv."""
        #         gen_name = str(gen_name)

        #         if gen_name not in gen_uid_to_type:
        #             raise RuntimeError(
        #                 f"Cannot map generator '{gen_name}' to a Unit Type. "
        #                 "The generator was not found in gen.csv."
        #             )

        #         unit_type = gen_uid_to_type[gen_name]

        #         if gen_name.endswith("-c"):
        #             candidate_type = candidate_unit_type_to_plot_type.get(unit_type)
        #             if candidate_type in GEN_TYPES:
        #                 return candidate_type

        #         plot_type = unit_type_to_plot_type.get(unit_type)

        #         if plot_type in GEN_TYPES:
        #             return plot_type

        #         raise RuntimeError(
        #             f"Cannot map generator '{gen_name}' with Unit Type "
        #             f"'{unit_type}' to a supported stackgraph type."
        #         )

        #     def validate_storage_name(storage_name):
        #         """Validate storage name using storage.csv when available."""
        #         if not storage_uid_to_type:
        #             return

        #         storage_name = str(storage_name)

        #         if storage_name not in storage_uid_to_type:
        #             raise RuntimeError(
        #                 f"Cannot map storage asset '{storage_name}' to a storage type. "
        #                 "The storage asset was not found in storage.csv."
        #             )

        #     generation = {}
        #     for g, val in gen_data.items():
        #         stage, period, commitment, dispatch, gen_name = parse_result_key(
        #             g,
        #             dispatch_required=True,
        #         )

        #         if stage not in generation:
        #             generation[stage] = {}
        #         stage_dict = generation[stage]

        #         if period not in stage_dict:
        #             stage_dict[period] = {}
        #         period_dict = stage_dict[period]

        #         if commitment not in period_dict:
        #             commitment_dict = period_dict[commitment] = {}
        #         else:
        #             commitment_dict = period_dict[commitment]

        #         if dispatch not in commitment_dict:
        #             commitment_dict[dispatch] = dict.fromkeys(GEN_TYPES, 0)
        #         dispatch_dict = commitment_dict[dispatch]

        #         gen_type = get_generator_plot_type(gen_name)
        #         dispatch_dict[gen_type] += val

        #     # generation = {}
        #     # for g, val in gen_data.items():
        #     #     c = list(pyo.ComponentUID(g)._cids)
        #     #     _, (stage,) = c.pop(0)
        #     #     if stage not in generation:
        #     #         generation[stage] = {}
        #     #     stage_dict = generation[stage]

        #     #     _, (period,) = c.pop(0)
        #     #     if period not in stage_dict:
        #     #         stage_dict[period] = {}
        #     #     period_dict = stage_dict[period]

        #     #     _, (commitment,) = c.pop(0)
        #     #     if commitment not in period_dict:
        #     #         period_dict[commitment] = {}
        #     #     commitment_dict = period_dict[commitment]

        #     #     _, (dispatch,) = c.pop(0)
        #     #     if dispatch not in commitment_dict:
        #     #         commitment_dict[dispatch] = dict.fromkeys(GEN_TYPES, 0)
        #     #     dispatch_dict = commitment_dict[dispatch]

        #     #     # gen_name = c[-1][0]
        #     #     # _type = None
        #     #     # for gt in GEN_TYPES:
        #     #     #     if gen_name.endswith(gt):
        #     #     #         _type = gt
        #     #     #         break
        #     #     # if _type is None:
        #     #     #     raise RuntimeError(f"Cannot map generator name '{gen_name}' to type")
        #     #     # dispatch_dict[_type] += val
        #     #     gen_name = c[-1][0]
        #     #     gen_type = get_generator_plot_type(gen_name)
        #     #     dispatch_dict[gen_type] += val

        #     # Add battery charging data to generation structure
        #     for g, val in charging_data.items():
        #         c = list(pyo.ComponentUID(g)._cids)
        #         _, (stage,) = c.pop(0)
        #         if stage not in generation:
        #             generation[stage] = {}
        #         stage_dict = generation[stage]

        #         _, (period,) = c.pop(0)
        #         if period not in stage_dict:
        #             stage_dict[period] = {}
        #         period_dict = stage_dict[period]

        #         _, (commitment,) = c.pop(0)
        #         if commitment not in period_dict:
        #             period_dict[commitment] = {}
        #         commitment_dict = period_dict[commitment]

        #         _, (dispatch,) = c.pop(0)
        #         if dispatch not in commitment_dict:
        #             commitment_dict[dispatch] = dict.fromkeys(GEN_TYPES, 0)
        #         dispatch_dict = commitment_dict[dispatch]

        #         # dispatch_dict["battery_charge"] -= val
        #         storage_name = c[-1][0]
        #         validate_storage_name(storage_name)
        #         dispatch_dict["battery_charge"] -= val

        #     # Add battery discharging data to generation structure
        #     # Per request, plot discharge as negative (below x-axis)
        #     for g, val in discharging_data.items():
        #         c = list(pyo.ComponentUID(g)._cids)
        #         _, (stage,) = c.pop(0)
        #         if stage not in generation:
        #             generation[stage] = {}
        #         stage_dict = generation[stage]

        #         _, (period,) = c.pop(0)
        #         if period not in stage_dict:
        #             stage_dict[period] = {}
        #         period_dict = stage_dict[period]

        #         _, (commitment,) = c.pop(0)
        #         if commitment not in period_dict:
        #             period_dict[commitment] = {}
        #         commitment_dict = period_dict[commitment]

        #         _, (dispatch,) = c.pop(0)
        #         if dispatch not in commitment_dict:
        #             commitment_dict[dispatch] = dict.fromkeys(GEN_TYPES, 0)
        #         dispatch_dict = commitment_dict[dispatch]

        #         # dispatch_dict["battery_discharge"] += val
        #         storage_name = c[-1][0]
        #         validate_storage_name(storage_name)
        #         dispatch_dict["battery_discharge"] += val

        #     # print("\n[DEBUG] Storage summary from JSON inputs")

        #     total_charging = sum(charging_data.values())
        #     total_discharging = sum(discharging_data.values())

        #     # print(f"[DEBUG] Total charging (raw): {total_charging:,.3f}")
        #     # print(f"[DEBUG] Total discharging (raw): {total_discharging:,.3f}")

        #     charging_by_suffix = defaultdict(float)
        #     for g, val in charging_data.items():
        #         name = g.split(".")[-1]
        #         if name.endswith("_battery"):
        #             charging_by_suffix["battery"] += val
        #         elif name.endswith("_ps"):
        #             charging_by_suffix["ps"] += val
        #         else:
        #             charging_by_suffix["other"] += val

        #     discharging_by_suffix = defaultdict(float)
        #     for g, val in discharging_data.items():
        #         name = g.split(".")[-1]
        #         if name.endswith("_battery"):
        #             discharging_by_suffix["battery"] += val
        #         elif name.endswith("_ps"):
        #             discharging_by_suffix["ps"] += val
        #         else:
        #             discharging_by_suffix["other"] += val

        #     # print("[DEBUG] Charging by storage type suffix:")
        #     # for k, v in charging_by_suffix.items():
        #     #     print(f"    {k}: {v:,.3f}")

        #     # print("[DEBUG] Discharging by storage type suffix:")
        #     # for k, v in discharging_by_suffix.items():
        #     #     print(f"    {k}: {v:,.3f}")

        #     time_periods = [
        #         (s, p, c, d)
        #         for s in generation
        #         for p in generation[s]
        #         for c in generation[s][p]
        #         for d in generation[s][p][c]
        #     ]
        #     times = list(range(len(time_periods)))

        #     loads = {}
        #     for g, val in loads_data.items():
        #         c = list(pyo.ComponentUID(g)._cids)
        #         _, (stage,) = c.pop(0)
        #         if stage not in loads:
        #             loads[stage] = {}
        #         stage_dict = loads[stage]
        #         _, (period,) = c.pop(0)
        #         if period not in stage_dict:
        #             stage_dict[period] = {}
        #         period_dict = stage_dict[period]
        #         _, (commitment,) = c.pop(0)
        #         if commitment not in period_dict:
        #             period_dict[commitment] = 0
        #         period_dict[commitment] += val  # Sum all buses for this time period

        #     loads_trace = []
        #     for s, p, c, d in time_periods:
        #         try:
        #             total_load = loads[s][p][c]
        #         except KeyError:
        #             total_load = 0
        #         loads_trace.append(total_load)

        #     # Build load_shed dict: sum all buses for each (stage, period, commitment)
        #     load_shed = {}
        #     for g, val in load_shed_data.items():
        #         c = list(pyo.ComponentUID(g)._cids)
        #         _, (stage,) = c.pop(0)
        #         if stage not in load_shed:
        #             load_shed[stage] = {}
        #         stage_dict = load_shed[stage]
        #         _, (period,) = c.pop(0)
        #         if period not in stage_dict:
        #             stage_dict[period] = {}
        #         period_dict = stage_dict[period]
        #         _, (commitment,) = c.pop(0)
        #         if commitment not in period_dict:
        #             period_dict[commitment] = 0
        #         period_dict[commitment] += val  # Sum all buses for this time period

        #     # Build load_shed_trace to match time_periods (repeat for each dispatch)
        #     load_shed_trace = []
        #     for s, p, c, d in time_periods:
        #         try:
        #             total_shed = load_shed[s][p][c]
        #         except KeyError:
        #             total_shed = 0
        #         load_shed_trace.append(total_shed)
        #     # print(load_shed_trace)

        #     HATCH_TO_PATTERN = {
        #         "": "",  # solid fill
        #         "....": ".",  # dots
        #         "////": "/",  # slashes
        #         "xxxx": "x",  # crosshatch
        #     }

        #     def plotly_stackgraph(
        #         times,
        #         time_periods,
        #         generation,
        #         GEN_TYPES,
        #         GEN_TYPE_ALIASES,
        #         GEN_TYPE_HATCHES,
        #         HATCH_TO_PATTERN,
        #         results_path,
        #     ):
        #         """This function creates an interactive Plotly stackgraph
        #         for given representative days.  Each bar represents one
        #         hour in one representative day. The x-axis is labeled with
        #         the representative day and hour (shown at hour 0 and 12).

        #         """

        #         n_hours_per_day = 24
        #         n_rep_days = len(rep_days)
        #         n_points = n_hours_per_day * n_rep_days

        #         # Convert the rep_days strings to pandas Timestamps for
        #         # formatting
        #         rep_days_dt = [pd.to_datetime(d) for d in rep_days]

        #         # Build x-axis labels and tick positions: For each hour
        #         # in each representative day, create a label.  Only show
        #         # the label for hour 0 and hour 12 of each day, leave
        #         # others blank for clarity.
        #         x_labels = []
        #         tickvals = []
        #         ticktext = []
        #         for i, day in enumerate(rep_days_dt):
        #             for h in range(n_hours_per_day):
        #                 idx = i * n_hours_per_day + h  # Position in the x-axis
        #                 if h == 0:
        #                     label = day.strftime("%b-%d 00:00")
        #                     x_labels.append(label)
        #                     tickvals.append(idx)
        #                     ticktext.append(label)
        #                 elif h == 12:
        #                     label = day.strftime("%b-%d 12:00")
        #                     x_labels.append(label)
        #                     tickvals.append(idx)
        #                     ticktext.append(label)
        #                 else:
        #                     x_labels.append("")

        #         # The x-axis for the bars is just integer positions (0 to n_points-1)
        #         times = list(range(n_points))

        #         # Prepare traces for each generator type
        #         traces = []
        #         for name, color in GEN_TYPES.items():
        #             label = GEN_TYPE_ALIASES.get(name, name)
        #             # One value per hour, for all representative days
        #             values = np.array(
        #                 [generation[s][p][c][d][name] for s, p, c, d in time_periods]
        #             )
        #             hatch = GEN_TYPE_HATCHES.get(name, "")
        #             pattern_shape = HATCH_TO_PATTERN.get(hatch, "")
        #             # Use lower opacity for candidate types (those with a
        #             # hatch)
        #             opacity = 0.7 if hatch else 1.0

        #             traces.append(
        #                 go.Bar(
        #                     x=times,  # integer positions for each hour
        #                     y=values,
        #                     name=label,
        #                     marker_color=color,
        #                     marker_pattern_shape=pattern_shape,
        #                     opacity=opacity,
        #                     marker_line_width=0,  # remove white line
        #                 )
        #             )
        #         # Add load shed as a stacked bar
        #         tab20 = plt.get_cmap("tab20")
        #         traces.append(
        #             go.Bar(
        #                 x=times,
        #                 y=load_shed_trace,
        #                 name="Load Shed",
        #                 marker_color=mcolors.to_hex(tab20(7)),
        #                 opacity=0.7,
        #                 marker_line_width=0,
        #             )
        #         )
        #         fig = go.Figure(data=traces)
        #         fig.add_trace(
        #             go.Scatter(
        #                 x=times,
        #                 y=loads_trace,
        #                 mode="lines+markers",
        #                 name="Total Load",
        #                 line=dict(color="black", width=3, dash="dash"),
        #                 marker=dict(size=4, color="black"),
        #                 showlegend=True,
        #             )
        #         )
        #         # fig.add_trace(
        #         #     go.Scatter(
        #         #         x=times,
        #         #         y=load_shed_trace,
        #         #         mode="lines+markers",
        #         #         name="Load Shed",
        #         #         line=dict(color="red", width=3, dash="dot"),
        #         #         marker=dict(size=4, color="magenta"),
        #         #         showlegend=True,
        #         #     )
        #         # )
        #         fig.update_layout(
        #             barmode="relative",
        #             bargap=0,  # remove white spacing between bars
        #             title="Generation Mix (Representative Days)",
        #             xaxis=dict(
        #                 # title="Hours",
        #                 title="Representative Days (labeled every 12 hours)",
        #                 tickvals=tickvals,  # show ticks at hour 0 and 12 of each rep day
        #                 ticktext=ticktext,  # show corresponding label
        #                 showgrid=True,
        #                 gridcolor="gray",
        #                 gridwidth=0.7,
        #                 linecolor="black",
        #                 mirror=True,
        #             ),
        #             yaxis=dict(
        #                 title="Nameplate Capacity [MW]",
        #                 showgrid=True,
        #                 gridcolor="gray",
        #                 gridwidth=0.7,
        #                 linecolor="black",
        #                 mirror=True,
        #             ),
        #             legend=dict(
        #                 yanchor="middle",
        #                 y=0.5,
        #                 xanchor="left",
        #                 x=1.02,
        #                 font=dict(size=14),
        #                 title="Generation Type",
        #             ),
        #             width=1200,
        #             height=600,
        #             plot_bgcolor="white",
        #             paper_bgcolor="white",
        #         )

        #         # Add vertical lines to visually separate each
        #         # representative day
        #         for i in range(1, n_rep_days):
        #             fig.add_vline(
        #                 x=i * n_hours_per_day,
        #                 line=dict(color="gray", width=1, dash="dot"),
        #                 opacity=0.5,
        #             )

        #         # Add a little space above the tallest bar
        #         all_series = {
        #             name: np.array(
        #                 [generation[s][p][c][d][name] for s, p, c, d in time_periods]
        #             )
        #             for name in GEN_TYPES
        #         }

        #         positive_stack = np.sum(
        #             [np.clip(vals, 0, None) for vals in all_series.values()],
        #             axis=0,
        #         )
        #         negative_stack = np.sum(
        #             [np.clip(vals, None, 0) for vals in all_series.values()],
        #             axis=0,
        #         )

        #         ymin = negative_stack.min() if len(negative_stack) else 0
        #         ymax = positive_stack.max() if len(positive_stack) else 0

        #         if loads_trace:
        #             ymax = max(ymax, max(loads_trace))

        #         lower = ymin * 1.25 if ymin < 0 else -1
        #         upper = ymax * 1.25 if ymax > 0 else 1

        #         fig.update_yaxes(
        #             range=[lower, upper],
        #             zeroline=True,
        #             zerolinewidth=2,
        #             zerolinecolor="black",
        #         )

        #         # Save as interactive HTML
        #         plot_path = f"{results_path}/plots/stackgraph_generators_interactive.html"
        #         fig.write_html(f"{plot_path}")
        #         print(f" -> Saved interactive stackgraph to {plot_path}")

        #         # print("\n[DEBUG] First 10 plotted values by type:")
        #         # for name in ["battery_charge", "battery_discharge"]:
        #         #     values = np.array(
        #         #         [generation[s][p][c][d][name] for s, p, c, d in time_periods]
        #         #     )
        #         #     print(f"[DEBUG] {name}: {values[:10]}")

        def plot_generation_pie_chart(
            generation, time_periods, GEN_TYPES, GEN_TYPE_ALIASES, results_path
        ):
            """Plots a pie chart of total generation by unit type."""
            # Sum total generation for each type
            total_by_type = {name: 0.0 for name in GEN_TYPES}
            for s, p, c, d in time_periods:
                for name in GEN_TYPES:
                    total_by_type[name] += generation[s][p][c][d][name]

            # Filter out types with zero total
            total_by_type = {k: v for k, v in total_by_type.items() if abs(v) > 1e-6}

            labels = [
                f"{GEN_TYPE_ALIASES.get(name, name)} ({total_by_type[name]:,.1f} MW)"
                for name in total_by_type
            ]
            values = [total_by_type[name] for name in total_by_type]
            colors = [GEN_TYPES[name] for name in total_by_type]

            fig = go.Figure(
                data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))]
            )
            fig.update_layout(
                title="Total Generation Mix by Unit Type",
                legend=dict(font=dict(size=14)),
                width=700,
                height=500,
            )

            # Save as HTML
            plot_path = f"{results_path}/plots/generation_mix_pie_chart.html"
            fig.write_html(plot_path)
            print(f" -> Saved generation mix pie chart to {plot_path}")

        def calculate_metrics(folder_name):
            """
            Reads the generation.json file and calculates the total generation by generator type.

            :param folder_name: Directory containing the generation.json file.
            :return: Dictionary with total generation by type.
            """
            messages = []
            gen_types = [
                "cc_gas",
                "ct_gas",
                "coal",
                "nuclear",
                "thermal_other",
                "hydro",
                "solar",
                "wind",
                "battery_discharge",
                "steam",
                "dr",
                "ES4",
                "battery_charge",
                "hydro-c",
                "gas_cc-c",
                "gas_ct-c",
                "battery-c",
                "wind-c",
                "pv-c",
                "steam-c",
                "ES4-c",
            ]

            def mw_to_gwh(total_mw, hours_per_period=1):
                """
                Converts total MW (sum over all periods) to GWh.
                :param total_mw: Total MW (sum of MW for each period)
                :param hours_per_period: Duration of each period in hours (default 1)
                :return: Total GWh
                """
                total_mwh = total_mw * hours_per_period
                total_gwh = total_mwh / 1000
                return total_gwh

            def add_generation_metrics():
                """Add total generation and generation by generator
                type.

                Battery discharge is read from discharging.json and
                added to total generation. Battery charging is read
                from charging.json and reported separately, but it is
                not subtracted from total generation.

                """
                total_gen_by_type = defaultdict(float)
                file_path = os.path.join(folder_name, "generation.json")

                if not os.path.exists(file_path):
                    print(f"[WARNING] File not found: {file_path}")
                    return

                with open(file_path, "r") as f:
                    data = json.load(f)

                for key, value in data.items():
                    # The generator type is always at the end after
                    # the last '.'
                    gen_name = key.split(".")[-1]
                    for gen_type in gen_types:
                        if gen_name.endswith(gen_type):
                            total_gen_by_type[gen_type] += value
                            break

                # Add battery discharging to total generation. Battery
                # discharge is stored separately from generation in
                # discharging.json.
                total_battery_discharging = 0
                discharging_file_path = os.path.join(folder_name, "discharging.json")

                if os.path.exists(discharging_file_path):
                    with open(discharging_file_path, "r") as f:
                        discharging_data = json.load(f)

                    total_battery_discharging = sum(
                        value
                        for key, value in discharging_data.items()
                        if key.endswith("_battery") or key.endswith("_ps")
                    )

                    total_gen_by_type["battery_discharge"] += total_battery_discharging
                else:
                    print(f"[WARNING] File not found: {discharging_file_path}")

                # Report battery charging separately. It is not included in
                # total generation.
                total_battery_charging = 0
                charging_file_path = os.path.join(folder_name, "charging.json")

                if os.path.exists(charging_file_path):
                    with open(charging_file_path, "r") as f:
                        charging_data = json.load(f)

                    total_battery_charging = sum(
                        value
                        for key, value in charging_data.items()
                        if key.endswith("_battery") or key.endswith("_ps")
                    )
                else:
                    print(f"[WARNING] File not found: {charging_file_path}")

                # Total generation, including discharging (charging
                # not included)
                total_gen_all_types = sum(total_gen_by_type.values())
                total_gen_gwh = mw_to_gwh(total_gen_all_types, hours_per_period=1)

                messages.append(f"Total generation (GWh): {total_gen_gwh}")
                messages.append("Total generation by generator type:")

                for gen_type, total in total_gen_by_type.items():
                    total_per_type_gwh = mw_to_gwh(total, hours_per_period=1)
                    messages.append(f"  {gen_type}_(GWh): {total_per_type_gwh}")

                total_battery_discharging_gwh = mw_to_gwh(
                    total_battery_discharging, hours_per_period=1
                )
                total_battery_charging_gwh = mw_to_gwh(
                    total_battery_charging, hours_per_period=1
                )
                messages.append(
                    f"Total battery discharging (GWh): {total_battery_discharging_gwh}"
                )
                messages.append(
                    f"Total battery charging (GWh): {total_battery_charging_gwh}"
                )

            def add_curtailment_metrics():
                """Add total curtailment."""
                for curt_file in ["curtailment.json"]:
                    file_path = os.path.join(folder_name, curt_file)

                    if not os.path.exists(file_path):
                        continue

                    with open(file_path, "r") as f:
                        data = json.load(f)

                    total_curtailment = sum(float(value) for value in data.values())
                    total_curtailment_gwh = mw_to_gwh(
                        total_curtailment, hours_per_period=1
                    )

                    messages.append(f"Total curtailment (GWh): {total_curtailment_gwh}")
                    return

                print("[WARNING] No curtailment file found.")

            def add_generator_investment_metrics():
                """Add total number of newly installed generators and
                installed generators by type.

                """
                installed_gens = set()
                installed_gens_by_type = defaultdict(int)

                investment_files = [
                    "renewable_investments.json",
                    "dispatchable_investments.json",
                ]

                for investment_file in investment_files:
                    file_path = os.path.join(folder_name, investment_file)

                    if not os.path.exists(file_path):
                        print(f"[WARNING] File not found: {file_path}")
                        continue

                    with open(file_path, "r") as f:
                        data = json.load(f)

                    for key, value in data.items():
                        is_gen_installed = (
                            "renewableInstalled" in key or "genInstalled" in key
                        )

                        if not is_gen_installed or float(value) < 0.001:
                            continue

                        gen_name = key.split(".")[-1]

                        if gen_name in installed_gens:
                            continue

                        installed_gens.add(gen_name)

                        for gen_type in gen_types:
                            if gen_name.endswith(gen_type):
                                installed_gens_by_type[gen_type] += 1
                                break

                messages.append(
                    f"Number of installed generators: {len(installed_gens)}"
                )
                messages.append("Number of installed generators by type:")

                for gen_type, count in installed_gens_by_type.items():
                    messages.append(f"  installed_{gen_type}: {count}")

            def add_branch_investment_metrics():
                """Add total number of newly installed branches."""
                installed_branches = set()
                file_path = os.path.join(folder_name, "dispatchable_investments.json")

                if not os.path.exists(file_path):
                    print(f"[WARNING] File not found: {file_path}")
                    messages.append("Number of installed branches: 0")
                    return

                with open(file_path, "r") as f:
                    data = json.load(f)

                for key, value in data.items():
                    if "branchInstalled" in key and float(value) >= 0.001:
                        branch_name = key.split(".")[-1]
                        installed_branches.add(branch_name)

                messages.append(
                    f"Number of installed branches: {len(installed_branches)}"
                )

            def add_generator_cost_metrics():
                """Read costs.json and report total generator costs by
                type.

                Includes:
                - thermalGeneratorCost by generator type
                - hydroGeneratorCost by hydro type
                - renewableGeneratorCost by solar/wind type

                Representative-period weights are read from
                weights.json and used when summing costs. Costs are
                reported in billions of dollars (B$).

                """
                costs_file_path = os.path.join(folder_name, "costs.json")
                weights_file_path = os.path.join(folder_name, "weights.json")

                if not os.path.exists(costs_file_path):
                    print(f"[WARNING] File not found: {costs_file_path}")
                    return

                with open(costs_file_path, "r") as f:
                    data = json.load(f)

                if os.path.exists(weights_file_path):
                    with open(weights_file_path, "r") as f:
                        weights = json.load(f)
                else:
                    print(
                        f"[WARNING] File not found: {weights_file_path}. Using weight = 1."
                    )
                    weights = {}

                def get_rep_weight(key):
                    """Return representative-period weight for a cost key."""
                    match = re.search(r"representativePeriod\[(.*?)\]", key)
                    if match is None:
                        return 1

                    rep_per = match.group(1).strip("'\"")
                    return float(weights.get(rep_per, 1))

                def collect_costs_by_type(cost_key, type_list):
                    cost_by_type = defaultdict(float)

                    for key, value in data.items():
                        match = re.search(rf"{cost_key}\[(.*?)\]", key)
                        if match is None:
                            continue

                        gen_name = match.group(1).strip("'\"")
                        weighted_value = float(value) * get_rep_weight(key)

                        for gen_type in type_list:
                            if gen_name.endswith(gen_type):
                                cost_by_type[gen_type] += weighted_value
                                break
                        else:
                            cost_by_type["other"] += weighted_value

                    return cost_by_type

                def add_cost_messages(label, cost_by_type):
                    total_cost = sum(cost_by_type.values())
                    total_cost_billions = total_cost / 1e9

                    messages.append(f"Total {label} cost (B$): {total_cost_billions}")
                    messages.append(f"Total {label} cost by type:")

                    for gen_type, total_cost in cost_by_type.items():
                        total_cost_billions = total_cost / 1e9
                        messages.append(
                            f"  {gen_type}_{label}_cost_(B$): {total_cost_billions}"
                        )

                thermal_cost_by_type = collect_costs_by_type(
                    "thermalGeneratorCost",
                    gen_types,
                )

                hydro_cost_by_type = collect_costs_by_type(
                    "hydroGeneratorCost",
                    ["hydro", "hydro-c"],
                )

                renewable_cost_by_type = collect_costs_by_type(
                    "renewableGeneratorCost",
                    ["solar", "wind", "pv-c", "wind-c"],
                )

                add_cost_messages("thermal_generator", thermal_cost_by_type)
                add_cost_messages("hydro_generator", hydro_cost_by_type)
                add_cost_messages("renewable_generator", renewable_cost_by_type)

            def add_fixed_operating_cost_metrics():
                """Read costs.json and report total fixed operating cost.

                Uses operatingCostCommitment entries as the fixed operating cost
                proxy and applies representative-period weights from weights.json.
                Costs are reported in billions of dollars (B$).
                """
                costs_file_path = os.path.join(folder_name, "costs.json")
                weights_file_path = os.path.join(folder_name, "weights.json")

                if not os.path.exists(costs_file_path):
                    print(f"[WARNING] File not found: {costs_file_path}")
                    return

                with open(costs_file_path, "r") as f:
                    data = json.load(f)

                if os.path.exists(weights_file_path):
                    with open(weights_file_path, "r") as f:
                        weights = json.load(f)
                else:
                    print(
                        f"[WARNING] File not found: {weights_file_path}. Using weight = 1."
                    )
                    weights = {}

                def get_rep_weight(key):
                    match = re.search(r"representativePeriod\[(.*?)\]", key)
                    if match is None:
                        return 1

                    rep_per = match.group(1).strip("'\"")
                    return float(weights.get(rep_per, 1))

                total_fixed_operating_cost = 0
                for key, value in data.items():
                    if not key.endswith("operatingCostCommitment"):
                        continue

                    total_fixed_operating_cost += float(value) * get_rep_weight(key)

                total_fixed_operating_cost_billions = total_fixed_operating_cost / 1e9

                messages.append(
                    f"Total fixed operating cost fom (B$): {total_fixed_operating_cost_billions}"
                )

            add_generation_metrics()
            add_curtailment_metrics()
            add_generator_investment_metrics()
            add_branch_investment_metrics()
            add_generator_cost_metrics()
            add_fixed_operating_cost_metrics()

            return messages

        def print_generation_for_days(
            rep_days, time_periods, loads_trace, generation, day_hour_list
        ):
            """Prints load and generation by type for multiple (day,
                hour) pairs.

            :param rep_days: List of representative day strings (e.g., "2034-07-12 00:00")
            :param time_periods: List of time period tuples (s, p, c, d)
            :param loads_trace: List of load values per time period
            :param generation: Nested dict of generation by time period and type
            :param day_hour_list: List of (target_day, target_hour) tuples

            """
            messages = []
            for target_day, target_hour in day_hour_list:
                # Find the index for desired day and time
                try:
                    day_idx = rep_days.index(target_day)
                except ValueError:
                    raise ValueError(f"Target day {target_day} not found in rep_days!")

                target_idx = day_idx * 24 + target_hour
                target_time_period = time_periods[target_idx]

                messages.append(
                    f"Results for representative day: {target_day}, hour: {target_hour} (index {target_idx})"
                )

                # Total load
                total_load_gw = loads_trace[target_idx] / 1000
                messages.append(f"Total load (GW): {total_load_gw:.2f}")

                # Generation per type
                messages.append("Generation by type:")
                for gen_type in [
                    "cc_gas",
                    "ct_gas",
                    "coal",
                    "nuclear",
                    "thermal_other",
                    "hydro",
                    "solar",
                    "wind",
                    "battery_discharge",
                    "steam",
                    "dr",
                    "ES4",
                    "battery_charge",
                    "hydro-c",
                    "gas_cc-c",
                    "gas_ct-c",
                    "battery-c",
                    "wind-c",
                    "pv-c",
                    "steam-c",
                    "ES4-c",
                ]:
                    value = generation[target_time_period[0]][target_time_period[1]][
                        target_time_period[2]
                    ][target_time_period[3]].get(gen_type, 0)
                    value_gw = value / 1000
                    messages.append(f"  {gen_type} (GW): {value_gw:.2f}")

            return messages

        def save_metrics_and_generation_to_csv(
            results_path,
            rep_days,
            time_periods,
            loads_trace,
            generation,
            day_hour_list,
            output_csv_file,
        ):

            def split_message(message):
                if ":" in message:
                    parts = message.rsplit(":", 1)
                    label = parts[0].strip()
                    value = parts[1].strip()
                    return label, value
                else:
                    return message, ""  # No value, just text

            metrics_lines = calculate_metrics(results_path)
            gen_lines = print_generation_for_days(
                rep_days, time_periods, loads_trace, generation, day_hour_list
            )

            with open(output_csv_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Description", "Value"])
                for line in metrics_lines:
                    label, value = split_message(line)
                    writer.writerow([label, value])
                for line in gen_lines:
                    label, value = split_message(line)
                    writer.writerow([label, value])

            print(f" -> Saved metrics and generation output to {output_csv_file}")

        plot_generation_pie_chart(
            generation, time_periods, GEN_TYPES, GEN_TYPE_ALIASES, results_path
        )
        plotly_stackgraph(
            times,
            time_periods,
            generation,
            GEN_TYPES,
            GEN_TYPE_ALIASES,
            GEN_TYPE_HATCHES,
            HATCH_TO_PATTERN,
            results_path,
        )

        calculate_metrics(results_path)
        print_generation_for_days(
            rep_days=rep_days,
            time_periods=time_periods,
            loads_trace=loads_trace,
            generation=generation,
            day_hour_list=day_hour_list,
        )

        save_metrics_and_generation_to_csv(
            results_path,
            rep_days,
            time_periods,
            loads_trace,
            generation,
            day_hour_list,
            f"{results_path}/metrics_and_generation_output.csv",
        )

    def create_html_report(self, results_path, plot_type):

        def html_results_tab(total_cost):
            return f"""
            <div id="Results" class="tabcontent">
            <h2>Results</h2>
            <table>
            <tr><th>Total Cost</th></tr>
            <tr><td>{total_cost}</td></tr>
            </table>
            </div>
            """

        def html_plots_tab(plot_files, plot_type):
            html = """
            <div id="Plots" class="tabcontent">
            <h2>Plots</h2>
            """

            plot_map = {
                "stackplot": ("gen_mix_summary_interactive.html", "Stack Plot"),
                "treemap": ("treemap_2034_interactive.html", "Treemap"),
                "piechart": ("pie_leader_2034_interactive.html", "Pie Chart"),
            }
            categories = ["Dispatchables", "Renewables"]

            for cat in categories:
                cat_lower = cat.lower()
                html += f"<h3>{cat}</h3><ul>\n"
                if plot_type == "all":
                    for _, (fname, label) in plot_map.items():
                        html += f'<li><a href="plots/{cat_lower}_{fname}" target="_blank">{label} ({cat})</a></li>\n'
                elif plot_type in plot_map:
                    fname, label = plot_map[plot_type]
                    html += f'<li><a href="plots/{cat_lower}_{fname}" target="_blank">{label} ({cat})</a></li>\n'
                else:
                    html += "<li>Plot type not supported</li>\n"
                html += "</ul>\n"

            # Add Stackgraph section and plot
            html += "<h3>Stackgraph</h3><ul>\n"
            html += '<li><a href="plots/stackgraph_generators_interactive.html" target="_blank">Stackgraph</a></li>\n'
            html += "</ul>\n"

            html += "</div>"
            return html

        # Read total cost from costs.json
        costs_file = f"{results_path}/costs.json"
        try:
            with open(costs_file, "r") as f:
                costs = json.load(f)
            total_cost = None
            for k in costs:
                if "total" in k.lower():
                    total_cost = costs[k]
                    break
            if total_cost is None and costs:
                total_cost = list(costs.values())[0]
        except Exception as e:
            print(f"[WARNING] Could not read total cost from {costs_file}: {e}")
            total_cost = "N/A"

        # Manually specify plot files
        plot_files = [
            ("Stack Plot (Dispatchables)", "plots/dispatchables_gen_mix_summary.png"),
            ("Stack Plot (Renewables)", "plots/renewables_gen_mix_summary.png"),
            ("Treemap (Dispatchables)", "plots/dispatchables_treemap_2034.png"),
            ("Treemap (Renewables)", "plots/renewables_treemap_2034.png"),
            ("Pie Chart (Dispatchables)", "plots/dispatchables_pie_leader_2034.png"),
            ("Pie Chart (Renewables)", "plots/renewables_pie_leader_2034.png"),
            ("Stackgraph", "plots/stackgraph_generators.png"),
        ]

        html = f"""
        <html>
        <head>
        <title>Expansion Planning Report</title>
        <style>
        body {{ font-family: Arial, sans-serif; }}
        .tab {{
        overflow: hidden;
        border-bottom: 1px solid #ccc;
        background-color: #f1f1f1;
        }}
        .tab button {{
        background-color: inherit;
        float: left;
        border: none;
        outline: none;
        cursor: pointer;
        padding: 14px 16px;
        transition: 0.3s;
        font-size: 17px;
        }}
        .tab button:hover {{ background-color: #ddd; }}
        .tab button.active {{ background-color: #ccc; }}
        .tabcontent {{
        display: none;
        padding: 20px;
        border: 1px solid #ccc;
        border-top: none;
        }}
        table, th, td {{
        border: 1px solid #ccc;
        border-collapse: collapse;
        padding: 8px;
        }}
        </style>
        <script>
        function openTab(evt, tabName) {{
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) {{
        tabcontent[i].style.display = "none";
        }}
        tablinks = document.getElementsByClassName("tablinks");
        for (i = 0; i < tablinks.length; i++) {{
        tablinks[i].className = tablinks[i].className.replace(" active", "");
        }}
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
        }}
        </script>
        </head>
        <body>
        <h1>Expansion Planning Report</h1>
        <div class="tab">
        <button class="tablinks" onclick="openTab(event, 'Results')" id="defaultOpen">Results</button>
        <button class="tablinks" onclick="openTab(event, 'Plots')">Plots</button>
        </div>
        {html_results_tab(total_cost)}
        {html_plots_tab(plot_files, plot_type)}
        <script>
        document.getElementById("defaultOpen").click();
        </script>
        </body>
        </html>
        """

        html_file = os.path.join(results_path, "gtep_report.html")
        with open(html_file, "w") as f:
            f.write(html)
        print(f" HTML report written to {html_file}")
