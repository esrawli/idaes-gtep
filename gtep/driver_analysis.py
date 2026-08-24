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
import logging

from gtep.gtep_analysis import ExpansionPlanningAnalysis
from gtep.gtep_solution import ExpansionPlanningSolution


logger = logging.getLogger("gtep.driver_analysis")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Input data and results settings
# ---------------------------------------------------------------------

rep_days = [
    "2030-01-01 00:00",
    "2030-04-01 00:00",
    "2030-07-12 00:00",
    "2030-10-01 00:00",
]

rep_weights = [90, 90, 95, 90]

data_path = "data/base_case_pcm_2030"
dir_name = "this_is_too_big"


# ---------------------------------------------------------------------
# Create solution object and plots
# ---------------------------------------------------------------------

sol_object = ExpansionPlanningSolution(data_path)

# Define the plot type for generation-mix plots.
# Options are "treemap", "piechart", or "all".
plot_type = "all"

# Create generation-mix plots for dispatchable, renewable, and
# combined generation.
case_json = "dispatchables"
sol_object.create_plots(case_json, dir_name, data_path, plot_type)

case_json = "renewables"
sol_object.create_plots(case_json, dir_name, data_path, plot_type)

case_json = "combined"
sol_object.create_plots(case_json, dir_name, data_path, plot_type)

# Create dispatch stackgraph.
day_hour_list = [("2030-07-12 00:00", 18)]
sol_object.create_stackgraph_and_metrics(dir_name, rep_days, day_hour_list)


# ---------------------------------------------------------------------
# Create analysis metrics from input data and saved result files
# ---------------------------------------------------------------------

analysis = ExpansionPlanningAnalysis(
    data_path,
    print_results=True,
    save_csv=True,
    csv_path=os.path.join(dir_name, "analysis_metrics.csv"),
)

analysis.run_all_metrics(
    dir_name,
    hours_per_period=1.0,
)
