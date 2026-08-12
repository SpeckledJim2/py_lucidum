"""Create one static GBM model-summary HTML report.

Normally run this script unchanged.  The YAML points to the 01 build, KPI
Specification, and output folder.  Lucidum supplies the tested evaluation
chart and report layout; no app or server is started.
"""

# %% Imports

from py_lucidum import gbm_evaluation_chart, report_filename, write_gbm_summary_report

from external_report_helpers import (
    config_path_from_command_line,
    gbm_summary_header,
    load_gbm_summary_settings,
)


# %% 1. Load the YAML settings and saved model results

config_path = config_path_from_command_line(__file__, "config_gbm_summary_report.yaml")
settings, performance, feature_importance, parameters = load_gbm_summary_settings(config_path)


# %% 2. Create the model evaluation chart

evaluation_chart = gbm_evaluation_chart(
    settings["dataset_path"],
    model_id=settings["model_id"],
    title="Model evaluation chart",
)


# %% 3. Write one self-contained HTML file

output_name = report_filename(
    settings["dataset_path"],
    "gbm",
    settings["report_name"],
)
output_path = settings["output_directory"] / output_name

write_gbm_summary_report(
    output_path,
    title=settings["report_title"],
    metadata=gbm_summary_header(settings, __file__),
    performance=performance,
    feature_importance=feature_importance,
    parameters=parameters,
    evaluation_chart=evaluation_chart,
    chart_height=settings["chart_height"],
)

print(f"GBM summary report: {output_path}")
