"""Compare two exact external model builds in a static Double Lift report.

Normally run this script unchanged. The YAML names the two build configs,
the SAMPLE populations, chart controls, and output folder. Lucidum is used
only as a chart library: no app or server is started.
"""

# %% Imports

from py_lucidum import double_lift_chart, report_filename, write_echarts_report

from external_report_helpers import (
    config_path_from_command_line,
    double_lift_report_header,
    load_double_lift_settings,
)


# %% 1. Load the comparison YAML and its two exact build configs

script_file = globals().get("__file__")
config_path = config_path_from_command_line(script_file, "config_double_lift.yaml")
settings = load_double_lift_settings(config_path)


# %% 2. Make one Double Lift chart for each selected SAMPLE population

prepared_reports = []

for report in settings["reports"]:
    chart = double_lift_chart(
        settings["dataset_path"],
        actual=settings["actual"],
        denominator=settings["denominator"],
        baseline_model_type=settings["baseline"]["model_type"],
        baseline_model_id=settings["baseline"]["model_id"],
        baseline_model_folder=settings["baseline"]["model_folder"],
        challenger_model_type=settings["challenger"]["model_type"],
        challenger_model_id=settings["challenger"]["model_id"],
        challenger_model_folder=settings["challenger"]["model_folder"],
        sample_column=settings["sample_column"],
        sample_values=report["sample_values"],
        controls=settings["chart"],
        kpi_spec=settings["kpi_spec_path"],
        title=report["title"],
    )
    prepared_reports.append({"settings": report, "chart": chart})


# %% 3. Write one self-contained HTML file per SAMPLE population

for prepared_report in prepared_reports:
    report = prepared_report["settings"]
    chart = prepared_report["chart"]
    output_name = report_filename(
        settings["dataset_path"],
        "double_lift",
        report["name"],
    )
    output_path = settings["output_directory"] / output_name
    write_echarts_report(
        [chart],
        output_path,
        title=report["title"],
        metadata=double_lift_report_header(
            settings,
            report,
            chart,
            script_file or "04_external_double_lift_demo.py",
        ),
        chart_height=settings["chart_height"],
    )
    print(f"Double Lift report: {output_path}")
