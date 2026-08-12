"""Create static GBM Actual-vs-Expected and rebased-SHAP HTML reports.

Normally run this script unchanged.  The YAML chooses the model, rows, feature
scenario, chart controls, and output folder.  Lucidum is used only as a chart
library: no app or server is started.
"""

# %% Imports

from py_lucidum import line_bar_chart, report_filename, write_echarts_report

from external_report_helpers import (
    config_path_from_command_line,
    features_for_report,
    load_report_settings,
    report_header,
)


# %% 1. Load the YAML settings and the selected feature-spec rows

config_path = config_path_from_command_line(__file__, "config_gbm_report.yaml")
settings, report_features = load_report_settings(config_path, "gbm")


# %% 2. Make one Lucidum-format chart for each selected feature

prepared_reports = []

for report in settings["reports"]:
    charts = []

    for feature in features_for_report(report_features, report):
        chart = line_bar_chart(
            settings["dataset_path"],
            x=feature["name"],
            actual=settings["actual"],
            expected=settings["expected"],
            expected_source=settings["expected_source"],
            expected_label=settings["expected_label"],
            denominator=settings["denominator"],
            sample_column=settings["sample_column"],
            sample_values=report["sample_values"],
            model_id=settings["model_id"],
            controls={**feature["controls"], "sigma": report["sigma"]},
            content=report["chart_content"],
            transform=report["transform"],
            partial_dependence=report["partial_dependence"],
            feature_spec=settings["feature_spec_path"],
            title=feature["title"],
        )
        charts.append(chart)

    prepared_reports.append({"settings": report, "charts": charts})


# %% 3. Write one self-contained HTML file

for prepared_report in prepared_reports:
    report = prepared_report["settings"]
    output_name = report_filename(
        settings["dataset_path"],
        settings["model_type"],
        report["name"],
    )
    output_path = settings["output_directory"] / output_name
    write_echarts_report(
        prepared_report["charts"],
        output_path,
        title=report["title"],
        metadata=report_header(settings, report, __file__),
        chart_height=settings["chart_height"],
    )
    print(f"GBM report: {output_path}")
