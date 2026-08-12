"""Tabulate an external GLM and create one static model-summary report.

Normally run this script unchanged.  The YAML chooses the 01 model build,
Feature Specification, KPI Specification, report title, and output folder.
Lucidum supplies the same tabulation, scoring, and XLSX export used by its app;
the app itself is not started.
"""

# %% Imports

from py_lucidum import (
    build_glm_tabulations,
    export_glm_tabulations,
    report_filename,
    write_glm_summary_report,
)

from external_report_helpers import (
    config_path_from_command_line,
    glm_summary_header,
    load_glm_summary_settings,
)


# %% 1. Load the YAML settings

config_path = config_path_from_command_line(__file__, "config_glm_summary_report.yaml")
settings = load_glm_summary_settings(config_path)


# %% 2. Build the rating tables and score the source data

tabulations = build_glm_tabulations(
    settings["dataset_path"],
    model_id=settings["model_id"],
    feature_spec_path=settings["feature_spec_path"],
    model_folder=settings["model_folder"],
)


# %% 3. Export the rating tables to Excel

workbook = export_glm_tabulations(
    settings["dataset_path"],
    model_id=settings["model_id"],
    scale="auto",  # exp for a fitted log link; linear for every other link
    model_folder=settings["model_folder"],
)

print(f"Tabulated scores: {tabulations['scoring_path']}")
print(f"GLM tabulations: {workbook['path']}")


# %% 4. Write one self-contained HTML file

output_name = report_filename(
    settings["dataset_path"],
    "glm",
    settings["report_name"],
)
output_path = settings["output_directory"] / output_name

write_glm_summary_report(
    output_path,
    title=settings["report_title"],
    dataset_path=settings["dataset_path"],
    model_id=settings["model_id"],
    kpi_spec_path=settings["kpi_spec_path"],
    tabulation_export=workbook,
    model_folder=settings["model_folder"],
    metadata=glm_summary_header(settings, __file__),
)

print(f"GLM summary report: {output_path}")


# %% 5. Optionally synchronize the updated model with Lucidum

if settings["install_in_lucidum"]:
    from lucidum_install import install_model_in_lucidum

    lucidum_model_folder = install_model_in_lucidum(
        dataset_path=settings["dataset_path"],
        model_folder=settings["model_folder"],
        model_type="glm",
        model_id=settings["model_id"],
        replace_existing=settings["replace_existing"],
    )
    print(f"Lucidum model folder: {lucidum_model_folder}")
