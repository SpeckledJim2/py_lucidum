# Lucidum User Guide

The [README](../README.md) is the product overview and covers the recommended
installation method and how to launch the included demo. This guide is the
operational reference for alternative setup, launch options, controls, limits,
and saved work.

## Contents

### Guide contents

| Area | Sections |
| --- | --- |
| Getting started | [Installation alternatives and launch options](#installation-alternatives-and-launch-options) · [Datasets and access](#datasets-and-access) |
| Shared analysis controls | [Metrics and shared controls](#metrics-and-shared-controls) |
| Saved work | [Saved work and model workspaces](#saved-work-and-model-workspaces) |
| Python and external workflows | [Python usage](#python-usage) · [External models and reports](#external-models-and-reports) |
| Help | [Monitoring and troubleshooting](#monitoring-and-troubleshooting) |

### Tools

| Tool | Tool |
| --- | --- |
| <img src="https://github.com/SpeckledJim2/py_lucidum/raw/main/docs/assets/tool-icons/dataset-viewer.svg" width="24" height="24" align="absmiddle" alt=""> [Dataset Viewer](#dataset-viewer) | <img src="https://github.com/SpeckledJim2/py_lucidum/raw/main/docs/assets/tool-icons/specifications.svg" width="24" height="24" align="absmiddle" alt=""> [Specifications](#specifications) |
| <img src="https://github.com/SpeckledJim2/py_lucidum/raw/main/docs/assets/tool-icons/column-profile.svg" width="24" height="24" align="absmiddle" alt=""> [Column Profile](#column-profile) | <img src="https://github.com/SpeckledJim2/py_lucidum/raw/main/docs/assets/tool-icons/line-and-bar.svg" width="24" height="24" align="absmiddle" alt=""> [Line and Bar](#line-and-bar) |
| <img src="https://github.com/SpeckledJim2/py_lucidum/raw/main/docs/assets/tool-icons/histogram.svg" width="24" height="24" align="absmiddle" alt=""> [Histogram](#histogram) | <img src="https://github.com/SpeckledJim2/py_lucidum/raw/main/docs/assets/tool-icons/uk-mapping.svg" width="24" height="24" align="absmiddle" alt=""> [UK Mapping](#uk-mapping) |
| <img src="https://github.com/SpeckledJim2/py_lucidum/raw/main/docs/assets/tool-icons/glm.svg" width="24" height="24" align="absmiddle" alt=""> [GLM](#glm) | <img src="https://github.com/SpeckledJim2/py_lucidum/raw/main/docs/assets/tool-icons/gbm.svg" width="24" height="24" align="absmiddle" alt=""> [GBM](#gbm) |

## Installation alternatives and launch options

Use the [README installation](../README.md#install-and-launch) for the recommended
`pipx` setup and optional modelling extras. Lucidum requires Python 3.13 or newer;
the package name is `py-lucidum`, the import is `py_lucidum`, and the installed
command is `lucidum`.

### Maintain a pipx installation

Upgrade the installed package with `pipx upgrade py-lucidum`. To change optional
extras, reinstall with the required specification:

```bash
pipx uninstall py-lucidum
pipx install --python python3.13 "py-lucidum[glm,gbm]"
```

Pin a known version when repeatability matters:

```bash
pipx install --python python3.13 "py-lucidum[glm,gbm]==0.5.13"
```

The pin fixes Lucidum, not its transitive dependencies. Use a lock or constraints
file when the entire Python environment must be fixed. Quote specifications that
contain extras because some shells interpret the brackets.

### Install with pip

Plain `pip` installs into the currently selected Python environment. On macOS or
Linux:

```bash
python3.13 -m venv lucidum-venv
source lucidum-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "py-lucidum[glm,gbm]"
```

On Windows PowerShell:

```powershell
py -3.13 -m venv lucidum-venv
.\lucidum-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "py-lucidum[glm,gbm]"
```

Omit the extras when you only need the exploration tools. Use the same package
specification in another Python project's dependency metadata or requirements file.
Do not use `sudo pip` for Lucidum.

### Install from a source checkout

Create an editable development environment from the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -e ".[glm,gbm]"
```

For a `pipx` command backed by a local checkout:

```bash
pipx install --python python3.13 /path/to/py_lucidum
pipx install --python python3.13 "/path/to/py_lucidum[glm,gbm]"
```

If PyPI is unavailable, an immutable release can also be installed from its Git tag:

```bash
pipx install --python python3.13 \
  "py-lucidum @ git+https://github.com/SpeckledJim2/py_lucidum.git@v0.5.13"
```

### LightGBM on macOS

LightGBM may require the OpenMP runtime. If GBM training reports a
`libomp.dylib` load error, install it with:

```bash
brew install libomp
```

### Launch Lucidum

The README covers the first demo and file launch. The general command forms are:

```bash
lucidum --demo --open
lucidum DATASET --open
```

`DATASET` can be a CSV, a Parquet file, or a same-schema Parquet folder. From a
source checkout, use `.venv/bin/lucidum` if the environment is not active. The
committed multi-file fixture can be launched with:

```bash
.venv/bin/lucidum datasets/monthly --port 8000
```

Without `--open`, the command prints the browser URL. Stop a terminal launch with
`Ctrl+C`.

### Choose tools

Without `--tools`, Lucidum loads Line and Bar, Dataset Viewer, Column Profile,
Histogram, UK Mapping, and Specifications. Line and Bar opens first unless a saved
startup favourite applies.

Use a comma-separated list to choose both the enabled tabs and their order:

```bash
lucidum path/to/my_data.parquet --tools line-bar,dataset-viewer,histogram
lucidum path/to/my_data.parquet --tools line-bar,uk-map,glm,gbm
lucidum path/to/my_data.parquet --tools all
```

GLM and GBM each require Line and Bar because model actions and shared outputs open
there. When one tool is enabled, the sidebar tool selector is hidden. With several
tools, the vertical tool rail remains available when the rest of the sidebar is
collapsed. Enabled GLM and GBM icons show the number of saved models, including
zero.

### Useful launch options

```bash
lucidum --demo --host 127.0.0.1 --port 8000 --open
lucidum --demo --buttons
lucidum --demo --title-prefix "Motor pricing data"
lucidum --demo --x DRIVER_AGE --actual PREMIUM --denominator ANNUAL_MILEAGE
lucidum --demo --filters specs/filter_spec.csv
lucidum --demo --kpis specs/kpi_spec.csv
lucidum --demo --features specs/feature_spec.csv
```

- `--host` and `--port` choose the listening address. If the port is omitted,
  Lucidum selects a free local port.
- `--open` opens the generated URL.
- `--buttons` adds **Stop app** and **Open monitor** to the app header.
- `--title-prefix` adds text before the file or folder name in the header. Demo mode
  defaults it to `Lucidum Demo Dataset`; pass an empty value to suppress it.
- `--x`, `--actual`, `--expected`, `--expected2`, and `--denominator` set initial
  Line and Bar choices.
- `--line-bar-favourite` opens a favourite by name or ID. The
  `line_bar_favourite` URL parameter can override the CLI or Python default.
- `--line-bar-favourites` sets the server-side JSON path used to store favourites.
- `--filters`, `--kpis`, and `--features` set specification CSV paths.
- `--no-filters`, `--no-kpis`, and `--no-features` disable automatic discovery for
  the matching specification.
- `--postcode-area`, `--postcode-sector`, `--postcode-unit`, `--latitude`, and
  `--longitude` override the columns used by UK Mapping.
- `--no-token` removes URL and API token protection. Use it only when an alternative
  access control is present or the app is strictly local.

Run `lucidum --help` for the complete option list and `lucidum --version` for the
installed version. The version is also shown in the app sidebar footer.

[Back to contents ↑](#contents)

## Datasets and access

### Supported inputs

CSV is supported for convenience. Parquet is recommended because DuckDB can query
it efficiently without first importing it into another format.

A folder input is treated as one dataset when every direct-child `.parquet` file has
identical column names and DuckDB types. Files are read in sorted order. Non-Parquet
files and nested folders are ignored. The header identifies the folder and file
count.

Folder inputs are for non-modelling tools only. GLM, GBM, and `--tools all` require a
single CSV or Parquet file because saved predictions, sample splits, SHAP values, and
model workspaces must be attached to one source file.

The input is treated as fixed while the app is running. Replace or edit a source file
only after stopping Lucidum, then relaunch against the new version.

Unreadable columns, such as Parquet strings containing invalid UTF-8, are excluded
from normal selectors. Column Profile reports them as skipped, and GBM displays them
as disabled invalid features.

### Local and network access

When you run Lucidum and open it on the same machine, the default setup is normally
all you need. If you host Lucidum on another machine or make it available across a
network, take the same precautions you would with any service that can access your
data: limit who can connect, keep the access token enabled, and follow your
organisation's security guidance.

The app code, chart libraries, and UK postcode geometry are bundled. UK Mapping's
**Blank** base makes no external tile requests. Esri, OSM, Aerial, Light, and Dark
bases fetch map tiles from the listed providers and require network access.

[Back to contents ↑](#contents)

## Metrics and shared controls

### Numerator and Denominator

The two shared metric controls are labelled **Numerator** and **Denominator**.
Internally and in saved specifications, some fields retain the historical names
Actual and Weight.

- **Average row value** analyses the Numerator directly.
- A numeric dataset Denominator analyses `Numerator / Denominator` and uses the
  denominator as volume or exposure where appropriate.
- An active GLM or GBM prediction can also be selected as the Denominator.
- Chart axes name the underlying calculation, such as
  `PREMIUM / glm_prediction`.

The sidebar summaries always follow the global filter. Tool-specific missing-value
choices can change a Line and Bar result without changing the meaning of the shared
Numerator and Denominator summaries.

Activating a saved model selects the Numerator and Denominator that were used to
train it, then exposes that family's primary prediction as the sole Line and Bar
Expected value. A compatible GLM/GBM prediction pair can remain selected when both
models use the unchanged metric pair.

Manually selected model comparisons remain visible when their training metrics no
longer match the current KPI. Line and Bar shows a compatibility warning beside the
affected prediction, GLM partial-dependence overlay, or GBM SHAP overlay. Restore the
saved metric pair or remove the component to clear the warning.

Building a new GLM or GBM is disabled while a model prediction is the Denominator.
GBM prediction chaining remains available through `init_score`.

### Filters

The footer filter box accepts DuckDB `WHERE` expressions:

```sql
DRIVER_AGE > 40
ANNUAL_MILEAGE >= 20000
VEHICLE_USAGE = 'Social only'
QUOTE_DATE >= DATE '2017-01-01'
```

Filters apply to the active analysis and are shared when switching tools. Saved
filter rows come from the Filter Specification and can be applied in **Single**,
**Multi**, or **Grouped** mode. The resulting expression remains visible and
editable in the footer, so named cohort or data-quality definitions do not hide the
SQL being applied.

### KPIs

KPI rows are read-only presets for Numerator, Denominator, decimal places, and
number, currency, or percentage formatting. They do not change the active filter or
tool view. Use a favourite when those choices must be restored together.

### Favourites

Favourites save one of these scopes:

- Metrics.
- Metrics plus filter.
- Line/Bar view.
- Dataset view.
- Histogram view.
- Map view.

The available view scopes depend on the active tool. The first saved favourite opens
automatically at startup when no explicit startup favourite is supplied. A named or
ID-based startup favourite takes precedence.

Lucidum validates saved columns, model sources, filter expressions, KPI rows, and
saved-filter rows against the current dataset before restoring them. It reports stale
fields and restores the valid remainder where possible.

[Back to contents ↑](#contents)

## Dataset Viewer

Apply any global filter, then open Dataset Viewer to inspect and copy the matching
source rows. The browser receives a preview rather than the complete dataset.

### Table controls

- Column headers can sort the preview. Search operates across the displayed table.
- Selection is axis-exclusive: select whole rows or whole columns before copying the
  result as CSV.
- Right-click a cell to copy it directly.
- **Transpose** turns original column names into rows. Search still matches those
  names while all preview-row columns remain available.
- **A–Z columns** alphabetises the dataset fields.
- The column chooser can search and limit the displayed fields.
- Right-click a column to pin it. Pinned names are listed above the table and remain
  first and visible while searching columns; pinning does not freeze the cell on
  screen.
- Column widths and sort state can be adjusted for the current view.

### Saved view and limits

The displayed preview is capped at 100 rows. Filters and analytical tools continue
to use the underlying dataset rather than only those rows.

A Dataset view favourite saves the filter, transpose state, alphabetical ordering,
column search, pinned columns, resized widths, and sort state. When a saved sort
column is no longer present, Lucidum reports it rather than failing the complete
restore.

[Back to contents ↑](#contents)

## Column Profile

Column Profile provides a summary and detail view for each readable dataset field.

### Summary and detail

- Physical and logical data types. Boolean fields are labelled `logical` and retain
  categorical analysis behaviour.
- Missing and non-missing row counts.
- Approximate or exact distinct counts, depending on the current calculation.
- Numeric and date minima, maxima, and distributions.
- Categorical value counts and common values.

Open one column for its detailed profile, or right-click its table row to copy the
feature name.

### Preview and exact calculations

Large datasets use a preview summary for a responsive first view. Choose the all-row
calculation when exact results are needed.

Columns that cannot be decoded safely are listed as skipped so users can fix the
source data without losing access to the other fields.

[Back to contents ↑](#contents)

## Line and Bar

Set the Numerator and Denominator in the sidebar, then use Line and Bar to group the
result over one or two selected features and add comparison series where required.

**In this section:** [Responses and features](#choose-responses-and-features) ·
[One-feature charts](#one-feature-charts) ·
[Two-feature charts](#two-feature-charts) ·
[Expected values and model overlays](#expected-values-and-model-overlays) ·
[Grouped table and favourites](#grouped-table-and-favourites)

### Choose responses and features

The response consists of the shared Numerator and optional Denominator. Add up to two
Expected series from numeric dataset columns or active model predictions.

A normal feature click makes that field the only Feature 1. To add or remove Feature
2, use Command-click on macOS or Ctrl-click on Windows/Linux. The Expected chooser
uses the same convention for a second series. **No expected line** clears both.

Each feature has its own grouping controls. The swap action exchanges Feature 1 and
Feature 2 together with those settings.

Feature and Expected lists can use A–Z order. When model predictions are selected,
feature importance ordering is also available. GLM importance leads for a sole GLM
Expected series; GBM leads for GBM-only, mixed-model, or no-model comparisons.

### One-feature charts

A one-feature chart combines the selected response lines with Weight or row-count
bars. Depending on the feature type, you can:

- Use fixed-width numeric bands, quantiles, or untreated numeric values.
- Treat a numeric or date feature as a factor.
- Bucket dates by a calendar period and optionally show empty periods.
- Show or hide rows where the feature is missing.
- Group low-volume factor levels into **Other**.
- Winsorise numeric tails at the selected percentile cutoffs.
- Sort alphabetically, by response, by volume, or by supported model explanation.
- Transform the response and add sigma bars.
- Add a GLM partial-dependence line, a GBM SHAP ribbon, or both when the active models
  are compatible.

Missing groups remain distinct from low-weight tail grouping. Hiding missing values
recalculates the chart, table, transforms, overlays, and displayed row count, but not
the global sidebar metric summary.

### Two-feature charts

The two feature types determine the chart:

- Two untreated continuous numeric or date features produce a 3D surface.
- One continuous and one factor-style feature produce colour-matched response lines
  with stacked Weight or row-count bars.
- Two factor-style features produce a heatmap.

Numeric and date features can be forced to factor style to switch between these
views. Date axes are chronological when left continuous. Each feature retains its
own banding, date bucket, factor override, and missing handling.

Mixed line/bar charts always display volume. When several responses are selected, a
**Plot** control chooses which response is emphasised. Surfaces and heatmaps display
one response or Weight/row count at a time.

Heatmaps can label cells with Actual, Weight, or both when the formatted text fits.
They allow up to 100,000 populated grouped cells. Other Line and Bar charts retain a
10,000-group limit. Reduce categories, increase band sizes, or add a filter when the
limit is exceeded.

### Expected values and model overlays

Active GLM and GBM predictions behave like selectable numeric data sources. Line and
Bar can also expose a model prediction-ratio feature and order features by saved
importance.

For a one-feature chart:

- A GLM partial-dependence overlay shows the active model's fitted relationship for
  the selected x-axis feature.
- A GBM SHAP ribbon shows explanation percentiles grouped using the same filter,
  denominator, banding, missing, tail, and transform settings as the chart.
- **Both** aligns compatible GLM and GBM explanations for comparison.

An overlay remains bound to the model selected in that browser page when the request
is made. If another page activates a different model, the already displayed result
does not silently switch its underlying explanation.

### Grouped table and favourites

The server-backed table contains both selected feature columns and every selected
response. It supports search and pagination and uses the same totals and grouping as
the chart.

A Line/Bar view favourite saves metrics, filter, one or two groupings, independent
missing and grouping controls, plot choice, tail handling, heatmap labels, and the
one-feature empty-period choice. Optional stale Feature 2 settings can be dropped
while a valid Feature 1 is restored.

[Back to contents ↑](#contents)

## Histogram

Set the Numerator, optional Denominator, and global filter before opening Histogram.
The plotted row value is the Numerator or `Numerator / Denominator`.

### Binning and display

- Choose a number of bins or an explicit width in the original units.
- Explicit-width boundaries are anchored to rounded width multiples.
- Discrete integer values use integer-aware bins where possible.
- Show responsive values above bars.
- Use count or probability on the y-axis, with optional cumulative display.
- Apply a log axis.
- Add mean and median reference lines.
- Resize the divider between the chart and its metrics table.

The x-axis title names the selected `Numerator / Denominator` calculation. Average
row value shows only the Numerator name.

### Sample and exact modes

The default sampled mode uses up to 100,000 rows for a responsive first result. Use
all-row mode when exact bin counts are required. Summary metrics remain tied to the
selected mode and are displayed in the table beside the chart.

A Histogram view favourite stores the filter, bin mode and value, labels,
distribution, y-axis type, log state, sample mode, and selected metrics.

[Back to contents ↑](#contents)

## UK Mapping

Set the active metric and filter, then choose Area, Sector, or Units in UK Mapping.
Configure the postcode fields at launch when the recognised defaults are unavailable.

**In this section:** [Postcode columns](#postcode-columns) ·
[Area and sector maps](#area-and-sector-maps) ·
[Smoothing Parquet exports](#smoothing-parquet-exports) · [Unit maps](#unit-maps) ·
[Search, filtering, and saved views](#search-filtering-and-saved-views)

### Postcode columns

The default names are `PostcodeArea`, `PostcodeSector`, `PostcodeUnit`, `lat`, and
`long`. Common uppercase aliases such as `POSTCODE_AREA`, `POSTCODE_UNIT`,
`LATITUDE`, and `LONGITUDE` are detected. Override them at launch when needed:

```bash
lucidum path/to/my_data.parquet \
  --postcode-area Area \
  --postcode-sector Sector \
  --postcode-unit Unit \
  --latitude latitude \
  --longitude longitude
```

### Area and sector maps

Area and sector geometry is bundled with Lucidum. Choose among Blank, Esri, OSM,
Aerial, Light, and Dark bases and the available analytical palettes.

- Sector smoothing combines each sector with neighbours reachable in **N1** to
  **N5** steps, where each step crosses a shared polygon boundary.
  Geographically separate islands therefore do not smooth with the mainland;
  for example, Isle of Wight sectors smooth only with other connected sectors
  on the island.
- Area and sector outlines can be Off, Thin, or Bold.
- Area labels can be enabled for an analytical overview and resize with the map.
- Light and Dark use vector maps that keep roads, Lucidum outlines, and place labels
  legible over the analytical fill.

Rows with blank postcodes and nonblank postcodes absent from the bundled geometry
are reported separately. Map legends and hotspot choices use only values attached to
geometry that can be drawn.

![Lucidum UK Postcode Area map](https://github.com/SpeckledJim2/py_lucidum/raw/main/docs/assets/postcode_area.png)

![Lucidum UK Postcode Sector map](https://github.com/SpeckledJim2/py_lucidum/raw/main/docs/assets/postcode_sector_light.png)

### Smoothing Parquet exports

In Sector mode, choose **-> .parquet** beside the **Smooth** heading to create a
reusable table for the active Numerator, Denominator, and global filter. The map
calculates only the selected smoothing depth while you explore it; saving deliberately
calculates all five depths, regardless of which one is displayed. A temporary popup
reports the result with a compact filename while retaining the absolute path as its
accessible detail. Saving does not redraw or refresh the map.

Lucidum stores the generated file under the current dataset-version workspace:

```text
.lucidum/datasets/<dataset-slug>/<dataset-signature>/uk_map/sector_smoothing/
```

The readable filename includes a 12-character identity for the complete source,
metric, postcode column, and normalized filter specification. Repeating an identical
save atomically replaces that artifact; changing the filter or metric creates a
different file. The Parquet is a standalone result and does not appear as another
Lucidum data source.

### Unit maps

Unit mode is available when postcode-unit and coordinate columns are present. It
loads the complete eligible point set for the active global filter, then pans and
zooms locally without repeatedly replacing the points with a viewport sample.

- **Min** paints the smallest possible settled points for very dense maps.
- **Adaptive** grows dots with zoom and adjusts their maximum size for the filtered
  point count. Small datasets remain easy to see while large datasets avoid
  oversized points.

![Lucidum UK Postcode Unit map](https://github.com/SpeckledJim2/py_lucidum/raw/main/docs/assets/postcode_unit.png)

### Search, filtering, and saved views

Search for a postcode to navigate to it. Right-click the map to stage a region and
apply it as one global area filter. Postcode popups provide actions to view rows,
zoom, copy the postcode, or apply its filter.

The control and information strips can be collapsed for the browser session without
changing the map camera or requesting a new summary. The information strip reports
the active calculation, plotted and missing rows, row count, and filter state.

A Map view favourite stores the level, base, palette, presentation controls, metric,
filter, and camera where possible. The control-strip collapsed state remains a
session preference rather than part of the favourite.

[Back to contents ↑](#contents)

## Specifications

Specifications edits the CSV files that drive saved filters, KPI presets, and
Feature Specifications. Each editor continuously validates the draft against the
current dataset. Save is available only when the current content is valid.

The specifications are ordinary CSV files. They can be edited in Lucidum, a text
editor, or a spreadsheet, generated by another process, and reviewed in source
control. Feature Specifications can therefore keep modelling selection, interaction
structure, tabulation grids, and external report controls consistent across sessions.

When automatic discovery is enabled, Lucidum looks first in the current directory
and then in `./specs/` for these names:

- `filter_spec.csv`
- `kpi_spec.csv`
- `feature_spec.csv`

Use the matching CLI option to choose another path. A missing or disabled
specification opens as an unsaved starter draft. The target path is shown, and no
file is written until **Save** is selected.

Demo mode loads the bundled filter, KPI, and Feature Specifications when no explicit
or locally discovered file takes precedence.

**In this section:** [Filter Specification](#filter-specification) ·
[KPI Specification](#kpi-specification) ·
[Feature Specification](#feature-specification)

### Filter Specification

A saved-filter CSV has exactly these fields:

```csv
theme,name,expression
SAMPLE,Training,SAMPLE = 'training'
SAMPLE,Test,SAMPLE = 'test'
SAMPLE,Validation,SAMPLE = 'validation'
DRIVER AGE,Young drivers,DRIVER_AGE < 30
```

`theme` groups related rows in the chooser, `name` is the displayed label, and
`expression` is a DuckDB `WHERE` fragment.

### KPI Specification

A KPI CSV has these fields:

```csv
group,name,actual,denominator,decimals,format
VEHICLE,Vehicle age,VEHICLE_AGE,N,1,number
DRIVER,Driver age,DRIVER_AGE,N,1,number
FINANCIAL,Premium,PREMIUM,N,2,currency
```

`denominator` accepts `N`, `Average row value`, an empty value, `__none__`, or a
numeric column. `format` accepts `number`, `currency`, or `percent`.

### Feature Specification

Feature Specifications begin with `Feature,Grouping`, followed by recognised
metadata fields and any number of scenario columns. A compact example is:

```csv
Feature,Grouping,Base,min,max,banding,scenario1
DRIVER_AGE,DRIVER,40,17,96,1,feature
```

- `Feature` must match a dataset column exactly.
- `Grouping` labels features and supplies optional GBM interaction groups.
- `Base` anchors zero/one Line and Bar or SHAP rescaling and GLM tabulation bases.
- Numeric `min`, `max`, and `banding` define GLM rating-table grids.
- Scenario columns select rows whose cell contains `feature`, case-insensitively.

Optional `chart_*` fields control reproducible external reports:

- `chart_banding`
- `chart_quantiles`
- `chart_low_weights`
- `chart_missings`
- `chart_labels`
- `chart_sort`
- `chart_transform`
- `chart_sigma`
- `chart_date_bucket`
- `chart_empty_periods`

Blank `chart_banding` falls back to `banding` and then the report YAML. Other blank
`chart_*` values use the matching YAML default. Older Feature Specifications without
the recognised metadata fields remain usable; columns after `Grouping` are treated
as scenarios.

[Back to contents ↑](#contents)

## GLM

Enable GLM after installing the `glm` extra:

```bash
lucidum path/to/my_data.parquet --tools line-bar,glm
```

GLM uses the shared Numerator as the response. With a Denominator, it fits the rate
`Numerator / Denominator`, uses the Denominator as observation weight, saves
`glm_prediction` on the original Numerator scale, and exposes
`glm_prediction_rate`.

**In this section:** [Formula builder](#formula-builder) ·
[Families, regularisation, and training rows](#families-regularisation-and-training-rows) ·
[Results and saved models](#results-and-saved-models) ·
[Tabulations](#tabulations) · [Line and Bar integration](#line-and-bar-integration)

### Formula builder

Enter either a full `response ~ terms` Formulaic formula or right-hand-side terms
only. RHS-only formulas use the active Numerator as the response. Lines can contain
`#` comments.

The formula context includes common numeric transforms plus `ifelse`, `pmin`,
`pmax`, `ns`, `bs`, `cs`, `poly`, and `C`. Explicit `offset(...)` terms are fitted
and applied again when predictions or tabulations are produced.

The formula and model-parameter panels are mutually exclusive and can be collapsed
to enlarge the editor. Formula actions clear the editor, adjust its font size, or
copy the live formula.

### Families, regularisation, and training rows

Supported families are:

- Normal.
- Poisson.
- Gamma.
- Tweedie, with a configurable variance power.
- Binomial.
- Inverse Gaussian.
- Negative binomial, with configurable theta.

The link defaults automatically. Choose **None** for an unregularised fit, **Auto**
to cross-validate ridge, elastic-net, and lasso mixes, or **Manual** to set the mix
and alpha. Penalised models show coefficients but omit standard errors and p-values
because those values are not equivalent to unpenalised GLM inference.

Choose **All** to fit every valid row, **Training** to fit rows whose physical
`SAMPLE` column equals `training`, or **Training + Test** to fit both `training`
and `test` rows. Sample matching is trimmed and case-insensitive. GLM does not
create a generated sample split.

### Results and saved models

After fitting, Lucidum scores the eligible rows and saves the model in the current
dataset workspace. The coefficient table is sortable and provides copy and download
actions. The model navigator shows the active model, formula/model size context,
training choices, diagnostics, and whether tabulations exist.

Select saved model rows to rename one model, activate one model, open its folder when
local desktop access is available, or delete one or more models. Deleting the active
model promotes the newest remaining model.

Activating a model publishes its predictions as a shared data source and restores
its training Numerator and Denominator.

### Tabulations

The Tabulations tab builds insurance-style rating tables from selected saved GLMs.
It uses the fitted estimator, formula terms, offsets, and Feature Specification
`Base`, `min`, `max`, and `banding` values. Missing numeric metadata can be estimated
from scored rows and is reported in the app.

Tables can be viewed or plotted on linear or exponential scales and exported to
XLSX. They also publish row-level `glm_tabulated_prediction` values.

A one-way table can be rebased from a selected cell. A two-feature interaction can
rebase either selected feature slice while transferring the necessary offsets to the
other one-way table. Lucidum verifies after every change that row-level tabulated
predictions and missing states have not changed. Active rebasing rules are shown
below the selectors and can be cleared for one table or all tables.

Older GLMs created before fitted-estimator persistence must be rebuilt before they
can produce tabulations or reconstructed overlays.

### Line and Bar integration

An active GLM adds `glm_prediction`, `glm_prediction_rate`, and any tabulated
prediction to shared selectors where applicable. On one-feature Line and Bar charts,
the GLM overlay displays the fitted relationship for the selected feature. Simple
models are evaluated directly over the plotted grid; interaction models use a
consistent partial-dependence treatment.

Tabulation plots identify the selected feature, keep dense categories legible,
emphasise the zero axis, and provide a chart-image copy action.

[Back to contents ↑](#contents)

## GBM

Enable GBM after installing the `gbm` extra:

```bash
lucidum path/to/my_data.parquet --tools line-bar,gbm
lucidum path/to/my_data.parquet --tools line-bar,uk-map,glm,gbm \
  --features specs/feature_spec.csv
```

GBM uses the active Numerator as response and the Denominator as exposure or offset
where required by the selected objective. It saves predictions and diagnostics for
reuse in later sessions.

**In this section:** [Features and parameters](#features-and-parameters) ·
[Interaction constraints](#interaction-constraints) ·
[Sample rows and training modes](#sample-rows-and-training-modes) ·
[Evaluation and models](#evaluation-and-models) ·
[Feature importance and EBM Gain](#feature-importance-and-ebm-gain) ·
[SHAP](#shap) · [Tree viewer](#tree-viewer) ·
[Shared outputs and exports](#shared-outputs-and-exports)

### Features and parameters

The first GBM tab contains a Feature grid, Parameter grid, and Evaluation Log.
Resize the boundaries between them to prioritise the current task; those layout
choices last for the browser session.

The Feature grid supports selection, monotonicity, saved Grouping metadata, and
saved Gain or mean absolute SHAP importance. A Feature Specification scenario
selects the usable features marked for that scenario.

Parameter values can contain grid-search braces:

```text
{200, 300, 400}
{0.05, 0.3; 0.05}
{bagging, goss}
```

Lucidum samples a large hypergrid deterministically, skips invalid combinations with
a notice, trains each selected combination, and activates the best completed model.

`tweedie_variance_power` is available for a Tweedie objective or metric and must
satisfy LightGBM's `1.0 <= value < 2.0` constraint. It is accepted but has no effect
when neither the objective nor metric is Tweedie.

The first parameter, `init_score`, can use the normal denominator-derived starting
point, a numeric dataset field, or an active GLM prediction. Selected prediction
values are transformed into the objective's linear-predictor space before training.

### Interaction constraints

GBM supports three user-facing constraint types:

- **Main effect only (1D)** prevents the feature from interacting.
- **Pair interaction (2D)** allows the selected pair. Pairs can overlap other pairs.
- **Constraint groups** allow selected features to interact only with members of the
  same Feature Specification `Grouping`.

Right-click a Feature cell to add or remove main-effect and pair constraints. The
pair manager lists every allowed pair and can seed a new pair from the clicked
feature.

Main-effect-only features cannot also be paired or grouped. Pair members cannot be
placed in a selected constraint group. When any explicit pair exists, pairs and
groups form an exhaustive allowlist: uncovered selected features are automatically
constrained to their main effect. Pair mode requires `num_leaves <= 3`; invalid
scalar settings fail validation and invalid grid combinations are skipped.

When constraint groups and nonzero SHAP rows are selected, GBM can also create one
SHAP-centred LightGBM text model for each chosen group. The app verifies its raw
predictions against the matching grouped SHAP contribution and reports the maximum
absolute difference. A group with no fitted trees is reported without creating an
empty model file.

### Sample rows and training modes

If the source has a `SAMPLE` column, GBM trains on `training`, uses `test` for early
stopping, and scores `validation` independently. Validation never affects fitting or
early stopping. Predictions are calculated for all eligible rows.

If no physical sample exists, create one reusable generated 60/20/20 split from the
tool. The split belongs to the current dataset workspace. For a durable modelling
pipeline, add a real `SAMPLE` column to the source data.

Normal mode trains the requested LightGBM directly. EBM mode starts with 2-leaf
trees, then increases the leaf count when the test metric stops improving, up to the
configured `num_leaves`. The configured iteration count is a total cap across all
stages. EBM mode requires both training and test rows.

### Evaluation and models

During training, the Evaluation Log displays current iteration and train/test metric
values. Grid searches also show the current candidate number. The live x-axis uses
the configured iteration limit; completed models use their actual tree count.

Choose the full evaluation history or a tail-focused view when early improvement
makes later changes hard to see. The completed plot adds one Validation marker at
the best iteration when Validation rows exist. Long histories can be sampled for the
browser chart without changing the complete saved evaluation data. Copy the chart
image from its action beside the view control.

The Model navigator compares saved models and their objective, metric, training, and
constraint context. Select rows to rename or activate one model, open one local model
folder, or delete several model folders. Deleting the active model promotes the
newest remaining model.

Saved feature scenarios, constraint groups, and interaction pairs are restored when
a model is selected. If the Feature Specification has changed, Lucidum marks saved
scenarios or groups as changed or missing while retaining the fitted model context.

### Feature importance and EBM Gain

Saved models with SHAP rows can switch the Feature grid between LightGBM Gain and
mean absolute SHAP importance. EBM models can also show **EBM Gain**, which groups
tree gain by fitted feature combination. A two-feature EBM Gain row can seed a pair
constraint before retraining.

### SHAP

Choose a nonzero SHAP row setting during training to enable the SHAP and Stacked SHAP
tabs. The 10,000-row and 100,000-row choices use deterministic samples from scored
rows; **All** saves every scored row.

The SHAP tab lists only trained features. Feature 1 defaults to the highest-Gain
feature and Feature 2 defaults to **None**. Both choosers can sort by importance or
A–Z. Compatible selections and legend visibility are retained when switching active
models.

One-feature SHAP plots show numeric percentile ribbons or categorical box plots.
Numeric features treated as factors retain natural numeric order. Dense numeric
flame plots provide zoom and suppress labels only when they cannot remain legible.

Two-feature SHAP plots show the sum of both selected contributions:

- Two continuous features use a dense-grid 3D surface.
- Continuous-by-factor selections use lines.
- Two factor-style selections use a heatmap.

Each feature has independent banding, tail grouping, and factor treatment. If a
Feature Specification supplies a `Base`, the `0` and `1` rescale choices can anchor
the explanation. The `1` choice exponentiates first and displays uplift from the base
as a percentage.

Stacked SHAP groups feature contributions by one selected model feature. It remains
on the linear-predictor contribution scale and does not apply Base rescaling. The
feature chooser can be collapsed or resized, and moves above the chart on narrow
screens.

When Feature Specification interaction groups were used, grouped SHAP contribution
fields are saved and exposed as selectable shared data outputs.

### Tree viewer

The Tree viewer provides a searchable tree list and an interactive diagram with
zoom, colour, and direction controls. Select a node to highlight its path from the
root. Split labels use compact formatting, long categorical splits are summarised in
the diagram, and the full rule remains available in the tooltip.

The summary identifies explicit constraints that governed the displayed tree as a
singleton, pair, or group. A pair or group is reported when any saved member appears;
this describes the training allowlist rather than claiming that every allowed member
was used in that tree.

### Shared outputs and exports

Activating a GBM refreshes its features, parameters, training mode, evaluation,
trees, SHAP views, and plot-ready outputs. Predictions, prediction rates, SHAP
values, and grouped interaction contributions appear in shared selectors where
applicable.

Line and Bar can compare the prediction with Actual and add a SHAP ribbon for the
selected feature. UK Mapping can aggregate compatible predictions and SHAP fields by
postcode. Saved GBM tabulations can be exported to XLSX from the Tabulations panel.

[Back to contents ↑](#contents)

## Saved work and model workspaces

### Favourite storage

By default, favourites are saved beside the dataset workspaces but above the dataset
version signature:

```text
.lucidum/datasets/<dataset-slug>/line_bar/favourites.json
```

They therefore survive replacement of a dataset at the same path. Set an explicit
file with `--line-bar-favourites` when a shared or server-specific location is
required. Its parent folder is created on first save; malformed JSON is reported and
is not overwritten by that request.

Metric favourites retain whether a model supplied the Denominator, so the reference
follows the active model of that family. Lucidum migrates supported older favourite
formats and reports fields that can no longer be restored. Drag favourites to set
their order; the first becomes the startup view unless the CLI, URL, or Python call
selects another one.

### Model workspaces

Lucidum stores modelling sidecars under:

```text
.lucidum/datasets/<dataset-slug>/<dataset-signature>/models/
```

The slug comes from the dataset filename. The signature uses the file's size,
modification time, row count, and schema fingerprint. Models trained for one file or
one version therefore do not silently appear against another.

Replacing or editing a dataset creates a new signature. Earlier models remain on
disk but are not shown for the new version; rebuild or deliberately reinstall a
compatible external model. Folder inputs do not create modelling workspaces.

### Managing saved models

The GLM and GBM Model navigators provide **Open folder** when Lucidum is accessed
through a local address and the host has a supported desktop file manager. The
action opens the folder on the server machine, so it is hidden for LAN browsers,
headless environments, and unsupported systems.

Model folders can be renamed, activated, or deleted from their navigator. Deletion
is permanent at the filesystem level; when the active model is deleted, Lucidum
promotes the newest remaining model so shared predictions continue to have an
unambiguous source.

[Back to contents ↑](#contents)

## Python usage

### Start the app

```python
import py_lucidum

py_lucidum.serve(py_lucidum.demo_dataset_path(), port=8000, open_browser=True)
py_lucidum.serve("path/to/my_data.parquet", port=8000, open_browser=True)
py_lucidum.serve("path/to/monthly_parquets/", port=8000, open_browser=True)
```

Common launch settings are also keyword arguments:

```python
url = py_lucidum.serve(
    "path/to/my_data.parquet",
    port=8000,
    open_browser=True,
    buttons=True,
    title_prefix="Motor pricing data",
    line_bar_favourites_path="config/favourites.json",
    line_bar_favourite="Loss curve",
)
```

In Positron, Jupyter, and similar notebook-style runtimes, `serve()` starts in the
background and returns the URL immediately. In a normal Python shell it blocks until
the server is stopped.

For ASGI setup, create the app and run it explicitly:

```python
import py_lucidum
from py_lucidum.app import create_app

app = create_app(
    "path/to/my_data.parquet",
    token="dev-token",
    tools=["line_bar", "dataset_viewer", "column_profile", "histogram"],
    defaults={
        "x": "DRIVER_AGE",
        "actual": "PREMIUM",
        "denominator": "ANNUAL_MILEAGE",
    },
    filters_path="specs/filter_spec.csv",
    kpis_path="specs/kpi_spec.csv",
    features_path="specs/feature_spec.csv",
)

py_lucidum.run_app(app, host="127.0.0.1", port=8000, open_browser=True)
```

### Postcode-sector smoothing

`smooth_postcode_sectors()` runs the same smoothing implementation without starting
the Lucidum app. It reads one source Parquet and atomically writes an explicit output
path, creating missing parent directories:

```python
from py_lucidum import demo_dataset_path, smooth_postcode_sectors

output_path = smooth_postcode_sectors(
    demo_dataset_path(),
    "local/postcode_sector_smoothing/motor_premiums_training.parquet",
    postcode_sector="POSTCODE_SECTOR",
    numerator="PREMIUM",
    filter="SAMPLE = 'training'",
)
print(output_path)
```

Pass a numeric column as `denominator` for a weighted ratio. If it is omitted, each
valid Numerator row contributes one to the denominator, producing Lucidum's
**Average row value** calculation. `filter` is an optional DuckDB `WHERE` expression.
The selected columns must be physical columns in the source Parquet.

The result is ordered by postcode sector and has one row per bundled geometry sector,
plus any valid source-sector key absent from the geometry:

```text
postcode_sector, numerator_sum, denominator_sum, unsmoothed,
smooth_n1, smooth_n2, smooth_n3, smooth_n4, smooth_n5
```

The two sums are the unsmoothed source aggregates. A geometry sector with no source
rows can still receive a value from its neighbours. A valid source sector outside
the bundled geometry is retained and each smoothed value falls back to `unsmoothed`.
Blank sector values are ignored; other filtered sector values must be uppercase
canonical sectors with one space, such as `AB10 1`.

The source and output paths must differ. An existing output is replaced only after a
complete successful calculation, so validation and calculation failures leave it
untouched. Run the complete demo from a source checkout with:

```bash
python examples/postcode_sector_smoothing_demo.py
```

It smooths average `PREMIUM` for `SAMPLE = 'training'` and writes beneath the
git-ignored `local/` directory. This standalone example is intentionally not part of
the numbered GLM/GBM files installed by `lucidum --sync-examples`.

### Reporting and modelling helpers

The public package also provides:

- `serve_line_bar()` as a focused Line and Bar launch helper.
- `build_glm_tabulations()`, `score_glm_tabulations()`, and
  `export_glm_tabulations()`.
- `line_bar_chart()` and `gbm_evaluation_chart()`.
- `write_echarts_report()`, `write_glm_summary_report()`, and
  `write_gbm_summary_report()`.
- `report_filename()`.
- `smooth_postcode_sectors()`.
- `extract_lightgbm_interaction_group()`.

The interaction-group extractor works directly on a LightGBM text model without
starting Lucidum or importing LightGBM. The requested feature names must exactly
match one non-overlapping saved interaction group. It verifies that retained trees
use only that group, rewrites the model metadata for the reduced feature set, and by
default centres raw scores to the matching grouped SHAP contribution. The source
model cannot be used as the output path and is never overwritten.

See the executable examples and [external-model guide](external-model-builds-and-reports.md)
for complete reporting calls and saved-model inputs.

[Back to contents ↑](#contents)

## External models and reports

The repository contains parallel YAML-controlled GLM and GBM workflows for building
and scoring a model outside the app, producing interactive HTML charts, and creating
model summaries. GLM summaries can also produce rating tables and XLSX output.

Reports read the named results folder directly and do not require a `.lucidum`
workspace. The optional installation step copies and activates the completed model
beside its source dataset so it can be reviewed in Lucidum without retraining.

Install the example dependencies from a checkout:

```bash
python -m pip install -e ".[glm,gbm,examples]"
```

To refresh an existing client workflow directory from an installed release, upgrade
Lucidum and sync its maintained Python files:

```bash
pipx upgrade py-lucidum
lucidum --sync-examples /path/to/client/examples
```

The destination directory is created when necessary. The command creates or
overwrites the six numbered scripts and four Python helpers, but never changes YAML,
formula, specification, data, unknown Python, or other client files. Use
`lucidum --sync-examples /path/to/client/examples --dry-run` to list the changes
without writing them. A virtual-environment installation can use the same
`lucidum` command after upgrading with `python -m pip install --upgrade`.

The numbered scripts under `examples/` can be run normally or as `# %%` cells in
Positron. See
[Build models outside Lucidum, then report or view them](external-model-builds-and-reports.md)
for the commands, YAML fields, data rules, artefacts, report settings, and direct API
examples.

[Back to contents ↑](#contents)

## Monitoring and troubleshooting

Launch with `--buttons` and choose **Open monitor**, or open `/monitor?token=...`
directly. The Model operations panel follows GLM builds, GLM tabulations, and GBM
training from preflight through background work. It reports wall time, CPU time,
average cores, observed memory, and a phase/request timeline.

**Copy diagnostics** produces a sanitised report containing operation, runtime, and
dataset metadata. It excludes request bodies, filter expressions, tokens, and
absolute paths. Operation history is bounded in memory and is not written to disk.

See the [Model Operations Monitor guide](telemetry-monitor.md) for metric definitions,
privacy details, limits, and troubleshooting.

Common first checks are:

- Run `lucidum --version` and confirm Python is 3.13 or newer.
- Use Parquet for large interactive datasets.
- Confirm every file in a Parquet folder has the same schema.
- Use a single file when GLM or GBM is enabled.
- Install `libomp` when LightGBM reports a missing OpenMP runtime on macOS.
- Reduce grouping cardinality or filter the data when Line and Bar reaches its group
  limit.
- Rebuild a model after editing or replacing the source dataset.
- Use the Blank UK map when working without internet access.

For implementation architecture, HTTP contracts, testing, or releases, use
[DEVELOPMENT.md](../DEVELOPMENT.md).

[Back to contents ↑](#contents)
