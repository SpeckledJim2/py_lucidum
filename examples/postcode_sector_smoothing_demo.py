"""Smooth training-row PREMIUM values across UK postcode sectors."""

# %% Imports

from pathlib import Path

import duckdb

from py_lucidum import demo_dataset_path, smooth_postcode_sectors


# %% 1. Choose the bundled demo data and an explicit output Parquet

script_file = globals().get("__file__")
project_root = Path(script_file).resolve().parents[1] if script_file else Path.cwd()
output_path = (
    project_root
    / "local"
    / "postcode_sector_smoothing"
    / "motor_premiums_training.parquet"
)


# %% 2. Smooth average PREMIUM for SAMPLE = training

# Choose the metric here. Both names must refer to physical numeric columns in
# the source Parquet.
#
# numerator="PREMIUM" means that PREMIUM is summed within each postcode sector.
#
# denominator=None means "Average row value": every row with a valid PREMIUM
# contributes 1 to the denominator, so the unsmoothed result is:
#
#     SUM(PREMIUM) / COUNT(valid PREMIUM rows)
#
# To calculate a weighted ratio instead, replace None with a numeric column name,
# for example denominator="EXPOSURE". The result would then be:
#
#     SUM(PREMIUM) / SUM(EXPOSURE)
#
# When changing these columns, also rename motor_premiums_training.parquet above
# so the output filename describes the calculation.
result_path = smooth_postcode_sectors(
    demo_dataset_path(),
    output_path,
    postcode_sector="POSTCODE_SECTOR",
    numerator="PREMIUM",
    denominator=None,
    filter="SAMPLE = 'training'",
)

print(f"Postcode-sector smoothing: {result_path}")


# %% 3. Inspect a few populated sectors

with duckdb.connect(database=":memory:") as connection:
    preview = connection.execute(
        """
        SELECT *
        FROM read_parquet(?)
        WHERE unsmoothed IS NOT NULL
        ORDER BY postcode_sector
        LIMIT 10
        """,
        [str(result_path)],
    ).fetchall()

for row in preview:
    print(row)
