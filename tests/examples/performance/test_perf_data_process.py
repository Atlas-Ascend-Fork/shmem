# Copyright (c) gcw_Z57idahf. 2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ================================================================================================================

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/utils/perf_data_process.py"
HEADERS = ["DataSize/B", "Npus", "Blocks", "UBsize/KB", "Bandwidth/GB/s", "CoreMaxTime/us"]
VALUES = [1048576, 2, 8, 16, 512, 16.384]
spec = importlib.util.spec_from_file_location("perf_data_process", SCRIPT)
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)


def write_csv(path, headers, values):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerow(values)


@pytest.mark.parametrize(
    ("headers", "values"),
    [
        (HEADERS, VALUES),
        (HEADERS + ["SingleCoreTime/us"], VALUES + [16.384, 15.0]),
        (
            HEADERS[:4] + ["Bandwidth/GB/s (1000)", "Bandwidth/GiB/s (1024)"] + HEADERS[5:],
            VALUES[:5] + [476.837158203125] + VALUES[5:],
        ),
        (
            HEADERS[:4] + ["Bandwidth/GB/s(1000)", "Bandwidth/GiB/s(1024)"] + HEADERS[5:],
            VALUES[:5] + [476.837158203125] + VALUES[5:],
        ),
        (
            HEADERS[:5] + ["Bandwidth/GiB/s"] + HEADERS[5:],
            VALUES[:5] + [476.837158203125] + VALUES[5:],
        ),
        (
            HEADERS[:4] + ["Bandwidth/GB/s (1000)", "Bandwidth/GiB/s (1024)"]
            + HEADERS[5:] + ["SingleCoreTime/us"],
            VALUES[:5] + [476.837158203125] + VALUES[5:] + [16.384, 15.0],
        ),
    ],
    ids=["legacy", "extra-core-times", "mte", "udma", "rdma", "mte-extra-core-times"],
)
def test_process_csv_preserves_column_meanings(tmp_path, headers, values):
    source = tmp_path / "performance.csv"
    write_csv(source, headers, values)
    images, frame = perf.process_csv(source, tmp_path / "picture", None, None, no_plot=True)
    assert images == []
    assert list(frame.columns) == HEADERS
    assert frame.iloc[0].tolist() == VALUES


def test_cli_report_uses_core_time_instead_of_gib_bandwidth(tmp_path):
    source = tmp_path / "put_float_0.csv"
    write_csv(
        source,
        HEADERS[:4] + ["Bandwidth/GB/s (1000)", "Bandwidth/GiB/s (1024)"] + HEADERS[5:],
        VALUES[:5] + [476.837158203125] + VALUES[5:],
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--csv-path", str(source), "--no-plot"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / "performance_report.md").read_text(encoding="utf-8").splitlines()
    table_header = next(line for line in lines if line.startswith("|") and "CoreMaxTime/us" in line)
    table_row = next(line for line in lines if line.startswith("|") and "1048576" in line)
    columns = [part.strip() for part in table_header.strip("|").split("|")]
    values = [part.strip() for part in table_row.strip("|").split("|")]
    assert float(values[columns.index("CoreMaxTime/us")]) == 16.384
