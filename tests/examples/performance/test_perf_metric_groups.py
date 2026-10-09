# Copyright (c) gcw_Z57idahf. 2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Offline regression for the documented full_op/comm_only MoE CSV format."""

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/utils/perf_data_process.py"
HEADERS = ["DataSize/B", "Npus", "Blocks", "UBsize/KB", "Bandwidth/GB/s", "CoreMaxTime/us"]
MOE_HEADERS = HEADERS + [
    "Metric", "GlobalDataSize/B", "PerPeBandwidth/GB/s", "BS", "H", "TopK",
    "ExpertPerPe", "Dtype", "Warmup", "Loops", "ProfPe", "CaseId", "SingleCoreTime/us",
]


def write_moe_csv(path):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(MOE_HEADERS)
        for metric, bandwidth, core_time in [("full_op", 100, 19.53125), ("comm_only", 200, 9.765625)]:
            writer.writerow([
                1048576, 2, 8, 16, bandwidth, core_time, metric, 2097152,
                bandwidth / 2, 8, 64, 2, 2, "int32_t", 5, 50, 0,
                "shape_8_64_2_4_2", core_time,
            ])


def load_perf():
    spec = importlib.util.spec_from_file_location("perf_metric_groups", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_moe_cli_retains_metric_identity_and_separate_charts(tmp_path):
    source = tmp_path / "combine_perf_rank0.csv"
    write_moe_csv(source)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = (tmp_path / "performance_report.md").read_text(encoding="utf-8")
    assert "Metric" in report and "full_op" in report and "comm_only" in report
    assert len(list((tmp_path / "picture").rglob("*.png"))) == 8
    for metric in ("full_op", "comm_only"):
        assert f"{metric}: bandwidth_max_heatmap" in report
        assert f"{metric}: bandwidth_mean_heatmap" in report


def test_moe_heatmap_statistics_do_not_mix_measurement_scopes(tmp_path, monkeypatch):
    source = tmp_path / "dispatch_perf_rank0.csv"
    write_moe_csv(source)
    perf = load_perf()
    values = []
    close = perf.plt.close

    def inspect_and_close(*args, **kwargs):
        if perf.plt.get_fignums():
            figure = perf.plt.gcf()
            if figure.axes and figure.axes[0].collections:
                values.append(float(figure.axes[0].collections[0].get_array().ravel()[0]))
        close(*args, **kwargs)

    monkeypatch.setattr(perf.plt, "close", inspect_and_close)
    images, frame = perf.process_csv(source, tmp_path / "picture", None, None)
    assert sorted(values) == pytest.approx([100, 100, 200, 200])
    assert frame["Metric"].tolist() == ["full_op", "comm_only"]
    assert len(images) == 8
    assert all(Path(image["path"]).is_file() for image in images)


def test_legacy_csv_keeps_existing_chart_names_and_columns(tmp_path):
    source = tmp_path / "put_float_0.csv"
    with source.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(HEADERS)
        writer.writerow([1048576, 2, 8, 16, 100, 20])
    perf = load_perf()
    images, frame = perf.process_csv(source, tmp_path / "picture", None, None)
    assert list(frame.columns) == HEADERS + ["y"]
    assert len(images) == 4
    assert {Path(image["path"]).parent for image in images} == {tmp_path / "picture"}
    assert all(Path(image["path"]).name.startswith("put_float_0_") for image in images)
