#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ================================================================================================================

import argparse
import csv
import re
from decimal import Decimal
from pathlib import Path


def normalized(value):
    return re.sub(r"[^a-z]", "", value.lower())


def find_column(headers, predicate):
    for index, header in enumerate(headers):
        if predicate(normalized(header)):
            return index
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-operation total time from RDMA demo msprof CSVs."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run output directory containing pe_0, pe_1, ...",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path (default: <run_dir>/totaltime.csv)",
    )
    args = parser.parse_args()

    files = sorted(args.run_dir.glob("pe_*/PROF_*/mindstudio_profiler_output/op_statistic*.csv"))
    if not files:
        parser.error(f"no op_statistic CSV files found under {args.run_dir}")

    output = args.output if args.output is not None else args.run_dir / "totaltime.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    averages = {}
    found_column = False
    for path in files:
        pe_match = re.fullmatch(r"pe_([0-9]+)", path.relative_to(args.run_dir).parts[0])
        if pe_match is None:
            continue
        pe = pe_match.group(1)
        with path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.reader(csv_file)
            try:
                headers = next(reader)
            except StopIteration:
                continue
            total_index = find_column(headers, lambda name: name.startswith("totaltime"))
            if total_index is None:
                continue
            found_column = True
            op_index = find_column(headers, lambda name: name in {"opname", "operatorname", "optype"})
            count_index = find_column(headers, lambda name: name == "count")
            avg_index = find_column(headers, lambda name: name.startswith("avgtime"))
            min_index = find_column(headers, lambda name: name.startswith("mintime"))
            max_index = find_column(headers, lambda name: name.startswith("maxtime"))
            for row in reader:
                if total_index < len(row):
                    op_name = row[op_index] if op_index is not None and op_index < len(row) else ""
                    count = row[count_index] if count_index is not None and count_index < len(row) else ""
                    avg_time = row[avg_index] if avg_index is not None and avg_index < len(row) else ""
                    min_time = row[min_index] if min_index is not None and min_index < len(row) else ""
                    max_time = row[max_index] if max_index is not None and max_index < len(row) else ""
                    rows.append([pe, op_name, count, row[total_index], avg_time, min_time, max_time, "", "", ""])
                    if avg_time:
                        averages.setdefault(op_name, []).append((pe, Decimal(avg_time)))

    if not found_column:
        parser.error(f"no Total Time/totaltime column found in CSVs under {args.run_dir}")
    for op_name, values in averages.items():
        mean = sum(value for _, value in values) / len(values)
        maximum = max(value for _, value in values)
        slowest = ";".join(pe for pe, value in values if value == maximum)
        rows.append(["ALL", op_name, "", "", "", "", "", str(mean), str(maximum), slowest])
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["PE", "Op Name", "N", "Total Time", "Avg Time", "Min Time", "Max Time",
                         "Mean PE Avg Time", "Max PE Avg Time", "Slowest PE"])
        writer.writerows(rows)
    print(f"CSV saved: {output.resolve()}")


if __name__ == "__main__":
    main()
