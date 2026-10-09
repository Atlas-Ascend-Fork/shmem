# Copyright (c) gcw_Z57idahf. 2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ================================================================================================================

"""Host regressions for Linux memory-query inputs; no physical memory scan."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[3] / "examples/rma_d2h_demo/check_support.py"
SPEC = importlib.util.spec_from_file_location("shmem_memory_query", SOURCE)
QUERY = importlib.util.module_from_spec(SPEC)
# Control the Linux page-size input so this host test can run on other systems.
with patch.object(os, "sysconf", return_value=4096, create=True):
    SPEC.loader.exec_module(QUERY)


class KernelConfigTests(unittest.TestCase):
    def read_zone_order(self, config_text, ubuntu):
        with tempfile.TemporaryDirectory(prefix="shmem-kernel-config-") as directory:
            config = Path(directory) / "config-host-test"
            if config_text is not None:
                config.write_text(config_text, encoding="utf-8")
            actual_open = open

            def open_kernel_config(path, *args, **kwargs):
                # Only map the OS-owned config location to a real fixture file.
                if str(path) == "/boot/config-host-test":
                    return actual_open(config, *args, **kwargs)
                raise FileNotFoundError(path)

            def uname(command, **kwargs):
                if command == ["uname", "-r"]:
                    return "host-test\n"
                self.assertEqual(command, ["uname", "-a"])
                return "Linux host-test Ubuntu\n" if ubuntu else "Linux host-test\n"

            with patch("builtins.open", side_effect=open_kernel_config):
                with patch.object(QUERY.subprocess, "check_output", side_effect=uname):
                    return QUERY.get_force_max_zoneorder()

    def test_existing_kernel_config_has_priority(self):
        self.assertEqual(
            self.read_zone_order("CONFIG_FORCE_MAX_ZONEORDER=12\n", False), 12
        )

    def test_existing_kernel_config_overrides_ubuntu_fallback(self):
        self.assertEqual(
            self.read_zone_order("CONFIG_FORCE_MAX_ZONEORDER=14\n", True), 14
        )

    def test_missing_config_retains_non_ubuntu_fallback(self):
        self.assertEqual(self.read_zone_order(None, False), 11)

    def test_missing_config_retains_ubuntu_fallback(self):
        self.assertEqual(self.read_zone_order(None, True), 13)


class AddressDisplayTests(unittest.TestCase):
    START = 0x29580000000  # First range used by the documented query script.
    MIB = 1024 * 1024

    def query(self, free_start, free_end, range_start, range_end):
        stdout = io.StringIO()
        segments = [(free_start // 4096, free_end // 4096)]
        target_ranges = [(range_start // 4096, range_end // 4096)]
        with patch.object(
            QUERY, "scan_node_free_segments", return_value=iter(segments)
        ):
            with contextlib.redirect_stdout(stdout):
                total = QUERY.run_for_node(0, 2, target_ranges, 2 * self.MIB)
        return total, stdout.getvalue()

    def test_aligned_range_is_printed_in_bytes(self):
        start, end = self.START, self.START + 8 * self.MIB - 1
        total, output = self.query(start, end, start, end)
        self.assertEqual(total, 8)
        self.assertIn(
            "FREE∩RANGE: 0x0000029580000000 - 0x0000029580800000  size=8.00 MB", output
        )

    def test_unaligned_range_preserves_byte_alignment(self):
        start, end = self.START + 4096, self.START + 8 * self.MIB - 4097
        total, output = self.query(
            start, end, self.START, self.START + 8 * self.MIB - 1
        )
        self.assertEqual(total, 4)
        self.assertIn(
            "FREE∩RANGE: 0x0000029580200000 - 0x0000029580600000  size=4.00 MB", output
        )

    def test_disjoint_ranges_do_not_print_a_segment(self):
        total, output = self.query(
            self.START,
            self.START + 4 * self.MIB - 1,
            self.START + 8 * self.MIB,
            self.START + 12 * self.MIB - 1,
        )
        self.assertEqual(total, 0)
        self.assertNotIn("FREE∩RANGE:", output)

    def test_small_range_keeps_existing_minimum_filter(self):
        total, output = self.query(
            self.START,
            self.START + self.MIB - 1,
            self.START,
            self.START + 8 * self.MIB - 1,
        )
        self.assertEqual(total, 0)
        self.assertNotIn("FREE∩RANGE:", output)


if __name__ == "__main__":
    unittest.main()
