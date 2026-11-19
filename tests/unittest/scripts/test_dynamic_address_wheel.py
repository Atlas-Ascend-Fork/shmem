# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise setup.py's real wheel staging and package-data inclusion without CANN."""
import os
import contextlib
import pathlib
import re
import runpy
import shutil
import subprocess
import tempfile
from unittest import mock
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py

ROOT = pathlib.Path(__file__).resolve().parents[3]


def check_build_mode(build_class):
    if not shutil.which("cmake"):
        print("SKIP wheel configure cache check: cmake is unavailable", flush=True)
        return

    class CapturedConfigure(Exception):
        pass

    with tempfile.TemporaryDirectory(prefix="shmem-wheel-mode-") as directory, contextlib.ExitStack() as cleanup:
        cleanup.callback(os.chdir, pathlib.Path.cwd())
        os.chdir(directory)
        pathlib.Path("CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.18)
project(mode NONE)
option(ACLSHMEM_ENABLE_DEVICE_SYMBOL "mode" OFF)
file(WRITE "${CMAKE_BINARY_DIR}/mode.txt" "${ACLSHMEM_ENABLE_DEVICE_SYMBOL}")
''')
        command = build_class(Distribution())
        # Use setup.py's actual configure arguments; no compiler or NPU required.
        for options, expected in (("-DACLSHMEM_ENABLE_DEVICE_SYMBOL=ON", "ON"), ("", "OFF")):
            with mock.patch.dict(os.environ, {"COMPILE_OPTIONS": options, "_SHMEM_PREBUILT": ""}):
                with mock.patch("subprocess.check_call", side_effect=CapturedConfigure) as call:
                    try:
                        command._build_cpp()
                    except CapturedConfigure:
                        pass
                args, kwargs = call.call_args
                # setup.py supplies an argv list and working directory, not shell options.
                subprocess.run(args[0], cwd=kwargs["cwd"], check=True)
            actual = pathlib.Path("build/mode.txt").read_text()
            assert actual == expected, f"wheel mode: expected {expected}, got {actual}"
        print("WHEEL_MODE_RESET_PASS", flush=True)


def main():
    previous = pathlib.Path.cwd()
    try:
        for mode in (0, 1):
            with tempfile.TemporaryDirectory(prefix="shmem-wheel-meta-") as directory, contextlib.ExitStack() as cleanup:
                base = pathlib.Path(directory)
                os.chdir(base)
                cleanup.callback(os.chdir, previous)
                prefix = base / "install/shmem"
                (prefix / "backends/950").mkdir(parents=True)
                (prefix / "backends/950/libshmem.so").write_bytes(b"fixture")
                # Keep distinct per-architecture archives; FIXED must drop stale ones.
                for backend in ("910", "950"):
                    directory = prefix / "backends" / backend
                    directory.mkdir(exist_ok=True)
                    (directory / "libshmem_device.a").write_bytes(backend.encode())
                files = (
                    "include/shmem.h", "src/device/shmemi_device_meta.h",
                    "src/device_simt/shmemi_device_simt_meta.h",
                    "src/host_device/shmemi_common_types.h",
                    "src/host_device/shmem_device_symbol.h",
                )
                for relative in files:
                    target = prefix / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(ROOT / relative, target)
                config = prefix / "include/host_device/shmem_common_macros.h"
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_text((ROOT / "include/host_device/shmem_common_macros.h").read_text().replace(
                    "#define ACLSHMEM_DYNAMIC_META 0", f"#define ACLSHMEM_DYNAMIC_META {mode}"))
                # Exercise build.sh's actual per-backend staging function.
                if shutil.which("bash"):
                    helper = re.search(
                        r"(?ms)^function fn_copy_meta_archive\(\)\n\{.*?^\}",
                        (ROOT / "scripts/build.sh").read_text(encoding="utf-8"))
                    assert helper is not None
                    (base / "build").mkdir()
                    (prefix / "lib").mkdir()
                    script = base / "copy_meta_archive.sh"
                    script.write_text('#!/usr/bin/env bash\nset -e\n' + helper.group(0) +
                                      '\nfn_copy_meta_archive "$1"\n', encoding="utf-8")
                    script.chmod(0o700)
                    for backend in ("910", "950"):
                        (prefix / "lib/libshmem_device.a").write_bytes(backend.encode())
                        subprocess.run([str(script), f"../install/shmem/backends/{backend}"],
                                       cwd=base / "build", check=True)
                        archive = prefix / "backends" / backend / "libshmem_device.a"
                        assert archive.exists() == bool(mode)
                        if mode:
                            assert archive.read_bytes() == backend.encode()
                package = base / "src/python/shmem"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text("")
                with mock.patch("setuptools.setup") as setup:
                    namespace = runpy.run_path(str(ROOT / "setup.py"))
                settings = setup.call_args.kwargs
                dist = Distribution({key: settings[key] for key in ("packages", "package_dir", "package_data")})
                dist.script_name = "setup.py"
                command = namespace["BuildCppLibs"](dist)
                command.build_lib = str(base / "wheel-staging")
                command.ensure_finalized()
                command._copy_libraries_to_package()
                build_py.run(command)  # Real setuptools package-data expansion, no C++ build.
                output = base / "wheel-staging/shmem"
                for relative in files + ("include/host_device/shmem_common_macros.h",):
                    assert (output / relative).read_bytes() == (prefix / relative).read_bytes(), relative
                assert not list(output.rglob("*.cmake"))
                assert not list(output.rglob("shmem_meta_*.cpp"))
                for backend in ("910", "950"):
                    archive = output / "backends" / backend / "libshmem_device.a"
                    assert archive.exists() == bool(mode)
                    if mode:
                        assert archive.read_bytes() == backend.encode()
                if mode:
                    archive = prefix / "backends/950/libshmem_device.a"
                    archive.unlink()
                    try:
                        command._copy_libraries_to_package()
                    except RuntimeError as error:
                        assert "libshmem_device.a" in str(error)
                    else:
                        raise AssertionError("Missing ROOT archive was accepted")
                    archive.write_bytes(b"950")
                config.unlink()
                try:
                    command._copy_libraries_to_package()
                except RuntimeError as error:
                    assert "shmem_common_macros.h" in str(error)
                else:
                    raise AssertionError("Missing package config was accepted")
                print(f"WHEEL_META_STAGING_PASS mode={mode}", flush=True)
        check_build_mode(namespace["BuildCppLibs"])
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
