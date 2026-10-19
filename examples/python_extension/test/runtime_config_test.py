# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You should have received a copy of the License with this program. If not, please write to Huawei Technologies
# Co., Ltd., 41 Beihang Road, Shunyi District, Beijing 101100, People's Republic of China.
# -----------------------------------------------------------------------------------------------------------
"""Python regression tests for InitAttr.runtime_config."""
import os

import torch
import torch.distributed as dist
import torch_npu
import shmem as ash


G_IP_PORT = "tcp://127.0.0.1:8669"
LOCAL_MEM_SIZE = 1024 * 1024 * 1024
INIT_ATTR_VERSION = (2 << 16) + 184


def run_runtime_config_tests():
    pe = dist.get_rank()

    # Export checks: the package must expose the new bindings.
    if not hasattr(ash, "RuntimeConfig") or not hasattr(ash, "ConfigState"):
        raise ValueError("[ERROR] RuntimeConfig/ConfigState are not exported")

    # A fresh InitAttr value-initializes with the format-2 version and a DEFAULT runtime_config
    # (regression for the OptionalAttr/init plumbing carrying the header defaults).
    attributes = ash.InitAttr()
    if attributes.option_attr.version != INIT_ATTR_VERSION:
        raise ValueError(
            f"[ERROR] InitAttr.option_attr.version = {attributes.option_attr.version}, "
            f"expected {INIT_ATTR_VERSION}"
        )
    if attributes.runtime_config.udma_shared_jetty != ash.ConfigState.DEFAULT:
        raise ValueError("[ERROR] InitAttr.runtime_config.udma_shared_jetty is not DEFAULT")

    # A standalone OptionalAttr carries the same version (regression for the
    # uninitialized-version fix in its constructor).
    optional_attr = ash.OptionalAttr()
    if optional_attr.version != INIT_ATTR_VERSION:
        raise ValueError(
            f"[ERROR] OptionalAttr.version = {optional_attr.version}, expected {INIT_ATTR_VERSION}"
        )

    # Full init with an explicit DISABLED (safe on every runtime -- no capability needed).
    # The ENABLED capability branch is covered by the C++ unittests, which can probe
    # DlHcommApi::HasChannelConfigApis directly.
    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError(f"[ERROR] set_conf_store_tls failed: {ret}")
    attributes = ash.InitAttr()
    attributes.my_rank = pe
    attributes.n_ranks = dist.get_world_size()
    attributes.local_mem_size = LOCAL_MEM_SIZE
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    attributes.runtime_config.udma_shared_jetty = ash.ConfigState.DISABLED
    ret = ash.aclshmem_init(attributes)
    if ret != 0:
        raise ValueError(f"[ERROR] aclshmem_init failed: {ret}")
    ash.aclshmem_finalize()

    print(f"pe[{pe}]: runtime_config_test passed")


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=local_pe)
    run_runtime_config_tests()
    print("runtime_config_test.py running success!")
