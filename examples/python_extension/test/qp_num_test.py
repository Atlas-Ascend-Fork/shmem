# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Python regression tests for aclshmemx_set_qp_num."""
import os
import re

import torch
import torch.distributed as dist
import torch_npu
import shmem as ash
from shmem._soc import select_backend


G_IP_PORT = "tcp://127.0.0.1:8669"
LOCAL_MEM_SIZE = 1024 * 1024 * 1024
MAX_QP_NUM = 56
MAX_AIV_PER_NPU = 72


def assert_nonzero(ret, description):
    if ret == 0:
        raise ValueError(f"[ERROR] {description} unexpectedly succeeded")


def has_rdma_v2_backend():
    """Match the runtime backend detection used by rdma_perftest/run.sh."""
    drivers = os.environ.get("IBV_EXTEND_DRIVERS", "")
    return bool(
        re.search(r"xscale|libxscale_nda\.so", drivers, re.IGNORECASE)
        or re.search(r"hns.*1825|libhrn5-rdma", drivers, re.IGNORECASE)
    )


def run_qp_num_tests():
    pe = dist.get_rank()

    # This also verifies the package-level lazy export and pybind symbol.
    if not callable(ash.aclshmemx_set_qp_num):
        raise ValueError("[ERROR] aclshmemx_set_qp_num is not exported")

    is_ascend950 = select_backend() == "950"
    rdma_v2_enabled = is_ascend950 and has_rdma_v2_backend()

    # Unsupported engines and values must be rejected before initialization.
    assert_nonzero(ash.aclshmemx_set_qp_num(ash.OpEngineType.MTE, 1), "MTE QP configuration")
    # SDMA is supported with range [1, min(vector-core count, ACLSHMEM_MAX_AIV_PER_NPU)].
    assert_nonzero(ash.aclshmemx_set_qp_num(ash.OpEngineType.SDMA, 0), "zero SDMA QP configuration")
    assert_nonzero(
        ash.aclshmemx_set_qp_num(ash.OpEngineType.SDMA, MAX_AIV_PER_NPU + 1),
        "out-of-range SDMA QP configuration",
    )
    ret = ash.aclshmemx_set_qp_num(ash.OpEngineType.SDMA, 1)
    if ret != 0:
        raise ValueError(f"[ERROR] valid SDMA qp_num=1 failed: {ret}")
    supported_engine = None
    if is_ascend950:
        assert_nonzero(ash.aclshmemx_set_qp_num(ash.OpEngineType.UDMA, 0), "zero QP configuration")
        assert_nonzero(
            ash.aclshmemx_set_qp_num(ash.OpEngineType.UDMA, MAX_QP_NUM + 1),
            "out-of-range QP configuration",
        )
        for qp_num in (1, MAX_QP_NUM):
            ret = ash.aclshmemx_set_qp_num(ash.OpEngineType.UDMA, qp_num)
            if ret != 0:
                raise ValueError(f"[ERROR] valid UDMA qp_num={qp_num} failed: {ret}")
        supported_engine = ash.OpEngineType.UDMA
    else:
        print(f"pe[{pe}]: [SKIP] UDMA QP tests require Ascend950")

    if rdma_v2_enabled:
        for qp_num in (1, MAX_QP_NUM):
            ret = ash.aclshmemx_set_qp_num(ash.OpEngineType.ROCE, qp_num)
            if ret != 0:
                raise ValueError(f"[ERROR] valid ROCE qp_num={qp_num} failed: {ret}")
        supported_engine = ash.OpEngineType.ROCE
    else:
        assert_nonzero(
            ash.aclshmemx_set_qp_num(ash.OpEngineType.ROCE, 1),
            "ROCE QP configuration without an Ascend950 RDMA v2 backend",
        )

    if supported_engine is None:
        print(f"pe[{pe}]: qp_num_test passed")
        return

    # Use MTE for initialization so this regression test does not require a
    # UDMA data transfer; the process-wide QP setting is frozen nevertheless.
    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError(f"[ERROR] set_conf_store_tls failed: {ret}")
    attributes = ash.InitAttr()
    attributes.my_rank = pe
    attributes.n_ranks = dist.get_world_size()
    attributes.local_mem_size = LOCAL_MEM_SIZE
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    ret = ash.aclshmem_init(attributes)
    if ret != 0:
        raise ValueError(f"[ERROR] aclshmem_init failed: {ret}")

    try:
        assert_nonzero(
            ash.aclshmemx_set_qp_num(supported_engine, 1),
            "QP configuration after initialization",
        )
    finally:
        ash.aclshmem_finalize()

    print(f"pe[{pe}]: qp_num_test passed")


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=local_pe)
    run_qp_num_tests()
    print("qp_num_test.py running success!")
