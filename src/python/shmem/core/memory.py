#!/usr/bin/env python
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
import logging

import shmem._pyshmem as _pyshmem
from shmem.core.utils import (
    Buffer,
    MemType,
    AclshmemError,
    AclshmemInvalid,
    _SIZE_T_MAX,
    _current_instance_id,
    _instance_lock,
    _validate_buffer_instance,
)

__all__ = ['buffer', 'calloc', 'align', 'free', 'get_peer_buffer', 'Buffer', 'MemType']

logger = logging.getLogger("aclshmem")

def _validate_size(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AclshmemInvalid(f"{name} must be an integer.")
    if value <= 0:
        raise AclshmemInvalid(f"{name} must be greater than zero.")
    if value > _SIZE_T_MAX:
        raise AclshmemInvalid(f"{name} exceeds the native size_t range.")


def _validate_mem_type(mem_type: MemType) -> None:
    if not isinstance(mem_type, MemType):
        raise AclshmemInvalid("mem_type must be MemType.HOST_SIDE or MemType.DEVICE_SIDE.")


def buffer(size, release=False, except_on_del=True, mem_type: MemType=MemType.DEVICE_SIDE) -> Buffer:
    """
    Allocates an ACLSHMEM-backed npu buffer.

    Args:
        size (int): The size in bytes of the buffer to allocate.
        release: Reserved parameter. **Ignored** in ACLSHMEM. Always treated as ``False``.
        except_on_del: Reserved parameter. **Ignored** in ACLSHMEM. Always treated as ``True``.
        mem_type (MemType): Symmetric heap to allocate from. Defaults to ``MemType.DEVICE_SIDE``.

    Returns:
        Buffer: A raw memory buffer via its address and byte length.

    Raises:
        AclshmemError: If the buffer could not be allocated properly.
    """
    _validate_size(size, "size")
    _validate_mem_type(mem_type)

    with _instance_lock:
        instance_id = _current_instance_id()
        ptr = _pyshmem.aclshmemx_malloc(size, mem_type)
        if ptr == 0:
            raise AclshmemError("Allocate buffer failed.")
        return Buffer._owning(ptr, size, mem_type, instance_id)


def calloc(count: int, size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> Buffer:
    """Allocate and zero ``count * size`` bytes from a symmetric heap."""
    _validate_size(count, "count")
    _validate_size(size, "size")
    _validate_mem_type(mem_type)
    if count > _SIZE_T_MAX // size:
        raise AclshmemInvalid("count * size exceeds the native size_t range.")

    with _instance_lock:
        instance_id = _current_instance_id()
        ptr = _pyshmem.aclshmemx_calloc(count, size, mem_type)
        if ptr == 0:
            raise AclshmemError("Allocate zero-initialized buffer failed.")
        return Buffer._owning(ptr, count * size, mem_type, instance_id)


def align(alignment: int, size: int, mem_type: MemType=MemType.DEVICE_SIDE) -> Buffer:
    """Allocate an aligned buffer from a symmetric heap."""
    _validate_size(alignment, "alignment")
    _validate_size(size, "size")
    _validate_mem_type(mem_type)
    if alignment & (alignment - 1):
        raise AclshmemInvalid("alignment must be a power of two.")

    with _instance_lock:
        instance_id = _current_instance_id()
        ptr = _pyshmem.aclshmemx_align(alignment, size, mem_type)
        if ptr == 0:
            raise AclshmemError("Allocate aligned buffer failed.")
        return Buffer._owning(ptr, size, mem_type, instance_id)


def free(buf: Buffer) -> None:
    """
    Initiate one native free call for an owning ACLSHMEM Buffer.

    Args:
        buf (Buffer): The buffer to be freed.
    """

    with _instance_lock:
        if not isinstance(buf, Buffer):
            raise AclshmemInvalid("buf must be a Buffer.")
        if not buf.owned:
            raise AclshmemInvalid("Cannot free a non-owning Buffer.")
        if buf.release_called:
            raise AclshmemInvalid("Native free has already been called for this Buffer.")

        _validate_buffer_instance(buf, "buf")

        # Claim the one allowed free attempt while the GIL is still held.  The
        # native binding releases the GIL, so setting this afterward would allow a
        # second Python thread to pass the checks and free the same address again.
        # Do not roll this state back: native free returns void and only logs errors.
        buf._release_called = True
        _pyshmem.aclshmemx_free(buf.addr, buf.mem_type)


def get_peer_buffer(buf: Buffer, pe: int) -> Buffer:
    """
    Get address that may be used to directly reference dest on the specified PE.

    Args:
        buf (Buffer): The symmetric address of the remotely accessible data.
        pe (int): PE number

    Returns:
        Buffer: A remote symmetric address on the specified PE that can be accessed using memory loads and stores.

    Raises:
        AclshmemInvalid: If buf is invalid, released, or belongs to another instance, or pe is not a
            non-negative integer (bool is not accepted).
        AclshmemError: If the current instance context is unavailable or the native peer address lookup fails.
    """
    with _instance_lock:
        if not isinstance(buf, Buffer):
            raise AclshmemInvalid("buf must be a Buffer.")
        buf._ensure_usable()
        _validate_buffer_instance(buf, "buf")
        if not isinstance(pe, int) or isinstance(pe, bool) or pe < 0:
            raise AclshmemInvalid("pe must be a non-negative integer.")

        peer_addr = _pyshmem.aclshmem_ptr(buf.addr, pe)
        if peer_addr == 0:
            raise AclshmemError("Get the symmetric address on a specified PE failed.")

        return Buffer._borrowed(peer_addr, buf.length, buf.mem_type, buf)
