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
import ctypes
import sys
import os
import socket
import threading

import shmem._pyshmem as _pyshmem

MemType = _pyshmem.MemType


class _NativeThreadFormatter(logging.Formatter):
    def format(self, record):
        record.native_tid = threading.get_native_id()
        return super().format(record)


def setup_aclshmem_logger(log_level: str="ERROR", log_file_path: str=None) -> None:
    """
    Initialize and configure the logger for the ACLSHMEM Python interface.

    The log format includes host, process ID, thread ID, and PE rank (if available):
        <host>:<pid>:<tid> [<PE>] ACLSHMEM <LEVEL> : <message>

    Args:
        log_level (str): Desired logging level. One of 'DEBUG', 'INFO', 'WARNING',
                         'ERROR', or 'CRITICAL'. Defaults to 'ERROR'.
        log_file_path (str, optional): If specified, logs will also be written to this file.
    """
    try:
        pe_rank = _pyshmem.my_pe()
    except Exception as e:
        print("Unable to retrieve PE rank. Ensure ACLSHMEM is initialized.")
        pe_rank = -1

    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Build log prefix
    host = socket.getfqdn().split(".")[0]
    pid = os.getpid()
    log_format = f"{host}:{pid}:%(native_tid)s [{pe_rank}] ACLSHMEM %(levelname)s : %(message)s"
    formatter = _NativeThreadFormatter(log_format)

    # Configure console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Set up dedicated logger (not root)
    logger = logging.getLogger("aclshmem")
    logger.setLevel(numeric_level)
    logger.handlers.clear()  # Remove existing handlers to avoid duplication
    logger.addHandler(console_handler)

    # Optional file logging
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

class AclshmemError(Exception):
    def __init__(self,  msg):
        super().__init__(msg)
        self.msg = msg

    def __repr__(self):
        return f"<AclshmemError: {self.msg}>"


class AclshmemInvalid(Exception):
    def __init__(self,  msg):
        super().__init__(msg)
        self.msg = msg

    def __repr__(self):
        return f"<AclshmemInvalid: {self.msg}>"


_MAX_INSTANCE_COUNT = 255
_INTPTR_MAX = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8 - 1)) - 1
_SIZE_T_MAX = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1
_instance_lock = threading.RLock()


def _validate_instance_id(instance_id: int) -> None:
    if (
        not isinstance(instance_id, int)
        or isinstance(instance_id, bool)
        or instance_id < 0
        or instance_id >= _MAX_INSTANCE_COUNT
    ):
        raise AclshmemInvalid(
            f"instance_id must be an integer in range [0, {_MAX_INSTANCE_COUNT - 1}]."
        )


def _validate_stream(stream, *, allow_none: bool) -> int:
    if stream is None:
        if allow_none:
            return 0
        raise AclshmemInvalid("stream must be a non-negative integer.")
    if not isinstance(stream, int) or isinstance(stream, bool) or stream < 0:
        raise AclshmemInvalid("stream must be a non-negative integer.")
    return stream


def _current_instance_id() -> int:
    ctx = _pyshmem.aclshmemx_instance_ctx_get()
    if ctx is None:
        raise AclshmemError("Get current ACLSHMEM instance context failed.")
    return ctx.id


def _validate_buffer_instance(buf, name: str) -> None:
    if buf.instance_id is None:
        return
    current_instance_id = _current_instance_id()
    if current_instance_id != buf.instance_id:
        raise AclshmemInvalid(
            f"{name} belongs to instance {buf.instance_id}, but current instance is "
            f"{current_instance_id}."
        )


class Buffer:
    """
    A lightweight wrapper representing a memory buffer by its address and size.

    Attributes:
        addr (int): Memory address (e.g., from CuPy, PyTorch, or CUDA driver).
        length (int): Size of the buffer in bytes.
        mem_type (MemType): Symmetric heap associated with the address.
        owned (bool): Whether this object may pass the address to ``free``.
            Direct construction describes an external address and is always
            non-owning. Allocation factories create owning buffers internally.
        instance_id (int or None): Instance that allocated an owning buffer.
            External descriptors have no allocation instance; borrowed views
            inherit the instance of their owner.
        release_called (bool): Whether a native ``free`` attempt has been
            initiated for this buffer or its owning buffer. This is not proof
            that the native deallocation succeeded because the API returns
            ``void``.
    """
    __slots__ = ('addr', 'length', 'mem_type', '_owned', '_release_called', '_owner', '_instance_id')

    def __init__(
        self,
        addr: int,
        length: int,
        mem_type: MemType=MemType.DEVICE_SIDE,
    ):
        self._initialize(addr, length, mem_type, owned=False, owner=None, instance_id=None)

    def _initialize(self, addr, length, mem_type, *, owned, owner, instance_id) -> None:
        if not isinstance(addr, int) or isinstance(addr, bool):
            raise AclshmemInvalid("addr must be an integer.")
        if addr <= 0 or addr > _INTPTR_MAX:
            raise AclshmemInvalid(f"addr must satisfy 0 < addr <= {_INTPTR_MAX}.")
        if not isinstance(length, int) or isinstance(length, bool):
            raise AclshmemInvalid("length must be an integer.")
        if length <= 0 or length > _SIZE_T_MAX:
            raise AclshmemInvalid(f"length must satisfy 0 < length <= {_SIZE_T_MAX}.")
        if not isinstance(mem_type, MemType):
            raise TypeError("mem_type must be an instance of MemType")
        if not isinstance(owned, bool):
            raise TypeError("owned must be a bool")
        if owner is not None and not isinstance(owner, Buffer):
            raise TypeError("owner must be a Buffer or None")
        if owned and owner is not None:
            raise ValueError("An owned Buffer cannot also be a borrowed view")
        if owned:
            _validate_instance_id(instance_id)
        elif owner is not None:
            instance_id = owner.instance_id
        elif instance_id is not None:
            raise ValueError("An external Buffer cannot declare an allocation instance")
        self.addr = addr
        self.length = length
        self.mem_type = mem_type
        self._owned = owned
        self._release_called = False
        self._owner = owner
        self._instance_id = instance_id

    @property
    def owned(self) -> bool:
        """Whether this object owns the native symmetric allocation."""
        return self._owned

    @property
    def instance_id(self):
        """Instance that owns the allocation, or ``None`` for external addresses."""
        return self._instance_id

    @property
    def release_called(self) -> bool:
        """
        Whether a native ``free`` attempt has been initiated for this allocation.

        Borrowed peer/view buffers inherit the state of their owning buffer.
        The state is claimed before entering the native API and is never rolled
        back; it cannot prove that the runtime released the allocation.
        """
        if self._release_called:
            return True
        return self._owner is not None and self._owner.release_called

    def _ensure_usable(self) -> None:
        if self.release_called:
            raise AclshmemInvalid("Buffer is no longer usable after free was called.")

    @classmethod
    def _owning(cls, addr: int, length: int, mem_type: MemType, instance_id: int):
        """Create an owning buffer for an allocation factory."""
        obj = cls.__new__(cls)
        obj._initialize(addr, length, mem_type, owned=True, owner=None, instance_id=instance_id)
        return obj

    @classmethod
    def _borrowed(cls, addr: int, length: int, mem_type: MemType, owner):
        """Create a non-owning peer/view buffer tied to an owning Buffer."""
        obj = cls.__new__(cls)
        obj._initialize(addr, length, mem_type, owned=False, owner=owner, instance_id=owner.instance_id)
        return obj

    def __repr__(self):
        return (
            f"Buffer(addr=0x{self.addr:x}, length={self.length}, "
            f"mem_type={self.mem_type}, owned={self.owned}, "
            f"instance_id={self.instance_id}, release_called={self.release_called})"
        )

    def __eq__(self, other):
        if not isinstance(other, Buffer):
            return False
        return (
            self.addr == other.addr
            and self.length == other.length
            and self.mem_type == other.mem_type
        )
