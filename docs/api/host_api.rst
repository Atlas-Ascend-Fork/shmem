HOST API
=================================

shmem_host_init.h
---------------------------------

.. doxygenfile:: shmem_host_init.h
    :project: ACLSHMEM_CPP_API

Synchronization timeout configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``attr.runtime_config.sync_timeout_us`` before SHMEM initialization.
The unit is microseconds. Zero means unspecified: disable timeout checks for the first instance, or
inherit the frozen value for subsequent instances.
The uint32_t field accepts explicit values in [1, ``ACLSHMEM_SYNC_TIMEOUT_MAX_US``]
(UINT32_MAX microseconds, approximately 71 minutes 35 seconds). A conflicting
explicit value while any instance is alive returns ``ACLSHMEM_NOT_SUPPORTED``.
Failed initialization leaves the effective configuration unchanged; finalizing
the last instance restores the disabled default. The setting covers collective synchronization
and software local barriers on supported HCCS/UB paths. It does not configure
RoCE synchronization timeout reporting or alter existing RoCE backend timeouts::

    aclshmemx_init_attr_t attr{};
    // Fill the remaining initialization attributes.
    attr.runtime_config.sync_timeout_us = 5000000; // 5 seconds
    int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);

shmem_log.h
---------------------------------

.. doxygenfile:: shmem_log.h
    :project: ACLSHMEM_CPP_API

shmem_host_heap.h
---------------------------------

.. doxygenfile:: shmem_host_heap.h
    :project: ACLSHMEM_CPP_API

shmem_host_rma.h
---------------------------------

.. doxygenfile:: shmem_host_rma.h
    :project: ACLSHMEM_CPP_API

shmem_host_cc.h
---------------------------------

.. doxygenfile:: shmem_host_cc.h
    :project: ACLSHMEM_CPP_API

shmem_host_p2p_sync.h
---------------------------------

.. doxygenfile:: shmem_host_p2p_sync.h
    :project: ACLSHMEM_CPP_API

shmem_host_so.h
---------------------------------

.. doxygenfile:: shmem_host_so.h
    :project: ACLSHMEM_CPP_API

shmem_host_team.h
---------------------------------

.. doxygenfile:: shmem_host_team.h
    :project: ACLSHMEM_CPP_API
