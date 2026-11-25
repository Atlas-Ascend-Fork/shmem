DEVICE API
=================================

shmem_device_amo.h
---------------------------------

.. doxygenfile:: shmem_device_amo.h
    :project: ACLSHMEM_CPP_API

shmem_device_cc.h
---------------------------------

.. doxygenfile:: shmem_device_cc.h
    :project: ACLSHMEM_CPP_API

Collective synchronization timeout reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The synchronization interfaces ``aclshmem_sync``, ``aclshmem_sync_all``,
``aclshmemx_sync_vec`` and ``aclshmemx_sync_vec_all`` detect timeouts in
software signal polling on supported HCCS/UB paths, including software local
barriers, when configured. RoCE synchronization timeout reporting is not
covered by this setting.

Timeout detection is **disabled by default**. Set a positive initialization attribute
``runtime_config.sync_timeout_us`` to configure the threshold in microseconds::

    aclshmemx_init_attr_t attr{};
    // Fill the remaining initialization attributes.
    attr.runtime_config.sync_timeout_us = 5000000; // 5 seconds
    int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);

Zero means unspecified: disable timeout checks before the first successful initialization,
or inherit the effective process value for later instances. Explicit thresholds
must fit in uint32_t: [1, ``ACLSHMEM_SYNC_TIMEOUT_MAX_US``] (UINT32_MAX, approximately
71 minutes 35 seconds). While any instance is
alive, a conflicting explicit threshold returns ``ACLSHMEM_NOT_SUPPORTED``.
Failed initialization does not publish a new value. Finalizing the last instance
restores the disabled default. Set the attribute on every PE and check initialization errors.
The threshold covers the synchronization interfaces listed above and software local
barriers. With timeout checks disabled, the original signal polling loop is used,
without failure counters or clock reads. The timeout mode is selected once on entry;
software local barriers inside the sync inherit that selection.
Disabled software waits have no timeout: they continue until the required
signal arrives. This does not disable existing RoCE backend timeouts or errors.
Hardware waits retain their existing behavior.
Rebuild the library and device kernels together.
When enabled, each core records its own start time on sync entry and shares it across phases. Time is checked every
1024 unsuccessful polls; the internal batch size is not printed. Signal
polling accepts expected or expected + 1 using uint32 arithmetic, before
checking for timeout. The threshold is not a guaranteed upper bound on
detection or host notification latency.

On timeout, the device prints ``SHMEM_SYNC_TIMEOUT`` and executes ``trap()``.
The sync does not return success. The host must check stream/device
synchronization errors and terminate or recreate the affected communication
context rather than resume using the failed state. Runtime error codes and
delivery times depend on the environment. Reporting is independent of the
SHMEM debug logging macro.

.. list-table:: Timeout diagnostic fields
    :header-rows: 1
    :widths: 25 75

    * - Field
      - Meaning
    * - ``instance, team, pe``
      - Instance ID, team handle and reporting world PE within that instance. Non-MULTI_INSTANCE builds report instance 0.
    * - ``core_type, core``
      - 1 = AIV, 2 = AIC, 0 = unknown; core is the logical GetBlockIdx() value, not a physical core ID.
    * - ``algorithm``
      - Internal synchronization diagnostic identifier. A standalone software local barrier reports 0; local barriers within a collective retain its parent identifier.
    * - ``phase``
      - 0 = entry local barrier, 1 = synchronization generation publication barrier, 2 = end-of-phase local barrier, 3 = send, 4 = signal wait.
    * - ``generation, round, slot``
      - Team synchronization generation, internal round (starting at 0) and signal-pool slot.
    * - ``peer``
      - World PE being waited for or sent to.
    * - ``expected, observed``
      - Expected and observed signal values.
    * - ``elapsed_us, timeout_us``
      - Time since this core entered the sync and the configured threshold, in microseconds.
    * - ``wait_core, local_round``
      - Logical AIV being waited for and local dissemination round in a software local barrier. The local barrier reports its own world PE in peer. Generation may be unknown before it is published. Round identifies an internal synchronization step.

Unavailable or inapplicable fields are -1.
The following log is illustrative, not an actual fault-test record.
PE0/AIV1 is waiting for PE1, whose signal is still 0 instead of 1.

.. code-block:: text

    SHMEM_SYNC_TIMEOUT instance=0 team=0 pe=0 core_type=1 core=1 algorithm=3 phase=4 generation=1 round=0 slot=0 peer=1 wait_core=-1 local_round=-1 expected=1 observed=0 elapsed_us=60000508 timeout_us=60000000

.. important::

    Time checks require software polling to continue. They cannot interrupt
    blocked memory accesses, hardware event waits or hardware
    ``AscendC::SyncAll`` barriers. Software local barrier tests do not
    establish timeout coverage for hardware local barriers. RoCE synchronization
    timeout reporting is not supported by this setting; RoCE waits and backend
    error handling retain their existing behavior. Standalone quiet and ordinary
    point-to-point waits also retain their existing behavior. Timeout detection
    applies only to the software polling paths covered by this mechanism.

shmem_device_mo.h
---------------------------------

.. doxygenfile:: shmem_device_mo.h
    :project: ACLSHMEM_CPP_API

shmem_device_p2p_sync.h
---------------------------------

.. doxygenfile:: shmem_device_p2p_sync.h
    :project: ACLSHMEM_CPP_API

shmem_device_rma.h
---------------------------------

.. doxygenfile:: gm2gm/shmem_device_rma.h
    :project: ACLSHMEM_CPP_API

.. doxygenfile:: ub2gm/shmem_device_rma.h
    :project: ACLSHMEM_CPP_API

.. doxygenfile:: ub2gm/engine/shmem_device_mte.h
    :project: ACLSHMEM_CPP_API

.. doxygenfile:: gm2gm/engine/shmem_device_mte.h
    :project: ACLSHMEM_CPP_API

.. doxygenfile:: gm2gm/engine/shmem_device_rdma.h
    :project: ACLSHMEM_CPP_API

.. doxygenfile:: gm2gm/engine/shmem_device_sdma.h
    :project: ACLSHMEM_CPP_API

UDMA put interfaces transfer at most 256 MB
(256 * 1024 * 1024 bytes) per asynchronous put request. The
``elem_size`` parameter is an element count, so the transferred byte
size is ``elem_size * sizeof(T)``. For larger data, split the operation
into multiple UDMA put requests whose byte size does not exceed 256 MB,
then use the matching completion or synchronization interface, such as
``aclshmemx_udma_quiet(pe)`` or the documented signal protocol, before
reading destination data or reusing buffers whose contents must remain
stable.

.. doxygenfile:: gm2gm/engine/shmem_device_udma.h
    :project: ACLSHMEM_CPP_API

shmem_device_so.h
---------------------------------

.. doxygenfile:: shmem_device_so.h
    :project: ACLSHMEM_CPP_API

shmem_device_team.h
---------------------------------

.. doxygenfile:: shmem_device_team.h
    :project: ACLSHMEM_CPP_API
