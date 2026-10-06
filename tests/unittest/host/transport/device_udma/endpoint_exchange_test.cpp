/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <limits>

// Test-only access override scoped strictly to dl_hcomm_api.h so this fixture can replace
// dynamically loaded HCOMM entry points with local recorders. Keep the undef immediately
// after this include; do not place other headers inside this scope or copy this pattern to
// other tests. Prefer an explicit injection interface if production code adds one later.
#define private public
#include "dl_hcomm_api.h"
#undef private

#include "device_udma_transport_manager.h"
#include "shmemi_host_common.h"

namespace shm {
namespace transport {
namespace device {

class UdmaEndpointExchangeTest : public testing::Test {
protected:
    using Descriptor = UdmaTransportManager::ExchangedEndpointDesc;
    using Exchange = UdmaTransportManager::EndpointExchange;

    void SetUp() override
    {
        saved_handle_ = g_boot_handle;
        saved_mem_reg_ = DlHcommApi::gHcommMemReg;
        saved_mem_unreg_ = DlHcommApi::gHcommMemUnreg;
        saved_endpoint_destroy_ = DlHcommApi::gHcommEndpointDestroy;
        saved_channel_create_ = DlHcommApi::gHcommChannelCreate;
        saved_channel_destroy_ = DlHcommApi::gHcommChannelDestroy;
        saved_config_create_ = DlHcommApi::gHcommChannelConfigCreate;
        g_boot_handle.npes = 3;
        g_boot_handle.allgather = AllGather;
        lengths_.clear();
        fail_call_ = 0;
        mem_reg_endpoints_.clear();
        channel_endpoints_.clear();
        channel_names_.clear();
        config_mismatch_ = 0;
        mask_matrix_.clear();
        next_handle_ = 1000;
        DlHcommApi::gHcommMemReg = RecordMemReg;
        DlHcommApi::gHcommMemUnreg = SucceedMemUnreg;
        DlHcommApi::gHcommEndpointDestroy = SucceedEndpointDestroy;
        DlHcommApi::gHcommChannelCreate = RecordChannelCreate;
        DlHcommApi::gHcommChannelDestroy = SucceedChannelDestroy;
        // Force the recorded fallback even if an earlier test loaded libhcomm.
        DlHcommApi::gHcommChannelConfigCreate = nullptr;
    }

    void TearDown() override
    {
        DlHcommApi::gHcommMemReg = saved_mem_reg_;
        DlHcommApi::gHcommMemUnreg = saved_mem_unreg_;
        DlHcommApi::gHcommEndpointDestroy = saved_endpoint_destroy_;
        DlHcommApi::gHcommChannelCreate = saved_channel_create_;
        DlHcommApi::gHcommChannelDestroy = saved_channel_destroy_;
        DlHcommApi::gHcommChannelConfigCreate = saved_config_create_;
        g_boot_handle = saved_handle_;
    }

    static int AllGather(const void* send, void* recv, int len, aclshmemi_bootstrap_handle_t* handle)
    {
        lengths_.push_back(len);
        if (lengths_.size() == fail_call_) {
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
        const size_t count = static_cast<size_t>(len) / sizeof(Descriptor);
        const auto* input = static_cast<const Descriptor*>(send);
        auto* output = static_cast<Descriptor*>(recv);
        for (int rank = 0; rank < handle->npes; ++rank) {
            for (size_t idx = 0; idx < count; ++idx) {
                output[rank * count + idx] = input[idx];
                output[rank * count + idx].peer_rank = static_cast<uint32_t>(rank);
            }
        }
        return ACLSHMEM_SUCCESS;
    }

    static int MirrorAllGather(const void* send, void* recv, int len, aclshmemi_bootstrap_handle_t* handle)
    {
        for (int rank = 0; rank < handle->npes; ++rank) {
            std::memcpy(static_cast<uint8_t*>(recv) + static_cast<size_t>(rank) * len, send, len);
        }
        return ACLSHMEM_SUCCESS;
    }

    static Result Gather(uint32_t chunk_count, Exchange& exchange)
    {
        exchange.max_count = 5;
        std::vector<Descriptor> local(exchange.max_count);
        for (uint32_t idx = 0; idx < exchange.max_count; ++idx) {
            local[idx].eid_index = idx;
            local[idx].valid = idx < 4 ? 1 : 0;
            local[idx].listen_port = static_cast<uint16_t>(100 + idx);
        }
        return UdmaTransportManager::GatherEndpointChunks(local, 3, chunk_count, exchange);
    }

    static uint32_t ChunkCount(uint32_t ranks) { return UdmaTransportManager::GetEndpointChunkCount(ranks); }

    static Result CheckQpOwners(
        const std::map<std::pair<uint32_t, uint32_t>, size_t>& bindings, bool shared = true, uint32_t qp_num = 16)
    {
        UdmaTransportManager manager;
        manager.shared_jetty_enabled_ = shared;
        manager.qp_num_ = qp_num;
        return manager.CheckPerQpEndpointSplit(bindings);
    }

    static bool GetSize(uint32_t ranks, uint32_t endpoints, size_t& count)
    {
        return UdmaTransportManager::GetEndpointExchangeSize(ranks, endpoints, count);
    }

    static uint32_t MaxEndpoints()
    {
        return std::numeric_limits<int>::max() / sizeof(UdmaTransportManager::ExchangedEndpointDesc);
    }

    static EndpointHandle Handle(uintptr_t value) { return reinterpret_cast<EndpointHandle>(value); }

    static HcommResult RecordMemReg(EndpointHandle endpoint, const char*, CommMem*, HcommMemHandle* mem_handle)
    {
        mem_reg_endpoints_.push_back(endpoint);
        *mem_handle = reinterpret_cast<HcommMemHandle>(next_handle_++);
        return 0;
    }

    static HcommResult SucceedMemUnreg(EndpointHandle, HcommMemHandle) { return 0; }
    static HcommResult SucceedEndpointDestroy(EndpointHandle) { return 0; }
    static HcommResult SucceedChannelDestroy(const ChannelHandle*, uint32_t) { return 0; }

    static HcommResult RecordChannelCreate(
        EndpointHandle endpoint, CommEngine, HcommChannelDesc* descs, uint32_t count, ChannelHandle* channels)
    {
        for (uint32_t idx = 0; idx < count; ++idx) {
            channel_endpoints_.push_back(endpoint);
            channel_names_.emplace_back(descs[idx].channelName == nullptr ? "" : descs[idx].channelName);
            channels[idx] = next_handle_++;
        }
        return 0;
    }

    static void AddEndpoint(
        UdmaTransportManager& manager, uint32_t eid, uint32_t peer, uintptr_t handle, uint32_t qp_idx = 0)
    {
        const UdmaTransportManager::EndpointMapKey key{
            eid, manager.EndpointPeerRank(peer), manager.EndpointQpSlot(qp_idx)};
        EndpointDesc desc{};
        manager.endpoint_handle_map_[key] = Handle(handle);
        manager.endpoint_desc_map_[key] = desc;
        manager.endpoint_listen_port_map_[key] = static_cast<uint16_t>(1000 + handle);
    }

    static void EnableSharedJetty(UdmaTransportManager& manager) { manager.shared_jetty_enabled_ = true; }
    static void SelectQps(UdmaTransportManager& manager, const std::vector<uint64_t>& masks)
    {
        manager.peer_qp_masks_ = masks;
    }

    static size_t RegisteredEndpointCount(const UdmaTransportManager& manager, uint64_t addr)
    {
        const auto region = manager.mem_record_map_.find(addr);
        return region == manager.mem_record_map_.end() ? 0U : region->second.size();
    }

    static size_t LogicalEndpointCount(const UdmaTransportManager& manager)
    {
        return manager.endpoint_desc_map_.size();
    }

    static void ConfigureManager(UdmaTransportManager& manager, uint32_t rank, uint32_t ranks, uint32_t qp_num)
    {
        manager.rank_id_ = rank;
        manager.rank_count_ = ranks;
        manager.qp_num_ = qp_num;
    }

    static void AddRoute(
        UdmaTransportManager& manager, uint32_t peer, uint32_t local_eid, uint32_t remote_eid, uint32_t net_layer)
    {
        manager.peer_routes_[peer].qp_routes.push_back(UdmaTransportManager::QpRoute{
            local_eid, remote_eid, NetType::TopoFileDesc, net_layer, "instance", "plane", 1});
    }

    static void AddRemoteDescriptor(
        Exchange& exchange, uint32_t rank, uint32_t index, uint32_t local_rank, uint32_t eid, uint32_t qp_slot = 0)
    {
        Descriptor& descriptor = exchange.descs[static_cast<size_t>(rank) * exchange.max_count + index];
        descriptor.valid = 1;
        descriptor.peer_rank = local_rank;
        descriptor.eid_index = eid;
        descriptor.qp_slot = qp_slot;
        descriptor.listen_port = static_cast<uint16_t>(2000 + eid);
    }

    static Result BuildDirect(
        UdmaTransportManager& manager, const Exchange& exchange, std::vector<uint32_t>& slots,
        std::vector<uint32_t>& dst_pes, std::vector<uint32_t>& qp_indices)
    {
        UdmaTransportManager::ChannelBuildState state;
        state.channel_names.reserve(static_cast<size_t>(manager.rank_count_ - 1) * manager.qp_num_);
        const Result result = manager.BuildDirectChannels(exchange, state);
        slots = state.slots;
        dst_pes = state.dst_pes;
        qp_indices = state.qp_indices;
        return result;
    }

    static bool BuildSyncEndpointsForTest(
        UdmaTransportManager& manager, const RootInfo& root_info, const TopoInfo& topo_info, uint32_t rank_count,
        std::vector<std::vector<SyncEndpoint>>& endpoints)
    {
        return manager.BuildSyncEndpoints(root_info, topo_info, rank_count, endpoints);
    }

    static int ConfigAllGather(const void* send, void* recv, int len, aclshmemi_bootstrap_handle_t* handle)
    {
        lengths_.push_back(len);
        if (lengths_.size() == fail_call_)
            return ACLSHMEM_BOOTSTRAP_ERROR;
        if (lengths_.size() == 1) {
            // Count, QP count, and sparse-mode flag form the first collective.
            EXPECT_EQ(len, 3 * sizeof(uint32_t));
            MirrorAllGather(send, recv, len, handle);
            if (config_mismatch_ != 0)
                static_cast<uint32_t*>(recv)[3 + config_mismatch_] ^= 1;
        } else if (lengths_.size() == 2 && !mask_matrix_.empty()) {
            EXPECT_EQ(mask_matrix_.size() * sizeof(uint64_t), static_cast<size_t>(len) * handle->npes);
            std::memcpy(recv, mask_matrix_.data(), mask_matrix_.size() * sizeof(uint64_t));
        } else {
            MirrorAllGather(send, recv, len, handle);
        }
        return ACLSHMEM_SUCCESS;
    }

    static Result ExchangeDescriptors(UdmaTransportManager& manager, Exchange& exchange)
    {
        g_boot_handle.npes = manager.rank_count_;
        g_boot_handle.allgather = ConfigAllGather;
        return manager.ExchangeEndpointDescriptors(exchange);
    }

    static uint64_t TestMask(uint32_t a, uint32_t b)
    {
        return a == b ? 0 : (uint64_t{1} | (uint64_t{1} << (1 + (a + b) % 31)));
    }

    aclshmemi_bootstrap_handle_t saved_handle_{};
    HcommMemRegFunc saved_mem_reg_{nullptr};
    HcommMemUnregFunc saved_mem_unreg_{nullptr};
    HcommEndpointDestroyFunc saved_endpoint_destroy_{nullptr};
    HcommChannelCreateFunc saved_channel_create_{nullptr};
    HcommChannelDestroyFunc saved_channel_destroy_{nullptr};
    HcommChannelConfigCreateFunc saved_config_create_{nullptr};
    inline static std::vector<int> lengths_;
    inline static size_t fail_call_{0};
    inline static std::vector<EndpointHandle> mem_reg_endpoints_;
    inline static std::vector<EndpointHandle> channel_endpoints_;
    inline static std::vector<std::string> channel_names_;
    inline static std::vector<uint64_t> mask_matrix_;
    inline static uint32_t config_mismatch_{0};
    inline static uintptr_t next_handle_{1000};
};

TEST_F(UdmaEndpointExchangeTest, AcceptsBoundaryAndRejectsNextEndpoint)
{
    constexpr uint32_t ranks = 1;
    const uint32_t endpoints = MaxEndpoints();
    size_t count = 0;
    ASSERT_TRUE(GetSize(ranks, endpoints, count));
    EXPECT_EQ(count, static_cast<size_t>(ranks) * endpoints);
    EXPECT_FALSE(GetSize(ranks, endpoints + 1, count));
    EXPECT_EQ(count, 0U);
}

TEST_F(UdmaEndpointExchangeTest, AllowsSameQpAcrossPeersOnState112)
{
    EXPECT_EQ(CheckQpOwners({{{8, 0}, 112}, {{9, 0}, 112}, {{10, 0}, 112}}), ACLSHMEM_SUCCESS);
}

TEST_F(UdmaEndpointExchangeTest, AllowsSharedDualQpOwnersAndIndependentMeshQueues)
{
    std::map<std::pair<uint32_t, uint32_t>, size_t> bindings;
    for (uint32_t qp = 0; qp < 16; ++qp) {
        bindings[{1, qp}] = qp; // MESH route stays independent of the cross route.
        for (uint32_t peer = 8; peer < 16; ++peer) {
            bindings[{peer, qp}] = 112 + qp;
        }
    }
    EXPECT_EQ(CheckQpOwners(bindings), ACLSHMEM_SUCCESS);
}

TEST_F(UdmaEndpointExchangeTest, RejectsDifferentQpsSharingWithinOnePeer)
{
    if (ACLSHMEM_UDMA_RELAY_ENABLED) {
        GTEST_SKIP() << "Multi-QP shared Jetty is direct-only";
    }
    EXPECT_EQ(CheckQpOwners({{{9, 0}, 112}, {{9, 1}, 112}}), ACLSHMEM_INNER_ERROR);
}

TEST_F(UdmaEndpointExchangeTest, RejectsDifferentQpsAfterValidCrossPeerSharing)
{
    if (ACLSHMEM_UDMA_RELAY_ENABLED) {
        GTEST_SKIP() << "Multi-QP shared Jetty is direct-only";
    }
    EXPECT_EQ(CheckQpOwners({{{8, 0}, 112}, {{9, 0}, 112}, {{10, 2}, 112}}), ACLSHMEM_INNER_ERROR);
}

TEST_F(UdmaEndpointExchangeTest, SharedOwnerCheckLeavesNonSharedModeUnchanged)
{
    EXPECT_EQ(CheckQpOwners({{{8, 0}, 112}, {{9, 1}, 112}}, false), ACLSHMEM_SUCCESS);
    EXPECT_EQ(CheckQpOwners({{{8, 0}, 112}, {{9, 0}, 112}}, true, 1), ACLSHMEM_SUCCESS);
    EXPECT_EQ(CheckQpOwners({}), ACLSHMEM_SUCCESS);
}

TEST_F(UdmaEndpointExchangeTest, KeepsExplicitClosFabricWhenPortsAlsoMatchMeshTopology)
{
    // PoD may expose an aggregated CLOS EID whose ports also occur in the
    // Mesh topology. The rootinfo CLOS declaration must take precedence;
    // only TOPO_FILE_DESC endpoints infer their fabric from topology ports.
    g_boot_handle.npes = 1;
    g_boot_handle.allgather = MirrorAllGather;

    RootInfo root_info;
    auto level = std::make_shared<LevelInfo>(LevelInfo{1, "clos-domain", NetType::Clos, ""});
    auto rank_addr = std::make_shared<RankAddr>();
    rank_addr->levelInfo = level;
    rank_addr->ports = {"shared-port"};
    root_info.eidIndexToRankAddr.emplace(7, rank_addr);

    TopoInfo topo_info;
    topo_info.meshTopoEdges.push_back({0, 1, {"shared-port"}, {"peer-mesh-port"}});
    topo_info.closTopoEdges.push_back({1, 0, 0, {"shared-port"}});

    UdmaTransportManager manager;
    std::vector<std::vector<SyncEndpoint>> endpoints;
    ASSERT_TRUE(BuildSyncEndpointsForTest(manager, root_info, topo_info, 1, endpoints));
    ASSERT_EQ(endpoints.size(), 1U);
    ASSERT_EQ(endpoints[0].size(), 1U);
    EXPECT_EQ(endpoints[0][0].netType, NetType::Clos);
    EXPECT_EQ(endpoints[0][0].fabric, TopoFabric::Clos);
}

TEST_F(UdmaEndpointExchangeTest, PortNamesAreScopedToLocalDeviceForBothTopologyEdgeEnds)
{
    g_boot_handle.npes = 1;
    g_boot_handle.allgather = MirrorAllGather;
    // Same label is CLOS on device 0, Mesh on device 4, and absent on device 9.
    TopoInfo topo;
    topo.meshTopoEdges.push_back({2, 4, {"0/6"}, {"1/0"}});
    topo.closTopoEdges.push_back({1, 0, 0, {"1/0"}});
    for (auto net : {NetType::TopoFileDesc, NetType::Clos, NetType::Mesh}) {
        for (uint32_t local : {0U, 4U, 2U, 9U}) {
            RootInfo root;
            root.localId = local;
            auto addr = std::make_shared<RankAddr>();
            addr->levelInfo = std::make_shared<LevelInfo>(LevelInfo{0, "domain", net, ""});
            addr->ports = {local == 2 ? "0/6" : "1/0"};
            root.eidIndexToRankAddr.emplace(7, addr);
            UdmaTransportManager manager;
            std::vector<std::vector<SyncEndpoint>> endpoints;
            ASSERT_TRUE(BuildSyncEndpointsForTest(manager, root, topo, 1, endpoints));
            const auto inferred = local == 0 ? TopoFabric::Clos : local == 9 ? TopoFabric::Unknown : TopoFabric::Mesh;
            EXPECT_EQ(
                endpoints[0][0].fabric, net == NetType::Clos ? TopoFabric::Clos :
                                        net == NetType::Mesh ? TopoFabric::Mesh :
                                                               inferred);
        }
    }
}

TEST_F(UdmaEndpointExchangeTest, KeepsGenuinelyAmbiguousLocalTopologyUnclassified)
{
    g_boot_handle.npes = 1;
    g_boot_handle.allgather = MirrorAllGather;
    RootInfo root;
    auto addr = std::make_shared<RankAddr>();
    addr->levelInfo = std::make_shared<LevelInfo>(LevelInfo{0, "domain", NetType::TopoFileDesc, ""});
    addr->ports = {"shared"};
    root.eidIndexToRankAddr.emplace(7, addr);
    TopoInfo topo;
    topo.meshTopoEdges.push_back({0, 1, {"shared"}, {"peer"}});
    topo.closTopoEdges.push_back({1, 0, 0, {"shared"}});
    UdmaTransportManager manager;
    std::vector<std::vector<SyncEndpoint>> endpoints;
    ASSERT_TRUE(BuildSyncEndpointsForTest(manager, root, topo, 1, endpoints));
    EXPECT_EQ(endpoints[0][0].fabric, TopoFabric::Unknown);
}

TEST_F(UdmaEndpointExchangeTest, ValidatesSymmetricMasksAtEp256And512)
{
    for (uint32_t ranks : {256U, 512U}) {
        lengths_.clear();
        mask_matrix_.resize(static_cast<size_t>(ranks) * ranks);
        for (uint32_t a = 0; a < ranks; ++a)
            for (uint32_t b = 0; b < ranks; ++b)
                mask_matrix_[static_cast<size_t>(a) * ranks + b] = TestMask(a, b);
        UdmaTransportManager manager;
        ConfigureManager(manager, 0, ranks, 32);
        SelectQps(manager, std::vector<uint64_t>(mask_matrix_.begin(), mask_matrix_.begin() + ranks));
        AddEndpoint(manager, 7, 1, 7);
        Exchange exchange;
        ASSERT_EQ(ExchangeDescriptors(manager, exchange), ACLSHMEM_SUCCESS);
        ASSERT_EQ(lengths_.size(), 3U);
        EXPECT_EQ(lengths_[1], ranks * sizeof(uint64_t));
        EXPECT_EQ(exchange.counts.size(), ranks);
    }
}

TEST_F(UdmaEndpointExchangeTest, RejectsAsymmetricMasksOnEveryRankBeforeChannelCreation)
{
    mask_matrix_ = {0, 3, 5, 3, 0, 3, 5, 1, 0}; // Only pair 1/2 disagrees.
    for (uint32_t rank = 0; rank < 3; ++rank) {
        lengths_.clear();
        UdmaTransportManager manager;
        ConfigureManager(manager, rank, 3, 4);
        SelectQps(
            manager, std::vector<uint64_t>(mask_matrix_.begin() + rank * 3, mask_matrix_.begin() + (rank + 1) * 3));
        AddEndpoint(manager, 7, (rank + 1) % 3, 7);
        Exchange exchange;
        EXPECT_EQ(ExchangeDescriptors(manager, exchange), ACLSHMEM_INVALID_VALUE);
        EXPECT_EQ(lengths_.size(), 2U);
        EXPECT_TRUE(channel_endpoints_.empty());
    }
}

TEST_F(UdmaEndpointExchangeTest, RejectsMixedDenseSparseOrQpCountsCollectively)
{
    for (uint32_t mismatch : {1U, 2U}) {
        lengths_.clear();
        config_mismatch_ = mismatch;
        UdmaTransportManager manager;
        ConfigureManager(manager, 0, 3, 4);
        AddEndpoint(manager, 7, 1, 7);
        Exchange exchange;
        EXPECT_EQ(ExchangeDescriptors(manager, exchange), ACLSHMEM_INVALID_VALUE);
        EXPECT_EQ(lengths_.size(), 1U);
    }
}

TEST_F(UdmaEndpointExchangeTest, PropagatesMaskExchangeFailureAndSkipsMaskExchangeWhenDense)
{
    UdmaTransportManager manager;
    ConfigureManager(manager, 0, 3, 4);
    AddEndpoint(manager, 7, 1, 7);
    Exchange exchange;
    ASSERT_EQ(ExchangeDescriptors(manager, exchange), ACLSHMEM_SUCCESS);
    EXPECT_EQ(lengths_.size(), 2U); // Config + descriptors; no mask collective.
    SelectQps(manager, {0, 3, 5});
    for (size_t failure : {1U, 2U}) {
        lengths_.clear();
        fail_call_ = failure;
        EXPECT_EQ(ExchangeDescriptors(manager, exchange), ACLSHMEM_BOOTSTRAP_ERROR);
        EXPECT_EQ(lengths_.size(), failure);
    }
}

TEST_F(UdmaEndpointExchangeTest, RejectsEmptyAndOverflowingDimensions)
{
    size_t count = 1;
    EXPECT_FALSE(GetSize(0, 1, count));
    EXPECT_EQ(count, 0U);
    EXPECT_FALSE(GetSize(1, 0, count));
    EXPECT_FALSE(GetSize(std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max(), count));
    EXPECT_EQ(count, 0U);
}

TEST_F(UdmaEndpointExchangeTest, AcceptsDescriptorMatricesAboveEightMiB)
{
    size_t count = 0;
    EXPECT_TRUE(GetSize(8, 7, count));
    EXPECT_EQ(count, 56U);
    EXPECT_TRUE(GetSize(256, 255, count));
    EXPECT_EQ(count, 256U * 255U);
    EXPECT_TRUE(GetSize(1024, 1023, count));
    EXPECT_EQ(count, 1024U * 1023U);
    EXPECT_TRUE(GetSize(1024, 8U * 1023U, count));
    EXPECT_EQ(count, 1024U * 8U * 1023U);
}

TEST_F(UdmaEndpointExchangeTest, SplitsLargeClusterMatricesWithoutAllocatingThem)
{
    for (uint32_t ranks : {3493U, 3494U, 4096U, 10000U, 16384U}) {
        size_t count = 0;
        ASSERT_TRUE(GetSize(ranks, ranks - 1, count));
        const uint32_t chunk = ChunkCount(ranks);
        ASSERT_GT(chunk, 0U);
        EXPECT_LT(chunk, ranks - 1);
        // Mirror config_store's int multiplication under UBSan.
        const int bytes = static_cast<int>(chunk * sizeof(Descriptor));
        const int receive_bytes = bytes * static_cast<int>(ranks);
        EXPECT_LE(receive_bytes, 8 * 1024 * 1024);
    }
    EXPECT_EQ(ChunkCount(0), 0U);
}

TEST_F(UdmaEndpointExchangeTest, PreservesRankStridePaddingAndTailAcrossChunks)
{
    Exchange exchange;
    ASSERT_EQ(Gather(2, exchange), ACLSHMEM_SUCCESS);
    EXPECT_EQ(lengths_, (std::vector<int>{2 * sizeof(Descriptor), 2 * sizeof(Descriptor), sizeof(Descriptor)}));
    ASSERT_EQ(exchange.descs.size(), 15U);
    for (uint32_t rank = 0; rank < 3; ++rank) {
        for (uint32_t idx = 0; idx < 5; ++idx) {
            const auto& desc = exchange.descs[rank * 5 + idx];
            EXPECT_EQ(desc.peer_rank, rank);
            EXPECT_EQ(desc.eid_index, idx);
            EXPECT_EQ(desc.valid, idx < 4 ? 1U : 0U);
            EXPECT_EQ(desc.listen_port, 100U + idx);
        }
    }
}

TEST_F(UdmaEndpointExchangeTest, SingleChunkMatchesSplitExchange)
{
    Exchange split;
    ASSERT_EQ(Gather(2, split), ACLSHMEM_SUCCESS);
    lengths_.clear();
    Exchange single;
    ASSERT_EQ(Gather(5, single), ACLSHMEM_SUCCESS);
    EXPECT_EQ(lengths_.size(), 1U);
    ASSERT_EQ(single.descs.size(), split.descs.size());
    for (size_t idx = 0; idx < split.descs.size(); ++idx) {
        EXPECT_EQ(single.descs[idx].eid_index, split.descs[idx].eid_index);
        EXPECT_EQ(single.descs[idx].peer_rank, split.descs[idx].peer_rank);
        EXPECT_EQ(single.descs[idx].valid, split.descs[idx].valid);
    }
}

TEST_F(UdmaEndpointExchangeTest, StopsAtFailedChunk)
{
    fail_call_ = 2;
    Exchange exchange;
    EXPECT_EQ(Gather(2, exchange), ACLSHMEM_BOOTSTRAP_ERROR);
    EXPECT_EQ(lengths_.size(), 2U);
}

TEST_F(UdmaEndpointExchangeTest, RegistersOneHeapOnEverySelectedUniqueEndpoint)
{
    UdmaTransportManager manager;
    AddEndpoint(manager, 10, 1, 1);
    AddEndpoint(manager, 11, 1, 2);
    AddEndpoint(manager, 12, 2, 3);

    TransportMemoryRegion region{};
    region.addr = 0x100000;
    region.size = 0x2000;
    region.flags = REG_MR_FLAG_HBM;
    ASSERT_EQ(manager.RegisterMemoryRegion(region), ACLSHMEM_SUCCESS);

    EXPECT_EQ(mem_reg_endpoints_, (std::vector<EndpointHandle>{Handle(1), Handle(2), Handle(3)}));
    EXPECT_EQ(RegisteredEndpointCount(manager, region.addr), 3U);
}

TEST_F(UdmaEndpointExchangeTest, RegistersSharedLocalEidOnlyOnceAcrossPeers)
{
    if constexpr (ACLSHMEM_UDMA_RELAY_ENABLED) {
        GTEST_SKIP() << "Cross-peer endpoint sharing applies to direct builds";
    }
    UdmaTransportManager manager;
    EnableSharedJetty(manager);
    AddEndpoint(manager, 7, 1, 7);
    AddEndpoint(manager, 7, 2, 7);
    AddEndpoint(manager, 7, 3, 7);

    TransportMemoryRegion region{};
    region.addr = 0x100000;
    region.size = 0x2000;
    region.flags = REG_MR_FLAG_HBM;
    ASSERT_EQ(manager.RegisterMemoryRegion(region), ACLSHMEM_SUCCESS);

    EXPECT_EQ(mem_reg_endpoints_, (std::vector<EndpointHandle>{Handle(7)}));
    EXPECT_EQ(RegisteredEndpointCount(manager, region.addr), 1U);
    EXPECT_EQ(LogicalEndpointCount(manager), 1U);
}

TEST_F(UdmaEndpointExchangeTest, KeepsLocalEidSeparateAcrossPeersWithoutSharedJetty)
{
    UdmaTransportManager manager;
    AddEndpoint(manager, 7, 1, 1);
    AddEndpoint(manager, 7, 2, 2);
    AddEndpoint(manager, 7, 3, 3);

    TransportMemoryRegion region{};
    region.addr = 0x100000;
    region.size = 0x2000;
    region.flags = REG_MR_FLAG_HBM;
    ASSERT_EQ(manager.RegisterMemoryRegion(region), ACLSHMEM_SUCCESS);
    EXPECT_EQ(mem_reg_endpoints_, (std::vector<EndpointHandle>{Handle(1), Handle(2), Handle(3)}));
    EXPECT_EQ(RegisteredEndpointCount(manager, region.addr), 3U);
}

TEST_F(UdmaEndpointExchangeTest, BuildsOneChannelForEveryPeerQpFromFinalRoutes)
{
    UdmaTransportManager manager;
    ConfigureManager(manager, 0, 3, 3);

    // Repeated local EIDs model two QPs sharing one endpoint. The channel count
    // must still be peer_count * qp_num, while endpoint/MR resources are reused.
    AddRoute(manager, 1, 10, 20, 0);
    AddRoute(manager, 1, 10, 20, 0);
    AddRoute(manager, 1, 11, 21, 1);
    AddRoute(manager, 2, 12, 22, 0);
    AddRoute(manager, 2, 13, 23, 1);
    AddRoute(manager, 2, 12, 22, 0);
    AddEndpoint(manager, 10, 1, 10);
    AddEndpoint(manager, 11, 1, 11);
    AddEndpoint(manager, 12, 2, 12);
    AddEndpoint(manager, 13, 2, 13);

    TransportMemoryRegion region{};
    region.addr = 0x200000;
    region.size = 0x4000;
    region.flags = REG_MR_FLAG_HBM;
    ASSERT_EQ(manager.RegisterMemoryRegion(region), ACLSHMEM_SUCCESS);
    ASSERT_EQ(mem_reg_endpoints_.size(), 4U);

    Exchange exchange;
    exchange.max_count = 3;
    exchange.counts = {0, 3, 3};
    exchange.descs.resize(9);
    AddRemoteDescriptor(exchange, 1, 0, 0, 20);
    AddRemoteDescriptor(exchange, 1, 1, 0, 21);
    AddRemoteDescriptor(exchange, 2, 0, 0, 22);
    AddRemoteDescriptor(exchange, 2, 1, 0, 23);

    std::vector<uint32_t> slots;
    std::vector<uint32_t> dst_pes;
    std::vector<uint32_t> qp_indices;
    ASSERT_EQ(BuildDirect(manager, exchange, slots, dst_pes, qp_indices), ACLSHMEM_SUCCESS);

    EXPECT_EQ(
        channel_endpoints_,
        (std::vector<EndpointHandle>{Handle(10), Handle(10), Handle(11), Handle(12), Handle(13), Handle(12)}));
    EXPECT_EQ(slots, (std::vector<uint32_t>{1, 1, 1, 2, 2, 2}));
    EXPECT_EQ(dst_pes, (std::vector<uint32_t>{1, 1, 1, 2, 2, 2}));
    EXPECT_EQ(qp_indices, (std::vector<uint32_t>{0, 1, 2, 0, 1, 2}));
}

TEST_F(UdmaEndpointExchangeTest, SparseChannelsKeepOriginalPeerAndQpIndices)
{
    if constexpr (ACLSHMEM_UDMA_RELAY_ENABLED)
        GTEST_SKIP() << "Sparse channels are direct-only";
    UdmaTransportManager manager;
    ConfigureManager(manager, 0, 3, 4);
    SelectQps(manager, {0, 9, 5}); // Peer1: QP0/3; peer2: QP0/2.
    for (uint32_t peer = 1; peer < 3; ++peer) {
        for (uint32_t qp = 0; qp < 4; ++qp)
            AddRoute(manager, peer, 10 + peer, 20 + peer, 0);
        AddEndpoint(manager, 10 + peer, peer, 10 + peer);
    }
    TransportMemoryRegion region{};
    region.addr = 0x200000;
    region.size = 0x4000;
    region.flags = REG_MR_FLAG_HBM;
    ASSERT_EQ(manager.RegisterMemoryRegion(region), ACLSHMEM_SUCCESS);
    Exchange exchange;
    exchange.max_count = 1;
    exchange.counts = {0, 1, 1};
    exchange.descs.resize(3);
    AddRemoteDescriptor(exchange, 1, 0, 0, 21);
    AddRemoteDescriptor(exchange, 2, 0, 0, 22);
    std::vector<uint32_t> slots, dst_pes, qp_indices;
    ASSERT_EQ(BuildDirect(manager, exchange, slots, dst_pes, qp_indices), ACLSHMEM_SUCCESS);
    EXPECT_EQ(slots, (std::vector<uint32_t>{1, 1, 2, 2}));
    EXPECT_EQ(dst_pes, slots);
    EXPECT_EQ(qp_indices, (std::vector<uint32_t>{0, 3, 0, 2}));
    EXPECT_EQ(channel_endpoints_.size(), 4U);
}

TEST_F(UdmaEndpointExchangeTest, SparseSharedChannelsPreserveRoutesNamesAndHighQpIdsAtEp256And512)
{
    if constexpr (ACLSHMEM_UDMA_RELAY_ENABLED)
        GTEST_SKIP() << "Sparse channels are direct-only";
    for (uint32_t ranks : {256U, 512U}) {
        channel_endpoints_.clear();
        channel_names_.clear();
        UdmaTransportManager manager;
        ConfigureManager(manager, 0, ranks, 32);
        EnableSharedJetty(manager);
        std::vector<uint64_t> masks(ranks);
        Exchange exchange;
        exchange.max_count = 32;
        exchange.counts.assign(ranks, 32);
        exchange.descs.resize(static_cast<size_t>(ranks) * 32);
        for (uint32_t peer = 1; peer < ranks; ++peer) {
            masks[peer] = TestMask(0, peer);
            for (uint32_t qp = 0; qp < 32; ++qp) {
                // Original parity remains the resolved plane; holes must not compact it.
                AddRoute(manager, peer, 10 + qp % 2, 20 + qp % 2, qp % 2);
                AddEndpoint(manager, 10 + qp % 2, peer, 10 + qp, qp);
                AddRemoteDescriptor(exchange, peer, qp, 0xFFFFFFFFU, 20 + qp % 2, qp);
            }
        }
        SelectQps(manager, masks);
        TransportMemoryRegion region{};
        region.addr = 0x200000;
        region.size = 0x4000;
        region.flags = REG_MR_FLAG_HBM;
        ASSERT_EQ(manager.RegisterMemoryRegion(region), ACLSHMEM_SUCCESS);
        EXPECT_EQ(RegisteredEndpointCount(manager, region.addr), 32U);
        std::vector<uint32_t> slots, peers, qps;
        ASSERT_EQ(BuildDirect(manager, exchange, slots, peers, qps), ACLSHMEM_SUCCESS);
        ASSERT_EQ(qps.size(), (ranks - 1) * 2);
        size_t index = 0;
        for (uint32_t peer = 1; peer < ranks; ++peer) {
            for (uint32_t qp : {0U, 1 + peer % 31}) {
                EXPECT_EQ(slots[index], peer);
                EXPECT_EQ(peers[index], peer);
                EXPECT_EQ(qps[index], qp);
                EXPECT_EQ(channel_endpoints_[index], Handle(10 + qp));
                EXPECT_EQ(
                    channel_names_[index], "aclshmem_udma_0_" + std::to_string(peer) + "_qp_" + std::to_string(qp));
                uint32_t layer = 99;
                ASSERT_EQ(manager.GetQpNetLayer(ACLSHMEM_DATA_OP_UDMA, peer, qp, layer), ACLSHMEM_SUCCESS);
                EXPECT_EQ(layer, qp % 2);
                ++index;
            }
        }
    }
}

TEST_F(UdmaEndpointExchangeTest, SharedJettyKeepsQpEndpointsSeparateWithLayerRoutes)
{
    if constexpr (ACLSHMEM_UDMA_RELAY_ENABLED) {
        GTEST_SKIP() << "Per-QP endpoint splitting applies to direct builds";
    }
    UdmaTransportManager manager;
    ConfigureManager(manager, 0, 3, 2);
    EnableSharedJetty(manager);
    // Both peers converge on the same EID, but each QP retains its own endpoint
    // and its resolved network layer, even while sharing with the other peer.
    for (uint32_t peer : {1U, 2U}) {
        AddRoute(manager, peer, 10, 20, 0);
        AddRoute(manager, peer, 10, 20, 1);
        AddEndpoint(manager, 10, peer, 10, 0);
        AddEndpoint(manager, 10, peer, 11, 1);
    }
    TransportMemoryRegion region{};
    region.addr = 0x200000;
    region.size = 0x4000;
    region.flags = REG_MR_FLAG_HBM;
    ASSERT_EQ(manager.RegisterMemoryRegion(region), ACLSHMEM_SUCCESS);
    EXPECT_EQ(RegisteredEndpointCount(manager, region.addr), 2U);
    EXPECT_EQ(mem_reg_endpoints_, (std::vector<EndpointHandle>{Handle(10), Handle(11)}));

    Exchange exchange;
    exchange.max_count = 2;
    exchange.counts = {0, 2, 2};
    exchange.descs.resize(6);
    for (uint32_t peer : {1U, 2U}) {
        for (uint32_t qp = 0; qp < 2; ++qp) {
            AddRemoteDescriptor(exchange, peer, qp, 0xFFFFFFFFU, 20, qp);
            uint32_t layer = 99;
            ASSERT_EQ(manager.GetQpNetLayer(ACLSHMEM_DATA_OP_UDMA, peer, qp, layer), ACLSHMEM_SUCCESS);
            EXPECT_EQ(layer, qp);
        }
    }
    std::vector<uint32_t> slots;
    std::vector<uint32_t> dst_pes;
    std::vector<uint32_t> qp_indices;
    ASSERT_EQ(BuildDirect(manager, exchange, slots, dst_pes, qp_indices), ACLSHMEM_SUCCESS);
    EXPECT_EQ(channel_endpoints_, (std::vector<EndpointHandle>{Handle(10), Handle(11), Handle(10), Handle(11)}));
    EXPECT_EQ(slots, (std::vector<uint32_t>{1, 1, 2, 2}));
    EXPECT_EQ(dst_pes, (std::vector<uint32_t>{1, 1, 2, 2}));
    EXPECT_EQ(qp_indices, (std::vector<uint32_t>{0, 1, 0, 1}));
}

} // namespace device
} // namespace transport
} // namespace shm
