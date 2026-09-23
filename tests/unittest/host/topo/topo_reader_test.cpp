/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>
#include <utility>

#include "topo/topo_reader.h"

namespace shm {
namespace transport {
namespace {

EidData MakeEid(uint8_t tag)
{
    EidData eid{};
    eid.fill(0);
    eid[15] = tag;
    return eid;
}

std::shared_ptr<RankAddr> MakeRankAddr(const std::string& port, const std::string& planeId, uint8_t tag)
{
    auto rankAddr = std::make_shared<RankAddr>();
    rankAddr->eidData = MakeEid(tag);
    rankAddr->planeId = planeId;
    rankAddr->ports = {port};
    return rankAddr;
}

std::shared_ptr<RankAddr> MakeRankAddr(const std::vector<std::string>& ports, const std::string& planeId, uint8_t tag)
{
    auto rankAddr = std::make_shared<RankAddr>();
    rankAddr->eidData = MakeEid(tag);
    rankAddr->planeId = planeId;
    rankAddr->ports = ports;
    return rankAddr;
}

SyncEndpoint MakeEndpoint(
    const std::string& netInstanceId, const std::string& planeId, const std::string& port, uint32_t eidIndex,
    TopoFabric fabric)
{
    SyncEndpoint endpoint;
    endpoint.netLayer = 0;
    endpoint.netInstanceId = netInstanceId;
    endpoint.netType = NetType::TopoFileDesc;
    endpoint.fabric = fabric;
    endpoint.planeId = planeId;
    endpoint.ports = {port};
    endpoint.eidIndex = eidIndex;
    return endpoint;
}

SyncEndpoint MakeEndpoint(
    const std::string& netInstanceId, const std::string& planeId, const std::vector<std::string>& ports,
    uint32_t eidIndex, TopoFabric fabric)
{
    SyncEndpoint endpoint;
    endpoint.netLayer = 0;
    endpoint.netInstanceId = netInstanceId;
    endpoint.netType = NetType::TopoFileDesc;
    endpoint.fabric = fabric;
    endpoint.planeId = planeId;
    endpoint.ports = ports;
    endpoint.eidIndex = eidIndex;
    return endpoint;
}

std::pair<std::string, std::string> CanonicalPair(std::string lhs, std::string rhs)
{
    if (lhs > rhs) {
        std::swap(lhs, rhs);
    }
    return {std::move(lhs), std::move(rhs)};
}

std::vector<std::pair<std::string, std::string>> ExtractCanonicalPairs(
    const RootInfo& localRoot, const RootInfo& remoteRoot, const std::vector<EidRoute>& routes)
{
    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(routes.size());
    for (const auto& route : routes) {
        const auto local_it = localRoot.eidIndexToRankAddr.find(route.localEidIndex);
        const auto remote_it = remoteRoot.eidIndexToRankAddr.find(route.remoteEidIndex);
        EXPECT_TRUE(local_it != localRoot.eidIndexToRankAddr.end());
        EXPECT_TRUE(remote_it != remoteRoot.eidIndexToRankAddr.end());
        if (local_it == localRoot.eidIndexToRankAddr.end() || remote_it == remoteRoot.eidIndexToRankAddr.end()) {
            continue;
        }
        EXPECT_NE(local_it->second, nullptr);
        EXPECT_NE(remote_it->second, nullptr);
        if (local_it->second == nullptr || remote_it->second == nullptr) {
            continue;
        }
        EXPECT_FALSE(local_it->second->ports.empty());
        EXPECT_FALSE(remote_it->second->ports.empty());
        if (local_it->second->ports.empty() || remote_it->second->ports.empty()) {
            continue;
        }
        out.push_back(CanonicalPair(local_it->second->ports.front(), remote_it->second->ports.front()));
    }
    return out;
}

bool ParseIndexedTopo(TopoInfo& topoInfo)
{
    nlohmann::json edges = nlohmann::json::array();
    for (const auto& edge : topoInfo.meshTopoEdges) {
        edges.push_back(
            {{"topo_type", "MESH"},
             {"local_a", edge.localA},
             {"local_b", edge.localB},
             {"local_a_ports", edge.localAPorts},
             {"local_b_ports", edge.localBPorts}});
    }
    for (const auto& edge : topoInfo.closTopoEdges) {
        edges.push_back(
            {{"topo_type", "CLOS"},
             {"net_layer", edge.netLayer},
             {"topo_instance_id", edge.topoInstanceId},
             {"local_a", edge.localA},
             {"local_a_ports", edge.ports}});
    }
    char path[] = "/tmp/shmem_topo_reader_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        return false;
    }
    close(fd);
    {
        std::ofstream file(path);
        file << nlohmann::json{{"edge_list", edges}};
    }
    const bool parsed = TopoReader::ParseTopoInfo(path, topoInfo);
    std::remove(path);
    return parsed;
}

class TopoReaderIndexedTest : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(RouteLookup, TopoReaderIndexedTest, testing::Bool());

} // namespace

TEST(TopoReaderTest, TopoFileDescUsesTopoTypeToSelectClosWhenMeshEdgeMissing)
{
    RootInfo rootInfo;
    rootInfo.deviceId = 0;
    rootInfo.localId = 0;

    rootInfo.eidIndexToRankAddr[0] = MakeRankAddr("0/0", "plane_flat", 0x10);
    rootInfo.portsToRankAddr["0/0"] = rootInfo.eidIndexToRankAddr[0];
    rootInfo.eidIndexToRankAddr[1] = MakeRankAddr("0/8", "plane_flat", 0x11);
    rootInfo.portsToRankAddr["0/8"] = rootInfo.eidIndexToRankAddr[1];
    rootInfo.eidIndexToRankAddr[2] = MakeRankAddr("4/0", "plane_flat", 0x20);
    rootInfo.portsToRankAddr["4/0"] = rootInfo.eidIndexToRankAddr[2];
    rootInfo.eidIndexToRankAddr[3] = MakeRankAddr("4/8", "plane_flat", 0x21);
    rootInfo.portsToRankAddr["4/8"] = rootInfo.eidIndexToRankAddr[3];

    TopoInfo topoInfo;
    topoInfo.meshTopoEdges.push_back(MeshTopoEdge{0, 1, {"0/0"}, {"1/0"}});
    topoInfo.meshTopoEdges.push_back(MeshTopoEdge{4, 5, {"4/0"}, {"5/0"}});
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 0, {"0/8"}});
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 4, {"4/8"}});

    std::vector<uint32_t> rankToLocalId{0, 4};
    std::vector<std::vector<SyncEndpoint>> endpoints(2);
    endpoints[0].push_back(MakeEndpoint("server-id", "plane_flat", "0/0", 0, TopoFabric::Mesh));
    endpoints[0].push_back(MakeEndpoint("server-id", "plane_flat", "0/8", 1, TopoFabric::Clos));
    endpoints[1].push_back(MakeEndpoint("server-id", "plane_flat", "4/0", 2, TopoFabric::Mesh));
    endpoints[1].push_back(MakeEndpoint("server-id", "plane_flat", "4/8", 3, TopoFabric::Clos));

    TopoQuerier querier(rootInfo, topoInfo, 0, rankToLocalId, endpoints);

    uint32_t localEidIndex = 0;
    EidData localEidRaw{};
    uint32_t remoteEidIndex = 0;
    ASSERT_TRUE(querier.GetEidRoute(1, localEidIndex, localEidRaw, remoteEidIndex));
    EXPECT_EQ(localEidIndex, 1U);
    EXPECT_EQ(remoteEidIndex, 3U);
    EXPECT_EQ(localEidRaw[15], 0x11);
}

TEST(TopoReaderTest, TopoFileDescMeshRouteDoesNotRequirePlaneId)
{
    RootInfo rootInfo;
    rootInfo.deviceId = 0;
    rootInfo.localId = 0;

    rootInfo.eidIndexToRankAddr[0] = MakeRankAddr("0/0", "", 0x10);
    rootInfo.eidIndexToRankAddr[1] = MakeRankAddr("4/0", "", 0x20);

    TopoInfo topoInfo;
    topoInfo.meshTopoEdges.push_back(MeshTopoEdge{0, 4, {"0/0"}, {"4/0"}});

    std::vector<uint32_t> rankToLocalId{0, 4};
    std::vector<std::vector<SyncEndpoint>> endpoints(2);
    endpoints[0].push_back(MakeEndpoint("server-id", "", "0/0", 0, TopoFabric::Mesh));
    endpoints[1].push_back(MakeEndpoint("server-id", "", "4/0", 1, TopoFabric::Mesh));

    TopoQuerier querier(rootInfo, topoInfo, 0, rankToLocalId, endpoints);

    uint32_t localEidIndex = 0;
    EidData localEidRaw{};
    uint32_t remoteEidIndex = 0;
    ASSERT_TRUE(querier.GetEidRoute(1, localEidIndex, localEidRaw, remoteEidIndex));
    EXPECT_EQ(localEidIndex, 0U);
    EXPECT_EQ(remoteEidIndex, 1U);
    EXPECT_EQ(localEidRaw[15], 0x10);
}

TEST_P(TopoReaderIndexedTest, TopoFileDescKeepsMeshWithinLayerZeroInstance)
{
    TopoInfo topo;
    topo.meshTopoEdges.push_back(MeshTopoEdge{0, 1, {"0/0"}, {"1/0"}});
    topo.closTopoEdges.push_back(ClosTopoEdge{1, 0, 0, {"0/8"}});
    topo.closTopoEdges.push_back(ClosTopoEdge{1, 0, 1, {"1/8"}});
    if (GetParam()) {
        ASSERT_TRUE(ParseIndexedTopo(topo));
    }
    std::vector<uint32_t> localIds{0, 1};
    std::vector<std::vector<SyncEndpoint>> endpoints(2);
    for (uint32_t rank = 0; rank < 2; ++rank) {
        const std::string port = std::to_string(rank);
        endpoints[rank].push_back(MakeEndpoint("server-" + port, "", port + "/0", 0, TopoFabric::Mesh));
        auto clos = MakeEndpoint("shared-clos", "plane-clos", port + "/8", 1, TopoFabric::Clos);
        clos.netLayer = 1;
        endpoints[rank].push_back(clos);
    }
    for (const auto peerType : {NetType::TopoFileDesc, NetType::Mesh}) {
        endpoints[1][0].netType = peerType;
        for (const bool sameInstance : {false, true}) {
            endpoints[1][0].netInstanceId = sameInstance ? "server-0" : "server-1";
            for (uint32_t rank = 0; rank < 2; ++rank) {
                RootInfo root;
                root.eidIndexToRankAddr[0] = MakeRankAddr(std::to_string(rank) + "/0", "", 0x10);
                root.eidIndexToRankAddr[1] = MakeRankAddr(std::to_string(rank) + "/8", "plane-clos", 0x11);
                TopoQuerier querier(root, topo, rank, localIds, endpoints);
                uint32_t local = 0;
                uint32_t remote = 0;
                EidData raw{};
                ASSERT_TRUE(querier.GetEidRoute(1 - rank, local, raw, remote));
                EXPECT_EQ(local, sameInstance ? 0U : 1U);
                EXPECT_EQ(remote, sameInstance ? 0U : 1U);
            }
        }
    }
}

TEST(TopoReaderTest, TopoFileDescPrefersMultiPortClosEndpoint)
{
    RootInfo rootInfo;
    rootInfo.deviceId = 0;
    rootInfo.localId = 0;

    rootInfo.eidIndexToRankAddr[0] = MakeRankAddr(std::vector<std::string>{"0/8", "0/9"}, "plane_flat", 0x10);
    rootInfo.eidIndexToRankAddr[1] = MakeRankAddr(std::vector<std::string>{"0/8"}, "plane_flat", 0x11);
    rootInfo.eidIndexToRankAddr[2] = MakeRankAddr(std::vector<std::string>{"4/8"}, "plane_flat", 0x20);
    rootInfo.eidIndexToRankAddr[3] = MakeRankAddr(std::vector<std::string>{"4/8", "4/9"}, "plane_flat", 0x21);

    TopoInfo topoInfo;
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 0, {"0/8"}});
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 4, {"4/8"}});

    std::vector<uint32_t> rankToLocalId{0, 4};
    std::vector<std::vector<SyncEndpoint>> endpoints(2);
    endpoints[0].push_back(
        MakeEndpoint("server-id", "plane_flat", std::vector<std::string>{"0/8", "0/9"}, 0, TopoFabric::Clos));
    endpoints[0].push_back(
        MakeEndpoint("server-id", "plane_flat", std::vector<std::string>{"0/8"}, 1, TopoFabric::Clos));
    endpoints[1].push_back(
        MakeEndpoint("server-id", "plane_flat", std::vector<std::string>{"4/8"}, 2, TopoFabric::Clos));
    endpoints[1].push_back(
        MakeEndpoint("server-id", "plane_flat", std::vector<std::string>{"4/8", "4/9"}, 3, TopoFabric::Clos));

    TopoQuerier querier(rootInfo, topoInfo, 0, rankToLocalId, endpoints);

    uint32_t localEidIndex = 0;
    EidData localEidRaw{};
    uint32_t remoteEidIndex = 0;
    ASSERT_TRUE(querier.GetEidRoute(1, localEidIndex, localEidRaw, remoteEidIndex));
    EXPECT_EQ(localEidIndex, 0U);
    EXPECT_EQ(remoteEidIndex, 3U);
    EXPECT_EQ(localEidRaw[15], 0x10);
}

TEST_P(TopoReaderIndexedTest, TopoFileDescPrefersMultiPortClosCandidateBeforeTopoInstanceId)
{
    RootInfo rootInfo;
    rootInfo.deviceId = 0;
    rootInfo.localId = 0;

    rootInfo.eidIndexToRankAddr[0] = MakeRankAddr("0/0", "plane_flat", 0x10);
    rootInfo.eidIndexToRankAddr[1] = MakeRankAddr(std::vector<std::string>{"0/8", "0/9"}, "plane_flat", 0x11);
    rootInfo.eidIndexToRankAddr[2] = MakeRankAddr("4/0", "plane_flat", 0x20);
    rootInfo.eidIndexToRankAddr[3] = MakeRankAddr(std::vector<std::string>{"4/8", "4/9"}, "plane_flat", 0x21);

    TopoInfo topoInfo;
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 0, {"0/0"}});
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 4, {"4/0"}});
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 1, 0, {"0/8", "0/9"}});
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 1, 4, {"4/8", "4/9"}});
    if (GetParam()) {
        ASSERT_TRUE(ParseIndexedTopo(topoInfo));
        ASSERT_FALSE(topoInfo.closEdgeIndicesByLocalA.empty());
    }

    std::vector<uint32_t> rankToLocalId{0, 4};
    std::vector<std::vector<SyncEndpoint>> endpoints(2);
    endpoints[0].push_back(MakeEndpoint("server-id", "plane_flat", "0/0", 0, TopoFabric::Clos));
    endpoints[0].push_back(
        MakeEndpoint("server-id", "plane_flat", std::vector<std::string>{"0/8", "0/9"}, 1, TopoFabric::Clos));
    endpoints[1].push_back(MakeEndpoint("server-id", "plane_flat", "4/0", 2, TopoFabric::Clos));
    endpoints[1].push_back(
        MakeEndpoint("server-id", "plane_flat", std::vector<std::string>{"4/8", "4/9"}, 3, TopoFabric::Clos));

    TopoQuerier querier(rootInfo, topoInfo, 0, rankToLocalId, endpoints);

    uint32_t localEidIndex = 0;
    EidData localEidRaw{};
    uint32_t remoteEidIndex = 0;
    ASSERT_TRUE(querier.GetEidRoute(1, localEidIndex, localEidRaw, remoteEidIndex));
    EXPECT_EQ(localEidIndex, 1U);
    EXPECT_EQ(remoteEidIndex, 3U);
    EXPECT_EQ(localEidRaw[15], 0x11);
}

TEST(TopoReaderTest, TopoFileDescAllowsMixedEndpointWhenRequestedPortMatchesFabric)
{
    RootInfo rootInfo;
    rootInfo.deviceId = 0;
    rootInfo.localId = 0;

    rootInfo.eidIndexToRankAddr[0] = MakeRankAddr(std::vector<std::string>{"0/0", "0/8"}, "plane_flat", 0x10);
    rootInfo.eidIndexToRankAddr[1] = MakeRankAddr(std::vector<std::string>{"4/0", "4/8"}, "plane_flat", 0x20);
    rootInfo.portsToRankAddr["0/0"] = rootInfo.eidIndexToRankAddr[0];
    rootInfo.portsToRankAddr["0/8"] = rootInfo.eidIndexToRankAddr[0];
    rootInfo.portsToRankAddr["4/0"] = rootInfo.eidIndexToRankAddr[1];
    rootInfo.portsToRankAddr["4/8"] = rootInfo.eidIndexToRankAddr[1];

    TopoInfo topoInfo;
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 0, {"0/8"}});
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 4, {"4/8"}});

    std::vector<uint32_t> rankToLocalId{0, 4};
    std::vector<std::vector<SyncEndpoint>> endpoints(2);
    endpoints[0].push_back(
        MakeEndpoint("server-id", "plane_flat", std::vector<std::string>{"0/0", "0/8"}, 0, TopoFabric::Unknown));
    endpoints[1].push_back(
        MakeEndpoint("server-id", "plane_flat", std::vector<std::string>{"4/0", "4/8"}, 1, TopoFabric::Unknown));

    TopoQuerier querier(rootInfo, topoInfo, 0, rankToLocalId, endpoints);

    uint32_t localEidIndex = 0;
    EidData localEidRaw{};
    uint32_t remoteEidIndex = 0;
    ASSERT_TRUE(querier.GetEidRoute(1, localEidIndex, localEidRaw, remoteEidIndex));
    EXPECT_EQ(localEidIndex, 0U);
    EXPECT_EQ(remoteEidIndex, 1U);
    EXPECT_EQ(localEidRaw[15], 0x10);
}

TEST_P(TopoReaderIndexedTest, TopoFileDescUsesCanonicalClosCandidateOrderAcrossPeers)
{
    RootInfo localRoot;
    localRoot.deviceId = 0;
    localRoot.localId = 0;
    localRoot.eidIndexToRankAddr[0] = MakeRankAddr("0/4", "plane_flat", 0x10);
    localRoot.eidIndexToRankAddr[1] = MakeRankAddr("0/5", "plane_flat", 0x11);

    RootInfo peerRoot;
    peerRoot.deviceId = 0;
    peerRoot.localId = 4;
    peerRoot.eidIndexToRankAddr[2] = MakeRankAddr("4/8", "plane_flat", 0x20);
    peerRoot.eidIndexToRankAddr[3] = MakeRankAddr("4/9", "plane_flat", 0x21);

    TopoInfo topoInfo;
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 0, {"0/4", "0/5"}});
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 4, {"4/8", "4/9"}});
    if (GetParam()) {
        ASSERT_TRUE(ParseIndexedTopo(topoInfo));
        ASSERT_FALSE(topoInfo.closEdgeIndicesByLocalA.empty());
    }

    std::vector<uint32_t> rankToLocalId{0, 4};
    std::vector<std::vector<SyncEndpoint>> endpoints(2);
    endpoints[0].push_back(MakeEndpoint("server-id", "plane_flat", "0/4", 0, TopoFabric::Unknown));
    endpoints[0].push_back(MakeEndpoint("server-id", "plane_flat", "0/5", 1, TopoFabric::Unknown));
    endpoints[1].push_back(MakeEndpoint("server-id", "plane_flat", "4/8", 2, TopoFabric::Unknown));
    endpoints[1].push_back(MakeEndpoint("server-id", "plane_flat", "4/9", 3, TopoFabric::Unknown));

    TopoQuerier localQuerier(localRoot, topoInfo, 0, rankToLocalId, endpoints);
    TopoQuerier peerQuerier(peerRoot, topoInfo, 1, rankToLocalId, endpoints);

    std::vector<EidRoute> localRoutes;
    std::vector<EidRoute> peerRoutes;
    ASSERT_EQ(localQuerier.GetEidRoutes(1, 4, localRoutes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(peerQuerier.GetEidRoutes(0, 4, peerRoutes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(localRoutes.size(), 4U);
    ASSERT_EQ(peerRoutes.size(), 4U);

    const auto localPairs = ExtractCanonicalPairs(localRoot, peerRoot, localRoutes);
    const auto peerPairs = ExtractCanonicalPairs(peerRoot, localRoot, peerRoutes);
    ASSERT_EQ(localPairs.size(), 4U);
    ASSERT_EQ(peerPairs.size(), 4U);

    const std::vector<std::pair<std::string, std::string>> expected{
        {"0/4", "4/8"},
        {"0/4", "4/9"},
        {"0/5", "4/8"},
        {"0/5", "4/9"},
    };
    EXPECT_EQ(localPairs, expected);
    EXPECT_EQ(peerPairs, expected);
}

TEST_P(TopoReaderIndexedTest, TopoFileDescKeepsClosRoutesReciprocalWhenPortNamesMatch)
{
    RootInfo localRoot;
    localRoot.deviceId = 0;
    localRoot.localId = 0;
    localRoot.eidIndexToRankAddr[0] = MakeRankAddr("0/4", "plane_flat", 0x10);
    localRoot.eidIndexToRankAddr[1] = MakeRankAddr("0/5", "plane_flat", 0x11);

    RootInfo peerRoot;
    peerRoot.deviceId = 0;
    peerRoot.localId = 4;
    peerRoot.eidIndexToRankAddr[2] = MakeRankAddr("0/4", "plane_flat", 0x20);
    peerRoot.eidIndexToRankAddr[3] = MakeRankAddr("0/5", "plane_flat", 0x21);

    TopoInfo topoInfo;
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 0, {"0/4", "0/5"}});
    topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, 0, 4, {"0/4", "0/5"}});
    if (GetParam()) {
        ASSERT_TRUE(ParseIndexedTopo(topoInfo));
    }

    std::vector<uint32_t> rankToLocalId{0, 4};
    std::vector<std::vector<SyncEndpoint>> endpoints(2);
    endpoints[0].push_back(MakeEndpoint("server-id", "plane_flat", "0/4", 0, TopoFabric::Clos));
    endpoints[0].push_back(MakeEndpoint("server-id", "plane_flat", "0/5", 1, TopoFabric::Clos));
    endpoints[1].push_back(MakeEndpoint("server-id", "plane_flat", "0/4", 2, TopoFabric::Clos));
    endpoints[1].push_back(MakeEndpoint("server-id", "plane_flat", "0/5", 3, TopoFabric::Clos));

    TopoQuerier localQuerier(localRoot, topoInfo, 0, rankToLocalId, endpoints);
    TopoQuerier peerQuerier(peerRoot, topoInfo, 1, rankToLocalId, endpoints);
    std::vector<EidRoute> localRoutes;
    std::vector<EidRoute> peerRoutes;
    ASSERT_EQ(localQuerier.GetEidRoutes(1, 4, localRoutes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(peerQuerier.GetEidRoutes(0, 4, peerRoutes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(localRoutes.size(), 4U);
    ASSERT_EQ(peerRoutes.size(), 4U);

    for (size_t i = 0; i < localRoutes.size(); ++i) {
        EXPECT_EQ(localRoutes[i].localEidIndex, peerRoutes[i].remoteEidIndex);
        EXPECT_EQ(localRoutes[i].remoteEidIndex, peerRoutes[i].localEidIndex);
    }
}

TEST_P(TopoReaderIndexedTest, TopoFileDescDistributesRoutesAcrossDistinctAggregatedEidPairs)
{
    RootInfo localRoot;
    RootInfo peerRoot;
    TopoInfo topoInfo;
    std::vector<uint32_t> rankToLocalId{0, 4};
    std::vector<std::vector<SyncEndpoint>> endpoints(2);
    for (uint32_t idx = 0; idx < 2; ++idx) {
        const std::string plane = "plane_" + std::to_string(idx);
        const std::vector<std::string> localPorts{"0/" + std::to_string(2 * idx), "0/" + std::to_string(2 * idx + 1)};
        const std::vector<std::string> peerPorts{"4/" + std::to_string(2 * idx + 1), "4/" + std::to_string(2 * idx)};
        localRoot.eidIndexToRankAddr[idx] = MakeRankAddr(localPorts, plane, 0x10 + idx);
        peerRoot.eidIndexToRankAddr[idx + 2] = MakeRankAddr(peerPorts, plane, 0x20 + idx);
        endpoints[0].push_back(MakeEndpoint("server-id", plane, localPorts, idx, TopoFabric::Clos));
        endpoints[1].push_back(MakeEndpoint("server-id", plane, peerPorts, idx + 2, TopoFabric::Clos));
        topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, idx, 0, localPorts});
        topoInfo.closTopoEdges.push_back(ClosTopoEdge{0, idx, 4, peerPorts});
    }
    if (GetParam()) {
        ASSERT_TRUE(ParseIndexedTopo(topoInfo));
        ASSERT_FALSE(topoInfo.closEdgeIndicesByLocalA.empty());
    }

    TopoQuerier localQuerier(localRoot, topoInfo, 0, rankToLocalId, endpoints);
    TopoQuerier peerQuerier(peerRoot, topoInfo, 1, rankToLocalId, endpoints);
    std::vector<EidRoute> localRoutes;
    std::vector<EidRoute> peerRoutes;
    ASSERT_EQ(localQuerier.GetEidRoutes(1, 4, localRoutes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(peerQuerier.GetEidRoutes(0, 4, peerRoutes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(localRoutes.size(), 4U);
    ASSERT_EQ(peerRoutes.size(), 4U);
    for (uint32_t idx = 0; idx < 4; ++idx) {
        EXPECT_EQ(localRoutes[idx].localEidIndex, idx % 2);
        EXPECT_EQ(localRoutes[idx].remoteEidIndex, idx % 2 + 2);
        EXPECT_EQ(localRoutes[idx].localEidRaw, MakeEid(0x10 + idx % 2));
        EXPECT_EQ(peerRoutes[idx].localEidIndex, localRoutes[idx].remoteEidIndex);
        EXPECT_EQ(peerRoutes[idx].remoteEidIndex, localRoutes[idx].localEidIndex);
    }
}

TEST_P(TopoReaderIndexedTest, ExplicitClosKeepsNetLayerAndInstanceScope)
{
    RootInfo localRoot;
    RootInfo peerRoot;
    localRoot.localId = 0;
    peerRoot.localId = 4;
    localRoot.eidIndexToRankAddr[0] = MakeRankAddr("0/4", "plane_0", 0x10);
    localRoot.eidIndexToRankAddr[1] = MakeRankAddr("0/4", "plane_0", 0x11);
    peerRoot.eidIndexToRankAddr[2] = MakeRankAddr("0/4", "plane_0", 0x20);
    peerRoot.eidIndexToRankAddr[3] = MakeRankAddr("0/4", "plane_0", 0x21);

    std::vector<uint32_t> rankToLocalId{0, 4};
    std::vector<std::vector<SyncEndpoint>> endpoints(2);
    auto localLow = MakeEndpoint("pod_0", "plane_0", "0/4", 0, TopoFabric::Clos);
    auto peerLow = MakeEndpoint("pod_1", "plane_0", "0/4", 2, TopoFabric::Clos);
    auto localHigh = MakeEndpoint("cluster", "plane_0", "0/4", 1, TopoFabric::Clos);
    auto peerHigh = MakeEndpoint("cluster", "plane_0", "0/4", 3, TopoFabric::Clos);
    localLow.netType = peerLow.netType = localHigh.netType = peerHigh.netType = NetType::Clos;
    localLow.netLayer = peerLow.netLayer = 1;
    localHigh.netLayer = peerHigh.netLayer = 2;
    for (const bool reusePort : {false, true}) {
        const std::string highPort = reusePort ? "0/4" : "1/4";
        localHigh.ports = peerHigh.ports = {highPort};
        localRoot.eidIndexToRankAddr[1]->ports = {highPort};
        peerRoot.eidIndexToRankAddr[3]->ports = {highPort};
        TopoInfo scopedTopo;
        scopedTopo.closTopoEdges = {
            ClosTopoEdge{1, 10, 0, {"0/4"}}, ClosTopoEdge{1, 10, 4, {"0/4"}}, ClosTopoEdge{2, 20, 0, {highPort}},
            ClosTopoEdge{2, 20, 4, {highPort}}};
        if (GetParam()) {
            ASSERT_TRUE(ParseIndexedTopo(scopedTopo));
        }
        endpoints[0] = {localLow, localHigh};
        endpoints[1] = {peerLow, peerHigh};
        TopoQuerier localQuerier(localRoot, scopedTopo, 0, rankToLocalId, endpoints);
        TopoQuerier peerQuerier(peerRoot, scopedTopo, 1, rankToLocalId, endpoints);
        for (const uint32_t qpCount : {1U, 4U}) {
            std::vector<EidRoute> localRoutes;
            std::vector<EidRoute> peerRoutes;
            ASSERT_EQ(localQuerier.GetEidRoutes(1, qpCount, localRoutes), ACLSHMEM_SUCCESS);
            ASSERT_EQ(peerQuerier.GetEidRoutes(0, qpCount, peerRoutes), ACLSHMEM_SUCCESS);
            ASSERT_EQ(localRoutes.size(), qpCount);
            ASSERT_EQ(peerRoutes.size(), qpCount);
            for (size_t i = 0; i < localRoutes.size(); ++i) {
                EXPECT_EQ(localRoutes[i].localEidIndex, 1U);
                EXPECT_EQ(localRoutes[i].remoteEidIndex, 3U);
                EXPECT_EQ(peerRoutes[i].localEidIndex, 3U);
                EXPECT_EQ(peerRoutes[i].remoteEidIndex, 1U);
            }
        }
    }
}

TEST_P(TopoReaderIndexedTest, ExplicitClosRejectsEndpointsFromAnotherLayer)
{
    RootInfo localRoot;
    localRoot.localId = 0;
    localRoot.eidIndexToRankAddr[0] = MakeRankAddr("0/4", "plane_0", 0x10);
    TopoInfo topoInfo;
    topoInfo.closTopoEdges = {ClosTopoEdge{1, 10, 0, {"0/4"}}, ClosTopoEdge{1, 10, 4, {"0/4"}}};
    if (GetParam()) {
        ASSERT_TRUE(ParseIndexedTopo(topoInfo));
    }
    auto endpoint = MakeEndpoint("cluster", "plane_0", "0/4", 0, TopoFabric::Clos);
    endpoint.netType = NetType::Clos;
    endpoint.netLayer = 2;
    std::vector<uint32_t> rankToLocalId{0, 4};
    std::vector<std::vector<SyncEndpoint>> endpoints{{endpoint}, {endpoint}};
    TopoQuerier querier(localRoot, topoInfo, 0, rankToLocalId, endpoints);
    std::vector<EidRoute> routes;
    EXPECT_NE(querier.GetEidRoutes(1, 1, routes), ACLSHMEM_SUCCESS);
}

TEST_P(TopoReaderIndexedTest, ExplicitClosPrefersLowerLayerWithinSharedInstance)
{
    RootInfo localRoot;
    RootInfo peerRoot;
    localRoot.localId = 0;
    peerRoot.localId = 4;
    localRoot.eidIndexToRankAddr[0] = MakeRankAddr("0/4", "plane_low", 0x10);
    localRoot.eidIndexToRankAddr[1] = MakeRankAddr("1/4", "plane_high", 0x11);
    peerRoot.eidIndexToRankAddr[2] = MakeRankAddr("0/4", "plane_low", 0x20);
    peerRoot.eidIndexToRankAddr[3] = MakeRankAddr("1/4", "plane_high", 0x21);
    TopoInfo topoInfo;
    topoInfo.closTopoEdges = {
        ClosTopoEdge{1, 7, 0, {"0/4"}}, ClosTopoEdge{1, 7, 4, {"0/4"}}, ClosTopoEdge{2, 7, 0, {"1/4"}},
        ClosTopoEdge{2, 7, 4, {"1/4"}}};
    if (GetParam()) {
        ASSERT_TRUE(ParseIndexedTopo(topoInfo));
    }
    std::vector<uint32_t> rankToLocalId{0, 4};
    std::vector<std::vector<SyncEndpoint>> endpoints(2);
    auto low = MakeEndpoint("shared", "plane_low", "0/4", 0, TopoFabric::Clos);
    auto high = MakeEndpoint("shared", "plane_high", "1/4", 1, TopoFabric::Clos);
    low.netType = high.netType = NetType::Clos;
    low.netLayer = 1;
    high.netLayer = 2;
    endpoints[0] = {low, high};
    low.eidIndex = 2;
    high.eidIndex = 3;
    endpoints[1] = {low, high};
    TopoQuerier localQuerier(localRoot, topoInfo, 0, rankToLocalId, endpoints);
    std::vector<EidRoute> routes;
    ASSERT_EQ(localQuerier.GetEidRoutes(1, 1, routes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(routes.size(), 1U);
    EXPECT_EQ(routes[0].localEidIndex, 0U);
    EXPECT_EQ(routes[0].remoteEidIndex, 2U);
}

} // namespace transport
} // namespace shm
