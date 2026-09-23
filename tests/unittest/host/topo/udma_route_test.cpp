/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for details.
 */
#include <algorithm>
#include <map>
#include <gtest/gtest.h>
#include "transport/topo/topo_reader.h"

namespace shm::transport {
namespace {
class UdmaRouteTest : public testing::Test {
protected:
    RootInfo roots[2];
    TopoInfo topo;
    std::vector<uint32_t> localIds{7, 3};
    std::vector<std::vector<SyncEndpoint>> endpoints{2};

    void Add(
        NetType type, const std::string& plane, uint32_t a, uint32_t b, size_t portsA, size_t portsB,
        uint32_t layer = 1, const std::string& instance = "domain")
    {
        for (uint32_t rank = 0; rank < 2; ++rank) {
            SyncEndpoint ep{
                layer, instance,         type, type == NetType::Mesh ? TopoFabric::Mesh : TopoFabric::Clos, plane,
                {},    rank == 0 ? a : b};
            for (size_t p = 0; p < (rank == 0 ? portsA : portsB); ++p) {
                ep.ports.push_back(
                    plane + "-" + std::to_string(rank) + "-" + std::to_string(ep.eidIndex) + "-" + std::to_string(p));
            }
            auto addr = std::make_shared<RankAddr>();
            addr->eidData[15] = static_cast<uint8_t>(ep.eidIndex);
            addr->ports = ep.ports;
            roots[rank].eidIndexToRankAddr[ep.eidIndex] = addr;
            for (const auto& port : ep.ports) {
                roots[rank].portsToRankAddr[port] = addr;
            }
            endpoints[rank].push_back(ep);
        }
        if (type == NetType::Mesh) {
            topo.meshTopoEdges.push_back(
                {localIds[0], localIds[1], endpoints[0].back().ports, endpoints[1].back().ports});
        } else {
            const uint32_t topoInstanceId = static_cast<uint32_t>(topo.closTopoEdges.size() / 2);
            topo.closTopoEdges.push_back({layer, topoInstanceId, localIds[0], endpoints[0].back().ports});
            topo.closTopoEdges.push_back({layer, topoInstanceId, localIds[1], endpoints[1].back().ports});
        }
    }

    void Check(
        const std::map<uint32_t, uint32_t>& policy, uint32_t count, const std::vector<uint32_t>& expected,
        bool multiLink = true)
    {
        SCOPED_TRACE(testing::Message() << "route count=" << count);
        ASSERT_FALSE(expected.empty());
        // Each PE owns an independent process-wide policy snapshot. Equal effective sparse policies must resolve
        // reciprocal EID routes even though they are not represented by the same map object.
        const std::map<uint32_t, uint32_t> forwardPolicy = policy;
        const std::map<uint32_t, uint32_t> reversePolicy = policy;
        TopoQuerier forward(roots[0], topo, 0, localIds, endpoints, forwardPolicy, multiLink);
        TopoQuerier reverse(roots[1], topo, 1, localIds, endpoints, reversePolicy, multiLink);
        std::vector<EidRoute> routes, back;
        ASSERT_EQ(forward.GetEidRoutes(1, count, routes), ACLSHMEM_SUCCESS);
        ASSERT_EQ(reverse.GetEidRoutes(0, count, back), ACLSHMEM_SUCCESS);
        ASSERT_EQ(routes.size(), count);
        ASSERT_EQ(back.size(), count);
        for (uint32_t q = 0; q < count; ++q) {
            EXPECT_EQ(routes[q].localEidIndex, expected[q % expected.size()]) << q;
            EXPECT_EQ(routes[q].localEidIndex, back[q].remoteEidIndex) << q;
            EXPECT_EQ(routes[q].remoteEidIndex, back[q].localEidIndex) << q;
            EXPECT_EQ(routes[q].localEidRaw[15], routes[q].localEidIndex);
            EXPECT_EQ(routes[q].portCount, back[q].portCount);
            EXPECT_EQ(routes[q].netType, back[q].netType);
            EXPECT_EQ(routes[q].netLayer, back[q].netLayer);
            EXPECT_EQ(routes[q].netInstanceId, back[q].netInstanceId);
            EXPECT_EQ(routes[q].planeId, back[q].planeId);
        }
        uint32_t local = 0, remote = 0;
        EidData raw{};
        ASSERT_TRUE(forward.GetEidRoute(1, local, raw, remote));
        TopoQuerier defaultForward(roots[0], topo, 0, localIds, endpoints, {}, multiLink);
        std::vector<EidRoute> defaultRoutes;
        ASSERT_EQ(defaultForward.GetEidRoutes(1, 1, defaultRoutes), ACLSHMEM_SUCCESS);
        ASSERT_EQ(defaultRoutes.size(), 1U);
        EXPECT_EQ(local, defaultRoutes.front().localEidIndex);
        EXPECT_EQ(remote, defaultRoutes.front().remoteEidIndex);
    }
};

TEST_F(UdmaRouteTest, UbxDefaultMixedAndRepeatedClos)
{
    for (auto& root : roots) {
        root.levels.push_back(std::make_shared<LevelInfo>(LevelInfo{0, "domain", NetType::TopoFileDesc, ""}));
    }
    Add(NetType::Mesh, "mesh", 0, 10, 1, 1, 0);
    // UBX rootinfo/topoinfo carries both fabrics in raw layer 0. Public QP
    // policy and route metadata expose the effective Mesh=0/CLOS=1 layers.
    Add(NetType::Clos, "clos", 1, 11, 4, 4, 0);
    for (auto& rankEndpoints : endpoints) {
        for (auto& endpoint : rankEndpoints) {
            endpoint.netType = NetType::TopoFileDesc;
        }
    }
    for (uint32_t q : {1U, 2U, 4U, 5U}) {
        Check({}, q, {0}, false);
        Check({}, q, {0, 1});
    }
    Check({{0, 0}, {1, 1}}, 2, {0, 1});
    Check({{0, 1}, {1, 0}}, 2, {1, 0});
    Check({{0, 0}, {1, 0}}, 2, {0, 0});
    Check({{1, 1}}, 3, {0, 1, 0});
    TopoQuerier querier(roots[0], topo, 0, localIds, endpoints, {}, true);
    std::vector<EidRoute> routes;
    ASSERT_EQ(querier.GetEidRoutes(1, 1, routes), ACLSHMEM_SUCCESS);
    EXPECT_EQ(routes[0].netInstanceId, "domain");
    EXPECT_EQ(routes[0].netType, NetType::Mesh);
    EXPECT_EQ(routes[0].netLayer, 0U);
    EXPECT_EQ(routes[0].planeId, "mesh");
    EXPECT_EQ(routes[0].portCount, 1U);

    ASSERT_EQ(querier.GetEidRoutes(1, 2, routes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(routes.size(), 2U);
    EXPECT_EQ(routes[0].netType, NetType::Mesh);
    EXPECT_EQ(routes[0].netLayer, 0U);
    EXPECT_EQ(routes[1].netType, NetType::Clos);
    EXPECT_EQ(routes[1].netLayer, 1U);
}

TEST_F(UdmaRouteTest, PodSortsShuffledPlanesAndKeepsCustomSelection)
{
    Add(NetType::Clos, "small", 2, 12, 2, 2);
    Add(NetType::Mesh, "mesh", 0, 10, 1, 1, 0);
    Add(NetType::Clos, "large", 1, 11, 6, 6);
    std::reverse(endpoints[1].begin(), endpoints[1].end());
    for (uint32_t q : {1U, 2U, 3U, 6U, 7U}) {
        Check({}, q, {0}, false);
        Check({}, q, {0, 1, 2});
    }
    Check({{0, 0}, {1, 1}, {2, 1}}, 3, {0, 1, 2});
    Check({{2, 1}, {0, 0}, {1, 1}}, 3, {0, 1, 2});
    Check({{0, 1}, {1, 1}, {2, 1}, {3, 1}}, 4, {1, 2, 1, 2});
    Check({{0, 1}}, 6, {1, 1, 2, 0, 1, 2});
}

TEST_F(UdmaRouteTest, ClosOnlyUsesAllCandidatesByDefault)
{
    Add(NetType::Clos, "small", 2, 12, 2, 2);
    Add(NetType::Clos, "large", 1, 11, 6, 6);
    for (uint32_t q : {1U, 2U, 3U, 6U, 7U}) {
        Check({}, q, {1, 2});
    }
}

TEST_F(UdmaRouteTest, MultiLinkOffDeduplicatesMetadataVariantsOfTheSameClosRoute)
{
    Add(NetType::Clos, "shared", 1, 11, 2, 2, 1, "instance-a");
    Add(NetType::Clos, "shared", 1, 11, 2, 2, 2, "instance-b");

    TopoQuerier querier(roots[0], topo, 0, localIds, endpoints, {}, false);
    std::vector<EidRoute> routes;
    ASSERT_EQ(querier.GetEidRoutes(1, 4, routes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(routes.size(), 4U);
    for (const auto& route : routes) {
        EXPECT_EQ(route.localEidIndex, 1U);
        EXPECT_EQ(route.remoteEidIndex, 11U);
        EXPECT_EQ(route.netLayer, 1U);
        EXPECT_EQ(route.netInstanceId, "instance-a");
        EXPECT_EQ(route.planeId, "shared");
    }
}

TEST_F(UdmaRouteTest, ExplicitLayerMatchesActualRouteLayer)
{
    Add(NetType::Mesh, "mesh", 0, 10, 1, 1, 0);
    Add(NetType::Clos, "layer-one", 1, 11, 2, 2, 1);
    Add(NetType::Clos, "layer-two", 2, 12, 2, 2, 2);
    Check({{0, 1}, {1, 2}, {2, 1}}, 3, {1, 2, 1});
}

TEST_F(UdmaRouteTest, IndependentMatchingPePoliciesResolveReciprocalRoutes)
{
    Add(NetType::Mesh, "mesh", 0, 10, 1, 1, 0);
    Add(NetType::Clos, "clos", 1, 11, 2, 2, 1);
    const std::map<uint32_t, uint32_t> pe0Policy{{0, 0}, {2, 1}};
    std::map<uint32_t, uint32_t> pe1Policy;
    pe1Policy.emplace(2, 1);
    pe1Policy.emplace(0, 0);

    TopoQuerier pe0(roots[0], topo, 0, localIds, endpoints, pe0Policy, true);
    TopoQuerier pe1(roots[1], topo, 1, localIds, endpoints, pe1Policy, true);
    std::vector<EidRoute> routes0;
    std::vector<EidRoute> routes1;
    ASSERT_EQ(pe0.GetEidRoutes(1, 3, routes0), ACLSHMEM_SUCCESS);
    ASSERT_EQ(pe1.GetEidRoutes(0, 3, routes1), ACLSHMEM_SUCCESS);
    ASSERT_EQ(routes0.size(), routes1.size());
    for (uint32_t qpIdx = 0; qpIdx < routes0.size(); ++qpIdx) {
        EXPECT_EQ(routes0[qpIdx].localEidIndex, routes1[qpIdx].remoteEidIndex) << qpIdx;
        EXPECT_EQ(routes0[qpIdx].remoteEidIndex, routes1[qpIdx].localEidIndex) << qpIdx;
        EXPECT_EQ(routes0[qpIdx].netLayer, routes1[qpIdx].netLayer) << qpIdx;
    }
}

TEST_F(UdmaRouteTest, ClosMetadataComesFromResolvedEndpointWhenEidIsShared)
{
    Add(NetType::Clos, "clos", 1, 11, 2, 2, 1, "resolved-instance");
    for (uint32_t rank = 0; rank < 2; ++rank) {
        SyncEndpoint conflicting = endpoints[rank][0];
        conflicting.netInstanceId = "wrong-instance";
        conflicting.ports = {"unrelated-" + std::to_string(rank)};
        endpoints[rank].insert(endpoints[rank].begin(), std::move(conflicting));
    }

    TopoQuerier querier(roots[0], topo, 0, localIds, endpoints, {}, true);
    std::vector<EidRoute> routes;
    ASSERT_EQ(querier.GetEidRoutes(1, 1, routes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(routes.size(), 1U);
    EXPECT_EQ(routes[0].netInstanceId, "resolved-instance");
}

TEST_F(UdmaRouteTest, MeshMetadataComesFromResolvedEndpointWhenEidIsShared)
{
    Add(NetType::Mesh, "mesh", 0, 10, 1, 1, 0, "resolved-instance");
    for (uint32_t rank = 0; rank < 2; ++rank) {
        SyncEndpoint conflicting = endpoints[rank][0];
        conflicting.netInstanceId = "wrong-instance";
        conflicting.planeId = "wrong-plane";
        endpoints[rank][0].ports.push_back("preferred-" + std::to_string(rank));
        endpoints[rank].insert(endpoints[rank].begin(), std::move(conflicting));
    }

    TopoQuerier querier(roots[0], topo, 0, localIds, endpoints, {}, true);
    std::vector<EidRoute> routes;
    ASSERT_EQ(querier.GetEidRoutes(1, 1, routes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(routes.size(), 1U);
    EXPECT_EQ(routes[0].netType, NetType::Mesh);
    EXPECT_EQ(routes[0].netInstanceId, "resolved-instance");
    EXPECT_EQ(routes[0].planeId, "mesh");
}

TEST_F(UdmaRouteTest, CrossDomainIgnoresExplicitNetLayerPolicy)
{
    Add(NetType::Mesh, "mesh", 0, 10, 1, 1, 0, "domain-a");
    Add(NetType::Clos, "clos", 1, 11, 2, 2, 1, "domain-a");
    endpoints[1][0].netInstanceId = "domain-b";

    Check({{0, 0}}, 3, {1});
}

TEST_F(UdmaRouteTest, UnknownInstanceUsesExplicitMeshOnlyForDirectPeer)
{
    Add(NetType::Mesh, "mesh", 0, 10, 1, 1, 0, "unknown");
    Add(NetType::Clos, "clos", 1, 11, 2, 2, 1, "clos-domain");
    Check({{0, 0}}, 1, {0});

    topo.meshTopoEdges[0].localA = 1;
    topo.meshTopoEdges[0].localB = 2;
    Check({{0, 0}}, 1, {1});
}

TEST_F(UdmaRouteTest, MeshMetadataAndDomainIgnoreClosEndpointSharingEid)
{
    Add(NetType::Mesh, "mesh", 0, 10, 1, 1, 0, "mesh-domain");
    Add(NetType::Clos, "clos", 1, 11, 2, 2, 1, "mesh-domain");
    for (uint32_t rank = 0; rank < 2; ++rank) {
        SyncEndpoint conflicting = endpoints[rank][0];
        conflicting.netType = NetType::Clos;
        conflicting.fabric = TopoFabric::Clos;
        conflicting.netInstanceId = "wrong-domain";
        conflicting.planeId = "wrong-plane";
        endpoints[rank].insert(endpoints[rank].begin(), std::move(conflicting));
    }

    Check({{0, 0}, {1, 1}}, 2, {0, 1});
    TopoQuerier querier(roots[0], topo, 0, localIds, endpoints, {}, true);
    std::vector<EidRoute> routes;
    ASSERT_EQ(querier.GetEidRoutes(1, 1, routes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(routes.size(), 1U);
    EXPECT_EQ(routes[0].netType, NetType::Mesh);
    EXPECT_EQ(routes[0].netInstanceId, "mesh-domain");
    EXPECT_EQ(routes[0].planeId, "mesh");
}

TEST_F(UdmaRouteTest, AsymmetricPortsUseMinimumAndIgnoreUnsharedPlanes)
{
    Add(NetType::Clos, "six-to-one", 1, 11, 6, 1);
    Add(NetType::Clos, "two", 2, 12, 2, 2);
    Add(NetType::Clos, "unshared", 3, 13, 8, 8);
    endpoints[1].back().netInstanceId = "other-domain";
    Check({}, 5, {2, 1});
    TopoQuerier querier(roots[0], topo, 0, localIds, endpoints);
    std::vector<EidRoute> routes;
    ASSERT_EQ(querier.GetEidRoutes(1, 2, routes), ACLSHMEM_SUCCESS);
    EXPECT_EQ(routes[0].portCount, 2U);
    EXPECT_EQ(routes[1].portCount, 1U);
}

TEST_F(UdmaRouteTest, EqualPortCountsUseLayerInstanceAndPlaneOrder)
{
    Add(NetType::Clos, "b", 4, 14, 2, 2, 2, "a");
    Add(NetType::Clos, "a", 3, 13, 2, 2, 2, "a");
    Add(NetType::Clos, "a", 2, 12, 2, 2, 2, "0");
    Add(NetType::Clos, "a", 1, 11, 2, 2, 1, "z");
    Check({}, 7, {1, 4, 3, 2});
    Check({}, 4, {1, 4, 3, 2});
}

TEST_F(UdmaRouteTest, CrossDomainDefaultExplicitlyExcludesLegacyMeshCandidate)
{
    Add(NetType::Mesh, "mesh", 0, 10, 1, 1, 0, "");
    Add(NetType::Clos, "clos", 1, 11, 2, 2, 1, "clos-domain");

    TopoQuerier querier(roots[0], topo, 0, localIds, endpoints, {}, true);
    std::vector<EidRoute> routes;
    ASSERT_EQ(querier.GetEidRoutes(1, 3, routes), ACLSHMEM_SUCCESS);
    ASSERT_EQ(routes.size(), 3U);
    for (const auto& route : routes) {
        EXPECT_EQ(route.netType, NetType::Clos);
        EXPECT_EQ(route.netLayer, 1U);
    }
}

TEST_F(UdmaRouteTest, RejectsIpAddressCandidatesForDirectUdma)
{
    Add(NetType::Clos, "roce", 1, 11, 2, 2, 1);
    endpoints[0][0].addrType = RankAddrType::IPV4;
    endpoints[1][0].addrType = RankAddrType::IPV4;

    TopoQuerier querier(roots[0], topo, 0, localIds, endpoints, {}, true);
    std::vector<EidRoute> routes;
    EXPECT_EQ(querier.GetEidRoutes(1, 1, routes), ACLSHMEM_INNER_ERROR);
    EXPECT_TRUE(routes.empty());
}

TEST_F(UdmaRouteTest, RejectsUnavailableNetLayerInvalidRankAndZeroCount)
{
    Add(NetType::Mesh, "mesh", 0, 10, 1, 1, 0);
    std::vector<EidRoute> routes(3);
    TopoQuerier meshOnly(roots[0], topo, 0, localIds, endpoints, {{0, 1}}, true);
    EXPECT_EQ(meshOnly.GetEidRoutes(1, 2, routes), ACLSHMEM_INVALID_VALUE);
    EXPECT_TRUE(routes.empty());
    uint32_t local = 0;
    uint32_t remote = 0;
    EidData raw{};
    EXPECT_TRUE(meshOnly.GetEidRoute(1, local, raw, remote));
    EXPECT_EQ(local, 0U);
    EXPECT_EQ(remote, 10U);
    TopoQuerier invalid(roots[0], topo, 0, localIds, endpoints, {{0, 99}}, true);
    EXPECT_EQ(invalid.GetEidRoutes(1, 2, routes), ACLSHMEM_INVALID_VALUE);
    EXPECT_TRUE(routes.empty());
    routes.resize(3);
    EXPECT_EQ(invalid.GetEidRoutes(2, 2, routes), ACLSHMEM_INVALID_VALUE);
    EXPECT_TRUE(routes.empty());
    routes.resize(3);
    EXPECT_EQ(invalid.GetEidRoutes(1, 0, routes), ACLSHMEM_INVALID_VALUE);
    EXPECT_TRUE(routes.empty());
}
} // namespace
} // namespace shm::transport
