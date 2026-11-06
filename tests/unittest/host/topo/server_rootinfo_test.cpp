/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See the License root of the software repository for the full text of the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "aclshmemi_product_strategy.h"
#include "aclshmemi_eid_parser.h"
#include "mock_hal.h"
#include "transport/topo/topo_reader.h"

namespace shm {
namespace topo {

using testing::_;
using testing::Invoke;
using testing::Return;

class ServerRootinfoTest : public testing::Test {
protected:
    void SetUp() override
    {
        testing::Mock::AllowLeak(&MockHal::instance());
        ON_CALL(MockHal::instance(), get_driver_install_path()).WillByDefault(Return("/driver-root"));
        ON_CALL(MockHal::instance(), get_server_id(_)).WillByDefault(Return("sp_42_srv_1"));
    }
    void TearDown() override { testing::Mock::VerifyAndClear(&MockHal::instance()); }

    static void add_eid(UBEntity& ue, int die, int fe, int port, uint8_t flags = 0)
    {
        auto& eid = ue.eidList[ue.eidNum++].eid;
        eid = {};
        eid.raw[5] = static_cast<uint8_t>((die << 6) | port);
        eid.raw[6] = static_cast<uint8_t>(fe);
        eid.raw[7] = flags;
    }

    void prepare_entities(const UEList& entities, bool query_spod = true)
    {
        EXPECT_CALL(MockHal::instance(), get_ue_list(_, _)).WillOnce(Invoke([entities](int, UEList* list) {
            *list = entities;
            return 0;
        }));
        if (query_spod) {
            EXPECT_CALL(MockHal::instance(), get_spod_info(_, _)).WillOnce(Invoke([](int, dcmi_spod_info* info) {
                *info = {};
                info->super_pod_id = 42;
                info->server_index = 1;
                return 0;
            }));
        } else {
            EXPECT_CALL(MockHal::instance(), get_spod_info(_, _)).Times(0);
        }
    }

    void prepare(uint32_t server, uint32_t type, bool query_spod = true)
    {
        if (query_spod) {
            EXPECT_CALL(MockHal::instance(), get_spod_info(_, _)).WillOnce(Invoke([=](int, dcmi_spod_info* info) {
                *info = {};
                info->super_pod_id = 42;
                info->server_index = server;
                info->super_pod_type = type;
                return 0;
            }));
        } else {
            EXPECT_CALL(MockHal::instance(), get_spod_info(_, _)).Times(0);
        }
        EXPECT_CALL(MockHal::instance(), get_ue_list(_, _)).WillOnce(Invoke([](int, UEList* list) {
            *list = {};
            list->ueNum = 5;
            add_eid(list->ueList[0], 0, 3, 1);
            add_eid(list->ueList[0], 0, 3, 63);
            add_eid(list->ueList[1], 1, 3, 2);
            add_eid(list->ueList[1], 1, 3, 63);
            add_eid(list->ueList[2], 0, 2, 4);
            add_eid(list->ueList[2], 0, 2, 63);
            add_eid(list->ueList[3], 1, 2, 5);
            add_eid(list->ueList[3], 1, 2, 63);
            add_eid(list->ueList[4], 0, 10, 8, 0xc0);
            auto& ip = list->ueList[4].eidList[0].eid;
            ip.raw[12] = 1;
            ip.raw[13] = 2;
            ip.raw[14] = 3;
            return 0;
        }));
    }
};

TEST_F(ServerRootinfoTest, Pc16SharesSuperpodButHasDistinctLocalIds)
{
    for (uint32_t server = 0; server < 2; ++server) {
        prepare(server, ACLSHMEMI_TOPO_TYPE_SERVER_16FM);
        auto strategy = aclshmemi_product_strategy_t::create(0x27);
        ASSERT_NE(strategy, nullptr);
        auto root = strategy->get_root_info(3, 0x27);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->topo_file_path, "/driver-root/driver/topo/950/atlas_850_2.json");
        ASSERT_EQ(root->ranks.size(), 1U);
        const auto& rank = root->ranks[0];
        EXPECT_EQ(rank.local_id, 3 + server * 8);
        ASSERT_EQ(rank.level_list.size(), 2U);
        const auto& mesh = rank.level_list[0];
        EXPECT_EQ(mesh.instance_id, "sp_42");
        EXPECT_EQ(mesh.net_type, "TOPO_FILE_DESC");
        ASSERT_EQ(mesh.addr_list.size(), 2U);
        EXPECT_THAT(mesh.addr_list[0].ports, testing::ElementsAre("0/1"));
        EXPECT_THAT(mesh.addr_list[1].ports, testing::ElementsAre("1/2"));
        EXPECT_EQ(rank.level_list[1].level, 2);
        ASSERT_EQ(rank.level_list[1].addr_list.size(), 1U);
        EXPECT_EQ(rank.level_list[1].addr_list[0].addr, "10.1.2.3");
    }
}

TEST_F(ServerRootinfoTest, OsScopeQueriesIdentityForTheRequestedDevice)
{
    for (uint32_t board : {0x44U, 0x29U, 0x2bU}) {
        SCOPED_TRACE(board);
        UEList entities{};
        entities.ueNum = 1;
        add_eid(entities.ueList[0], 1, 3, 1);
        prepare_entities(entities, false);
        EXPECT_CALL(MockHal::instance(), get_server_id(3)).WillOnce(Return("sp_7_srv_2"));
        auto strategy = aclshmemi_product_strategy_t::create(board);
        ASSERT_NE(strategy, nullptr);
        auto root = strategy->get_root_info(3, board);
        ASSERT_TRUE(root);
        ASSERT_EQ(root->ranks.size(), 1U);
        ASSERT_FALSE(root->ranks[0].level_list.empty());
        EXPECT_EQ(root->ranks[0].level_list[0].instance_id, "sp_7_srv_2");
    }
}

TEST_F(ServerRootinfoTest, NoSuperpodBoardsUseOsIdentityWithoutQueryingSpod)
{
    for (uint32_t board :
         {ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP, ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE}) {
        SCOPED_TRACE(board);
        UEList entities{};
        entities.ueNum = 1;
        add_eid(entities.ueList[0], 1, 3, 2);
        EXPECT_CALL(MockHal::instance(), get_ue_list(3, _)).WillOnce(Invoke([entities](int, UEList* list) {
            *list = entities;
            return 0;
        }));
        EXPECT_CALL(MockHal::instance(), get_spod_info(_, _)).Times(0);
        EXPECT_CALL(MockHal::instance(), get_server_id(3)).WillOnce(Return("os_test-boot-id"));
        auto strategy = aclshmemi_product_strategy_t::create(board);
        ASSERT_NE(strategy, nullptr);
        auto root = strategy->get_root_info(3, board);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->topo_file_path, "/driver-root/driver/topo/950/atlas_850_1.json");
        ASSERT_EQ(root->ranks.size(), 1U);
        EXPECT_EQ(root->ranks[0].local_id, 3);
        ASSERT_EQ(root->ranks[0].level_list.size(), 1U);
        const auto& mesh = root->ranks[0].level_list[0];
        EXPECT_EQ(mesh.level, 0);
        EXPECT_EQ(mesh.instance_id, "os_test-boot-id");
        EXPECT_EQ(mesh.net_type, "TOPO_FILE_DESC");
        ASSERT_EQ(mesh.addr_list.size(), 1U);
        EXPECT_THAT(mesh.addr_list[0].ports, testing::ElementsAre("1/2"));
    }
}

TEST_F(ServerRootinfoTest, SuperpodScopesStillRejectSpodQueryFailure)
{
    for (uint32_t board : {0x23U, 0x25U, 0x27U, 0x2dU, 0x2fU, 0x46U, 0x48U}) {
        SCOPED_TRACE(board);
        EXPECT_CALL(MockHal::instance(), get_ue_list(3, _)).WillOnce(Return(0));
        EXPECT_CALL(MockHal::instance(), get_spod_info(3, _)).WillOnce(Return(-1));
        EXPECT_CALL(MockHal::instance(), get_server_id(_)).Times(0);
        auto strategy = aclshmemi_product_strategy_t::create(board);
        ASSERT_NE(strategy, nullptr);
        EXPECT_FALSE(strategy->get_root_info(3, board));
        testing::Mock::VerifyAndClearExpectations(&MockHal::instance());
    }
}

TEST_F(ServerRootinfoTest, CardIdentityPreservesTwoCardGroupSuffix)
{
    for (uint32_t board : {ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH, ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH}) {
        for (int phy_id : {3, 4}) {
            SCOPED_TRACE(testing::Message() << board << ", phy_id=" << phy_id);
            UEList entities{};
            entities.ueNum = 1;
            add_eid(entities.ueList[0], 0, 0, 1);
            entities.ueList[0].eidList[0].eid.raw[15] = board == ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH ? 0x50 : 0x08;
            prepare_entities(entities, false);
            EXPECT_CALL(MockHal::instance(), get_server_id(phy_id)).WillOnce(Return("sp_7_srv_2"));
            auto strategy = aclshmemi_product_strategy_t::create(board);
            ASSERT_NE(strategy, nullptr);
            auto root = strategy->get_root_info_without_roce(phy_id, board);
            ASSERT_TRUE(root);
            ASSERT_EQ(root->ranks.size(), 1U);
            ASSERT_EQ(root->ranks[0].level_list.size(), 1U);
            const auto expected = board == ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH ?
                                      "sp_7_srv_2_" + std::to_string(phy_id / 2) :
                                      std::string("sp_7_srv_2");
            EXPECT_EQ(root->ranks[0].level_list[0].instance_id, expected);
            ASSERT_EQ(root->ranks[0].level_list[0].addr_list.size(), 1U);
        }
    }
}

TEST_F(ServerRootinfoTest, MissingServerIdentityFailsGeneration)
{
    for (uint32_t board : {0x44U, 0x29U, 0x2bU, 0x6aU, 0x6cU}) {
        SCOPED_TRACE(board);
        UEList entities{};
        entities.ueNum = 1;
        add_eid(entities.ueList[0], 1, 3, 1);
        prepare_entities(entities, false);
        EXPECT_CALL(MockHal::instance(), get_server_id(0)).WillOnce(Return(std::nullopt));
        auto strategy = aclshmemi_product_strategy_t::create(board);
        ASSERT_NE(strategy, nullptr);
        EXPECT_FALSE(strategy->get_root_info_without_roce(0, board));
    }
}

TEST_F(ServerRootinfoTest, Board2bUsesOsMeshAndUboeWithoutSuperplane)
{
    prepare(1, 0, false);
    auto strategy = aclshmemi_product_strategy_t::create(0x2b);
    ASSERT_NE(strategy, nullptr);
    auto root = strategy->get_root_info(0, 0x2b);
    ASSERT_TRUE(root);
    EXPECT_EQ(root->topo_file_path, "/driver-root/driver/topo/950/atlas_850_1.json");
    ASSERT_EQ(root->ranks.size(), 1U);
    const auto& layers = root->ranks[0].level_list;
    ASSERT_EQ(layers.size(), 2U);
    EXPECT_EQ(layers[0].instance_id, "sp_42_srv_1");
    EXPECT_EQ(layers[0].level, 0);
    EXPECT_EQ(layers[1].level, 2);
    EXPECT_EQ(layers[1].instance_id, "cluster");
    ASSERT_EQ(layers[1].addr_list.size(), 1U);
    EXPECT_EQ(layers[1].addr_list[0].addr_type, "IPV4");
    EXPECT_EQ(layers[1].addr_list[0].plane_id, "plane_uboe");
}

TEST_F(ServerRootinfoTest, Board27OrdinaryServerKeepsSeparateSuperplane)
{
    prepare(1, 0);
    aclshmemi_server_product_t strategy;
    auto root = strategy.get_root_info(0, 0x27);
    ASSERT_TRUE(root);
    ASSERT_EQ(root->ranks.size(), 1U);
    const auto& layers = root->ranks[0].level_list;
    ASSERT_EQ(layers.size(), 3U);
    EXPECT_EQ(layers[0].instance_id, "sp_42_srv_1");
    EXPECT_EQ(layers[1].instance_id, "sp_42");
    ASSERT_EQ(layers[1].addr_list.size(), 1U);
    EXPECT_THAT(layers[1].addr_list[0].ports, testing::ElementsAre("0/1"));
    EXPECT_EQ(layers[2].level, 2);
}

TEST_F(ServerRootinfoTest, GeneratedUboeAddressIsNotUsedForRoce)
{
    for (uint32_t board : {0x27U, 0x2bU}) {
        SCOPED_TRACE(board);
        prepare(0, ACLSHMEMI_TOPO_TYPE_SERVER_8P, board != ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE);
        auto strategy = aclshmemi_product_strategy_t::create(board);
        ASSERT_NE(strategy, nullptr);
        auto root = strategy->get_root_info(0, board);
        ASSERT_TRUE(root);
        const auto json = nlohmann::json::parse(aclshmemi_root_info_to_string(*root));
        bool has_uboe = false;
        for (const auto& layer : root->ranks[0].level_list) {
            for (const auto& address : layer.addr_list) {
                has_uboe = has_uboe || (address.plane_id == "plane_uboe" && address.addr_type == "IPV4");
            }
        }
        ASSERT_TRUE(has_uboe);
        net_addr_t ip;
        EXPECT_FALSE(shm::transport::TopoReader::ParseRdmaNetAddrJson(json, 0, ip));
    }
}

TEST_F(ServerRootinfoTest, RoceAddressIsSelectedRegardlessOfUboeLayerOrder)
{
    for (const char* plane : {"roce", "plane0"}) {
        for (bool roce_first : {false, true}) {
            SCOPED_TRACE(testing::Message() << plane << ", roce_first=" << roce_first);
            prepare(0, ACLSHMEMI_TOPO_TYPE_SERVER_8P, false);
            aclshmemi_server_product_t strategy;
            auto root = strategy.get_root_info(0, 0x2b);
            ASSERT_TRUE(root);
            auto json = nlohmann::json::parse(aclshmemi_root_info_to_string(*root));
            const nlohmann::json roce_layer = {
                {"net_layer", 3},
                {"net_type", "CLOS"},
                {"net_instance_id", "cluster"},
                {"rank_addr_list",
                 {{{"addr_type", "IPV4"}, {"addr", "192.0.2.9"}, {"plane_id", plane}, {"ports", {"d2h"}}}}}};
            auto& layers = json["rank_list"][0]["level_list"];
            if (roce_first) {
                layers.insert(layers.begin(), roce_layer);
            } else {
                layers.push_back(roce_layer);
            }
            net_addr_t ip;
            ASSERT_TRUE(shm::transport::TopoReader::ParseRdmaNetAddrJson(json, 0, ip));
            EXPECT_EQ(ip.type, IpV4);
            in_addr expected{};
            ASSERT_EQ(inet_pton(AF_INET, "192.0.2.9", &expected), 1);
            EXPECT_EQ(ip.ip.ipv4.s_addr, expected.s_addr);
        }
    }
}

TEST_F(ServerRootinfoTest, RoceAddressWithoutPlaneRetainsLegacyBehavior)
{
    const nlohmann::json json = {
        {"rank_list",
         {{{"device_id", 0},
           {"local_id", 0},
           {"level_list",
            {{{"net_layer", 3},
              {"net_type", "CLOS"},
              {"rank_addr_list", {{{"addr_type", "IPV6"}, {"addr", "2001:db8::9"}}}}}}}}}}};
    net_addr_t ip;
    ASSERT_TRUE(shm::transport::TopoReader::ParseRdmaNetAddrJson(json, 0, ip));
    EXPECT_EQ(ip.type, IpV6);
    in6_addr expected{};
    ASSERT_EQ(inet_pton(AF_INET6, "2001:db8::9", &expected), 1);
    EXPECT_EQ(memcmp(&ip.ip.ipv6, &expected, sizeof(expected)), 0);
}

TEST_F(ServerRootinfoTest, FlexBoardsHaveSeparateEightNpuDomains)
{
    for (uint32_t board : {0x2dU, 0x2fU}) {
        prepare(2, 0);
        auto strategy = aclshmemi_product_strategy_t::create(board);
        ASSERT_NE(strategy, nullptr);
        ASSERT_NE(dynamic_cast<aclshmemi_pod_flex_product_t*>(strategy.get()), nullptr);
        auto root = strategy->get_root_info(9, board);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->topo_file_path, "/driver-root/driver/topo/950/atlas_950_2.json");
        ASSERT_EQ(root->ranks.size(), 1U);
        ASSERT_FALSE(root->ranks[0].level_list.empty());
        EXPECT_EQ(root->ranks[0].level_list[0].instance_id, "sp_42_srv_2_board1");
    }
}

TEST_F(ServerRootinfoTest, El550BoardsUseBothClosLevels)
{
    for (uint32_t board : {0x46U, 0x48U}) {
        prepare(0, 0);
        auto strategy = aclshmemi_product_strategy_t::create(board);
        ASSERT_NE(strategy, nullptr);
        auto root = strategy->get_root_info(0, board);
        ASSERT_TRUE(root);
        ASSERT_EQ(root->ranks.size(), 1U);
        const auto& layers = root->ranks[0].level_list;
        ASSERT_EQ(layers.size(), 2U);
        EXPECT_EQ(layers[0].instance_id, "sp_42_srv_0");
        EXPECT_EQ(layers[1].instance_id, "sp_42");
        EXPECT_EQ(layers[0].addr_list.size(), board == 0x46 ? 1U : 2U);
    }
}

TEST_F(ServerRootinfoTest, El550ClosLayersShareTheClosQpPolicy)
{
    using namespace shm::transport;
    for (uint32_t board : {0x46U, 0x48U}) {
        SCOPED_TRACE(board);
        std::vector<RootInfo> roots(2);
        std::vector<std::vector<SyncEndpoint>> endpoints(2);
        TopoInfo topo;
        const std::vector<uint32_t> local_ids{0, 1};
        for (uint32_t peer = 0; peer < 2; ++peer) {
            prepare(0, 0);
            auto strategy = aclshmemi_product_strategy_t::create(board);
            auto generated = strategy->get_root_info(peer, board);
            ASSERT_TRUE(generated);
            // Build the routing fixture from the generated layers without file I/O.
            uint32_t eid_index = 0;
            for (const auto& generated_layer : generated->ranks[0].level_list) {
                ASSERT_EQ(generated_layer.net_type, "CLOS");
                auto level = std::make_shared<LevelInfo>(LevelInfo{
                    static_cast<uint32_t>(generated_layer.level), generated_layer.instance_id, NetType::Clos, ""});
                roots[peer].levels.push_back(level);
                for (const auto& generated_addr : generated_layer.addr_list) {
                    auto addr = std::make_shared<RankAddr>();
                    ASSERT_EQ(generated_addr.addr_type, "EID");
                    ASSERT_TRUE(EidConverter::Convert(RankAddrType::EID, generated_addr.addr, addr->eidData));
                    addr->levelInfo = level;
                    addr->ports = generated_addr.ports;
                    addr->planeId = generated_addr.plane_id;
                    roots[peer].eidIndexToRankAddr[eid_index++] = addr;
                }
            }
            for (const auto& [index, addr] : roots[peer].eidIndexToRankAddr) {
                const auto& level = *addr->levelInfo;
                SyncEndpoint endpoint;
                endpoint.netLayer = level.netLayer;
                endpoint.netInstanceId = level.netInstanceId;
                endpoint.netType = level.netType;
                endpoint.fabric = TopoFabric::Clos;
                endpoint.planeId = addr->planeId;
                endpoint.ports = addr->ports;
                endpoint.eidIndex = index;
                endpoints[peer].push_back(endpoint);
                topo.closTopoEdges.push_back({level.netLayer, 0, peer, addr->ports});
            }
        }
        std::vector<EidRoute> forward;
        for (uint32_t peer = 0; peer < 2; ++peer) {
            TopoQuerier querier(roots[peer], topo, peer, local_ids, endpoints, {{0, 1}, {1, 1}}, true);
            std::vector<EidRoute> routes;
            ASSERT_EQ(querier.GetEidRoutes(1 - peer, 2, routes), ACLSHMEM_SUCCESS);
            ASSERT_EQ(routes.size(), 2U);
            for (const auto& route : routes) {
                EXPECT_EQ(route.netType, NetType::Clos);
                EXPECT_EQ(route.netLayer, 1U);
            }
            // Both raw layers remain available in the effective CLOS pool.
            EXPECT_EQ(roots[peer].eidIndexToRankAddr.at(routes[0].localEidIndex)->levelInfo->netLayer, 0U);
            EXPECT_EQ(roots[peer].eidIndexToRankAddr.at(routes[1].localEidIndex)->levelInfo->netLayer, 1U);
            if (peer == 0) {
                forward = routes;
            } else {
                for (size_t qp = 0; qp < routes.size(); ++qp) {
                    EXPECT_EQ(routes[qp].localEidIndex, forward[qp].remoteEidIndex);
                    EXPECT_EQ(routes[qp].remoteEidIndex, forward[qp].localEidIndex);
                }
            }
            TopoQuerier mesh_policy(roots[peer], topo, peer, local_ids, endpoints, {{0, 0}}, true);
            EXPECT_EQ(mesh_policy.GetEidRoutes(1 - peer, 1, routes), ACLSHMEM_INVALID_VALUE);
        }
    }
}

TEST_F(ServerRootinfoTest, RejectsHardwareQueryFailure)
{
    EXPECT_CALL(MockHal::instance(), get_ue_list(_, _)).WillOnce(Return(-1));
    aclshmemi_server_product_t strategy;
    EXPECT_FALSE(strategy.get_root_info(0, 0x2b));
}

TEST_F(ServerRootinfoTest, SkipsUnconfiguredUboeAddress)
{
    EXPECT_CALL(MockHal::instance(), get_spod_info(_, _)).Times(0);
    EXPECT_CALL(MockHal::instance(), get_ue_list(_, _)).WillOnce(Invoke([](int, UEList* list) {
        *list = {};
        list->ueNum = 2;
        add_eid(list->ueList[0], 1, 3, 1);
        add_eid(list->ueList[1], 0, 254, 8, 0xc0);
        return 0;
    }));
    aclshmemi_server_product_t strategy;
    auto root = strategy.get_root_info(0, 0x2b);
    ASSERT_TRUE(root);
    ASSERT_EQ(root->ranks.size(), 1U);
    ASSERT_EQ(root->ranks[0].level_list.size(), 1U);
    EXPECT_EQ(root->ranks[0].level_list[0].level, 0);
}

TEST_F(ServerRootinfoTest, FlexRtpUsesEidAndFixedScaleOutPort)
{
    EXPECT_CALL(MockHal::instance(), get_spod_info(_, _)).WillOnce(Invoke([](int, dcmi_spod_info* info) {
        *info = {};
        return 0;
    }));
    EXPECT_CALL(MockHal::instance(), get_ue_list(_, _)).WillOnce(Invoke([](int, UEList* list) {
        *list = {};
        list->ueNum = 1;
        add_eid(list->ueList[0], 0, 0, 8, 0x80);
        return 0;
    }));
    aclshmemi_server_product_t strategy;
    auto root = strategy.get_root_info(0, 0x2f);
    ASSERT_TRUE(root);
    ASSERT_EQ(root->ranks.size(), 1U);
    const auto& layers = root->ranks[0].level_list;
    ASSERT_EQ(layers.size(), 1U);
    EXPECT_EQ(layers[0].level, 2);
    ASSERT_EQ(layers[0].addr_list.size(), 1U);
    EXPECT_EQ(layers[0].addr_list[0].addr_type, "EID");
    EXPECT_THAT(layers[0].addr_list[0].ports, testing::ElementsAre("0/8"));
    EXPECT_EQ(layers[0].addr_list[0].plane_id, "plane_ub_rtp");
}

TEST_F(ServerRootinfoTest, RejectsPc16LocalIdOverflow)
{
    prepare(UINT32_MAX, ACLSHMEMI_TOPO_TYPE_SERVER_16FM);
    aclshmemi_server_product_t strategy;
    EXPECT_FALSE(strategy.get_root_info(0, 0x27));
}

TEST_F(ServerRootinfoTest, MeshSelectsLargestFeOnRequestedDie)
{
    UEList entities{};
    entities.ueNum = 6;
    // Include a smaller FE before and after the winner to detect first/last-match selection.
    add_eid(entities.ueList[0], 1, 2, 2);
    add_eid(entities.ueList[1], 1, 5, 5);
    add_eid(entities.ueList[1], 1, 5, 63);
    add_eid(entities.ueList[2], 1, 3, 3);
    add_eid(entities.ueList[3], 0, 15, 6);       // Wrong die, larger FE.
    add_eid(entities.ueList[4], 1, 14, 8, 0xc0); // Scale-out, larger FE.
    // ueList[5] is empty.
    prepare_entities(entities, false);
    aclshmemi_server_product_t strategy;
    auto root = strategy.get_root_info(0, ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP);
    ASSERT_TRUE(root);
    ASSERT_EQ(root->ranks.size(), 1U);
    ASSERT_EQ(root->ranks[0].level_list.size(), 1U);
    const auto& layer = root->ranks[0].level_list[0];
    EXPECT_EQ(layer.level, 0);
    EXPECT_EQ(layer.instance_id, "sp_42_srv_1");
    EXPECT_EQ(layer.net_type, "TOPO_FILE_DESC");
    ASSERT_EQ(layer.addr_list.size(), 1U);
    EXPECT_EQ(layer.addr_list[0].addr, "00000000004505000000000000000000");
    EXPECT_EQ(layer.addr_list[0].addr_type, "EID");
    EXPECT_EQ(layer.addr_list[0].plane_id, "plane_1");
    EXPECT_THAT(layer.addr_list[0].ports, testing::ElementsAre("1/5"));
}

TEST_F(ServerRootinfoTest, ClosSelectsExactEntityAndGroupsPhysicalPorts)
{
    UEList entities{};
    entities.ueNum = 3;
    add_eid(entities.ueList[0], 0, 2, 9); // Correct FE, wrong die.
    add_eid(entities.ueList[0], 0, 2, 63);
    add_eid(entities.ueList[1], 1, 3, 10); // Correct die, wrong FE.
    add_eid(entities.ueList[1], 1, 3, 63);
    add_eid(entities.ueList[2], 1, 2, 4);
    add_eid(entities.ueList[2], 1, 2, 63); // Group is not the first EID.
    add_eid(entities.ueList[2], 1, 2, 5);
    prepare_entities(entities);
    aclshmemi_server_product_t strategy;
    auto root = strategy.get_root_info(0, ACLSHMEMI_MAIN_BOARD_ID_SERVER_550EL_100);
    ASSERT_TRUE(root);
    ASSERT_EQ(root->ranks.size(), 1U);
    const auto& layers = root->ranks[0].level_list;
    ASSERT_EQ(layers.size(), 2U);
    for (size_t i = 0; i < layers.size(); ++i) {
        EXPECT_EQ(layers[i].level, i);
        EXPECT_EQ(layers[i].net_type, "CLOS");
        EXPECT_EQ(layers[i].instance_id, i == 0 ? "sp_42_srv_1" : "sp_42");
        ASSERT_EQ(layers[i].addr_list.size(), 1U);
        const auto& addr = layers[i].addr_list[0];
        EXPECT_EQ(addr.addr_type, "EID");
        EXPECT_EQ(addr.addr, "00000000007f02000000000000000000");
        EXPECT_EQ(addr.plane_id, "plane_clos_1");
        EXPECT_THAT(addr.ports, testing::ElementsAre("1/4", "1/5"));
    }
}

TEST_F(ServerRootinfoTest, ClosWithoutPortGroupDoesNotCreateLayer)
{
    UEList entities{};
    entities.ueNum = 1;
    add_eid(entities.ueList[0], 1, 2, 4);
    add_eid(entities.ueList[0], 1, 2, 5);
    prepare_entities(entities);
    aclshmemi_server_product_t strategy;
    auto root = strategy.get_root_info(0, ACLSHMEMI_MAIN_BOARD_ID_SERVER_550EL_100);
    ASSERT_TRUE(root);
    ASSERT_EQ(root->ranks.size(), 1U);
    EXPECT_TRUE(root->ranks[0].level_list.empty());
}

TEST_F(ServerRootinfoTest, UboeSelectsLastUsableAddressAndSkipsTrailing254)
{
    UEList entities{};
    entities.ueNum = 1;
    auto& ue = entities.ueList[0];
    add_eid(ue, 0, 10, 8, 0xc0);
    ue.eidList[0].eid.raw[14] = 1;
    add_eid(ue, 0, 10, 8, 0xc0);
    ue.eidList[1].eid.raw[14] = 2;
    add_eid(ue, 0, 254, 8, 0xc0);
    prepare_entities(entities, false);
    aclshmemi_server_product_t strategy;
    auto root = strategy.get_root_info(0, ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE);
    ASSERT_TRUE(root);
    ASSERT_EQ(root->ranks.size(), 1U);
    ASSERT_EQ(root->ranks[0].level_list.size(), 1U);
    const auto& layer = root->ranks[0].level_list[0];
    EXPECT_EQ(layer.level, 2);
    EXPECT_EQ(layer.instance_id, "cluster");
    EXPECT_EQ(layer.net_type, "CLOS");
    ASSERT_EQ(layer.addr_list.size(), 1U);
    EXPECT_EQ(layer.addr_list[0].addr_type, "IPV4");
    EXPECT_EQ(layer.addr_list[0].addr, "10.0.0.2");
    EXPECT_EQ(layer.addr_list[0].plane_id, "plane_uboe");
    EXPECT_THAT(layer.addr_list[0].ports, testing::ElementsAre("1/8"));
}

TEST_F(ServerRootinfoTest, UboeWithOnly254AddressesDoesNotCreateLayer)
{
    UEList entities{};
    entities.ueNum = 1;
    add_eid(entities.ueList[0], 0, 254, 8, 0xc0);
    add_eid(entities.ueList[0], 0, 254, 9, 0xc0);
    prepare_entities(entities, false);
    aclshmemi_server_product_t strategy;
    auto root = strategy.get_root_info(0, ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE);
    ASSERT_TRUE(root);
    ASSERT_EQ(root->ranks.size(), 1U);
    EXPECT_TRUE(root->ranks[0].level_list.empty());
}

TEST_F(ServerRootinfoTest, FlexSelectsLastRtpEidAlongsideMeshAndClos)
{
    for (uint32_t board : {ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX, ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX_RTP}) {
        SCOPED_TRACE(board);
        UEList entities{};
        entities.ueNum = 3;
        add_eid(entities.ueList[0], 0, 3, 1);
        add_eid(entities.ueList[1], 1, 2, 5);
        add_eid(entities.ueList[1], 1, 2, 63);
        add_eid(entities.ueList[2], 0, 1, 8, 0x80);
        add_eid(entities.ueList[2], 0, 2, 8, 0x80);
        prepare_entities(entities);
        auto strategy = aclshmemi_product_strategy_t::create(board);
        ASSERT_NE(strategy, nullptr);
        auto root = strategy->get_root_info(9, board);
        ASSERT_TRUE(root);
        ASSERT_EQ(root->ranks.size(), 1U);
        const auto& layers = root->ranks[0].level_list;
        ASSERT_EQ(layers.size(), 3U);
        EXPECT_EQ(layers[0].instance_id, "sp_42_srv_1_board1");
        EXPECT_EQ(layers[1].instance_id, "sp_42");
        EXPECT_EQ(layers[2].instance_id, "cluster");
        const char* expected_eids[] = {
            "00000000000103000000000000000000", "00000000007f02000000000000000000", "00000000000802800000000000000000"};
        const char* expected_planes[] = {"plane_0", "plane_clos_1", "plane_ub_rtp"};
        const char* expected_ports[] = {"0/1", "1/5", "0/8"};
        for (size_t i = 0; i < layers.size(); ++i) {
            EXPECT_EQ(layers[i].level, i);
            EXPECT_EQ(layers[i].net_type, i == 0 ? "TOPO_FILE_DESC" : "CLOS");
            ASSERT_EQ(layers[i].addr_list.size(), 1U);
            EXPECT_EQ(layers[i].addr_list[0].addr_type, "EID");
            EXPECT_EQ(layers[i].addr_list[0].addr, expected_eids[i]);
            EXPECT_EQ(layers[i].addr_list[0].plane_id, expected_planes[i]);
            EXPECT_THAT(layers[i].addr_list[0].ports, testing::ElementsAre(expected_ports[i]));
        }
    }
}

TEST_F(ServerRootinfoTest, UbxKeepsMeshClosPortsAndGroupInLayerZero)
{
    UEList entities{};
    entities.ueNum = 2;
    add_eid(entities.ueList[0], 1, 3, 1);
    add_eid(entities.ueList[0], 1, 3, 63);
    add_eid(entities.ueList[1], 1, 2, 4);
    add_eid(entities.ueList[1], 1, 2, 63);
    add_eid(entities.ueList[1], 1, 2, 5);
    prepare_entities(entities, false);
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX);
    ASSERT_NE(strategy, nullptr);
    auto root = strategy->get_root_info(3, ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX);
    ASSERT_TRUE(root);
    EXPECT_EQ(root->topo_file_path, "/driver-root/driver/topo/950/atlas_850_3.json");
    ASSERT_EQ(root->ranks.size(), 1U);
    EXPECT_EQ(root->ranks[0].local_id, 3);
    ASSERT_EQ(root->ranks[0].level_list.size(), 1U);
    const auto& layer = root->ranks[0].level_list[0];
    EXPECT_EQ(layer.level, 0);
    EXPECT_EQ(layer.net_type, "TOPO_FILE_DESC");
    EXPECT_EQ(layer.instance_id, "sp_42_srv_1");
    ASSERT_EQ(layer.addr_list.size(), 4U);
    const char* expected_eids[] = {
        "00000000004103000000000000000000", "00000000004402000000000000000000", "00000000004502000000000000000000",
        "00000000007f02000000000000000000"};
    const char* expected_planes[] = {"plane_1", "plane_clos_1_4", "plane_clos_1_5", "plane_clos_1"};
    for (size_t i = 0; i < layer.addr_list.size(); ++i) {
        EXPECT_EQ(layer.addr_list[i].addr_type, "EID");
        EXPECT_EQ(layer.addr_list[i].addr, expected_eids[i]);
        EXPECT_EQ(layer.addr_list[i].plane_id, expected_planes[i]);
    }
    EXPECT_THAT(layer.addr_list[0].ports, testing::ElementsAre("1/1"));
    EXPECT_THAT(layer.addr_list[1].ports, testing::ElementsAre("1/4"));
    EXPECT_THAT(layer.addr_list[2].ports, testing::ElementsAre("1/5"));
    EXPECT_THAT(layer.addr_list[3].ports, testing::ElementsAre("1/4", "1/5"));
}

TEST_F(ServerRootinfoTest, UbxRejectsRootInfoWithoutUsableAddresses)
{
    prepare_entities(UEList{}, false);
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX);
    ASSERT_NE(strategy, nullptr);
    EXPECT_FALSE(strategy->get_root_info(0, ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX));
}

TEST_F(ServerRootinfoTest, Type1PreservesEmptyMeshAndClosLayers)
{
    prepare_entities(UEList{});
    aclshmemi_server_product_t strategy;
    auto root = strategy.get_root_info(0, ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1);
    ASSERT_TRUE(root);
    ASSERT_EQ(root->ranks.size(), 1U);
    const auto& layers = root->ranks[0].level_list;
    ASSERT_EQ(layers.size(), 2U);
    EXPECT_EQ(layers[0].level, 0);
    EXPECT_EQ(layers[0].net_type, "MESH");
    EXPECT_EQ(layers[0].instance_id, "sp_42_srv_1");
    EXPECT_TRUE(layers[0].addr_list.empty());
    EXPECT_EQ(layers[1].level, 1);
    EXPECT_EQ(layers[1].net_type, "CLOS");
    EXPECT_EQ(layers[1].instance_id, "sp_42");
    EXPECT_TRUE(layers[1].addr_list.empty());
}

TEST_F(ServerRootinfoTest, OrdinaryNonCardGenerationDoesNotDiscoverRoce)
{
    for (uint32_t board : {0x25U, 0x27U, 0x07U, 0x44U, 0x2dU}) {
        SCOPED_TRACE(board);
        if (board == ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX) {
            UEList entities{};
            entities.ueNum = 1;
            add_eid(entities.ueList[0], 1, 3, 1);
            prepare_entities(entities, false);
        } else {
            prepare(0, ACLSHMEMI_TOPO_TYPE_SERVER_8P);
        }
        EXPECT_CALL(MockHal::instance(), get_visible_phy_ids()).Times(0);
        EXPECT_CALL(MockHal::instance(), get_npu_count()).Times(0);
        auto strategy = aclshmemi_product_strategy_t::create(board);
        ASSERT_NE(strategy, nullptr);
        auto root = strategy->get_root_info(0, board);
        ASSERT_TRUE(root);
        ASSERT_EQ(root->ranks.size(), 1U);
        for (const auto& layer : root->ranks[0].level_list) {
            EXPECT_NE(layer.level, 3);
        }
    }
}

TEST_F(ServerRootinfoTest, ScaleOutClassificationAndCna)
{
    dcmi_urma_eid_t eid{};
    eid.raw[7] = 0xc0;
    eid.raw[6] = 192;
    eid.raw[12] = 168;
    eid.raw[13] = 1;
    eid.raw[14] = 2;
    EXPECT_TRUE(aclshmemi_eid_parser_t::is_uboe(eid));
    EXPECT_FALSE(aclshmemi_eid_parser_t::is_ub_rtp(eid));
    EXPECT_EQ(aclshmemi_eid_parser_t::get_cna(eid), "192.168.1.2");
    eid.raw[7] = 0x80;
    EXPECT_TRUE(aclshmemi_eid_parser_t::is_ub_rtp(eid));
    EXPECT_TRUE(aclshmemi_eid_parser_t::is_scale_out(eid));
    eid.raw[7] = 0;
    EXPECT_FALSE(aclshmemi_eid_parser_t::is_scale_out(eid));
}

} // namespace topo
} // namespace shm
