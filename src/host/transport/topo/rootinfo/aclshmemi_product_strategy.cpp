/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclshmemi_product_strategy.h"

#include <algorithm>
#include <arpa/inet.h>
#include <climits>
#include <utility>

#include <securec.h>

#include "aclshmemi_eid_parser.h"
#include "aclshmemi_hal.h"
#include "aclshmemi_npu_nic_affinity.h"
#include "aclshmemi_server_rules.h"
#include "host_rdma.h"
#include "utils/shmemi_logger.h"

namespace shm {
namespace topo {

static constexpr const char* TOPO_FILE_DIR_PATH = "driver/topo/950";
static constexpr const char* ROCE_PLANE_ID = "roce";

static std::string build_topo_file_path(const std::string& driver_path, const std::string& topo_filename)
{
    return driver_path + "/" + TOPO_FILE_DIR_PATH + "/" + topo_filename;
}

template <typename... Args>
static bool format_rootinfo_string(
    const char* context, char* buffer, size_t buffer_size, const char* format, Args... args)
{
    int ret = snprintf_s(buffer, buffer_size, buffer_size - 1, format, args...);
    if (ret < 0) {
        SHM_LOG_ERROR(context << " failed: snprintf_s ret=" << ret);
        return false;
    }
    return true;
}

std::optional<std::string> aclshmemi_topo_file_path(
    const std::string& driver_path, uint32_t mainboard_id, uint32_t spod_type)
{
    const char* filename = nullptr;
    switch (mainboard_id) {
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH:
            filename = "atlas_350_1.json";
            break;
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH:
            filename = "atlas_350_2.json";
            break;
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH:
            filename = "atlas_350_3.json";
            break;
        case ACLSHMEMI_MAIN_BOARD_ID_POD:
        case ACLSHMEMI_MAIN_BOARD_ID_POD_2D:
            filename = "atlas_950_1.json";
            break;
        case ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX:
        case ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX_RTP:
            filename = "atlas_950_2.json";
            break;
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX:
            filename = "atlas_850_3.json";
            break;
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_550EL_100:
            filename = "atlas_550EL_100.json";
            break;
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_550EL_200:
            filename = "atlas_550EL_200.json";
            break;
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_UBOE:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE:
            filename = spod_type == ACLSHMEMI_TOPO_TYPE_SERVER_16FM ? "atlas_850_2.json" : "atlas_850_1.json";
            break;
        default:
            return std::nullopt;
    }
    return build_topo_file_path(driver_path, filename);
}

static const aclshmemi_pod_ub_entity_rule_t g_pod_ub_rules[] = {
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 1, 2, {0, 1, 2, 3, 5, 6}, {0, 1, 2, 3}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 0, 2, {1, 2}, {0, 1, 2, 3}, "plane_pg_1"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 0, 2, {0, 1, 2, 3, 4, 5}, {4, 5, 6, 7}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 1, 2, {1, 2}, {4, 5, 6, 7}, "plane_pg_1"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 1, 2, {0, 1, 2, 3, 5, 6}, {0, 1, 2, 3}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 0, 2, {1, 2}, {0, 1, 2, 3}, "plane_pg_1"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 0, 2, {0, 1, 2, 3, 4, 5}, {4, 5, 6, 7}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 1, 2, {1, 2}, {4, 5, 6, 7}, "plane_pg_1"},
};

std::optional<aclshmemi_root_info_t> aclshmemi_product_strategy_t::get_root_info_without_roce(
    int phy_id, uint32_t mainboard_id)
{
    return get_root_info(phy_id, mainboard_id);
}

std::optional<aclshmemi_root_info_t> aclshmemi_product_strategy_t::get_root_info_for_family(
    int phy_id, uint32_t mainboard_id, int family)
{
    (void)family;
    return get_root_info(phy_id, mainboard_id);
}

std::unique_ptr<aclshmemi_product_strategy_t> aclshmemi_product_strategy_t::create(uint32_t mainboard_id)
{
    switch (mainboard_id) {
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH:
            return std::make_unique<aclshmemi_card_product_t>();
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_UBOE:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_550EL_100:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_550EL_200:
            return std::make_unique<aclshmemi_server_product_t>();
        case ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX:
        case ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX_RTP:
            return std::make_unique<aclshmemi_pod_flex_product_t>();
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX:
            return std::make_unique<aclshmemi_ubx_product_t>();
        case ACLSHMEMI_MAIN_BOARD_ID_POD:
        case ACLSHMEMI_MAIN_BOARD_ID_POD_2D:
            return std::make_unique<aclshmemi_pod_product_t>();
        default:
            return nullptr;
    }
}
std::optional<aclshmemi_root_info_t> aclshmemi_card_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    return build_root_info(phy_id, mainboard_id, true, AF_UNSPEC);
}

std::optional<aclshmemi_root_info_t> aclshmemi_card_product_t::get_root_info_for_family(
    int phy_id, uint32_t mainboard_id, int family)
{
    return build_root_info(phy_id, mainboard_id, true, family);
}

std::optional<aclshmemi_root_info_t> aclshmemi_card_product_t::get_root_info_without_roce(
    int phy_id, uint32_t mainboard_id)
{
    return build_root_info(phy_id, mainboard_id, false);
}

std::optional<aclshmemi_root_info_t> aclshmemi_card_product_t::build_root_info(
    int phy_id, uint32_t mainboard_id, bool include_roce, int family)
{
    auto& hal = aclshmemi_hal_t::instance();

    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    std::string driver_path = hal.get_driver_install_path();
    auto topo_path = aclshmemi_topo_file_path(driver_path, mainboard_id, ACLSHMEMI_TOPO_TYPE_IGNORE);
    if (!topo_path) {
        return std::nullopt;
    }
    aclshmemi_root_info_set_topo_file_path(root_info, *topo_path);

    UEList ue_list;
    memset_s(&ue_list, sizeof(ue_list), 0, sizeof(ue_list));
    int ret = hal.get_ue_list(phy_id, &ue_list);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_card_product_t::get_root_info failed: get_ue_list failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, phy_id, phy_id);

    dcmi_urma_eid_info_t eid_list[MAX_EID_NUM];
    memset_s(eid_list, sizeof(eid_list), 0, sizeof(eid_list));
    size_t eid_cnt = MAX_EID_NUM;
    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        for (uint32_t j = 0; j < ue_list.ueList[i].eidNum && eid_cnt > 0; ++j) {
            if (eid_cnt > 0) {
                eid_list[MAX_EID_NUM - eid_cnt] = ue_list.ueList[i].eidList[j];
                eid_cnt--;
            }
        }
    }
    eid_cnt = MAX_EID_NUM - eid_cnt;

    if (mainboard_id == ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH) {
        auto mesh_layer = process_mesh2p_layer(phy_id, eid_list, eid_cnt);
        if (!mesh_layer && eid_cnt > 0) {
            return std::nullopt;
        }
        if (mesh_layer) {
            aclshmemi_rank_add_net_layer(rank, *mesh_layer);
        }
    } else if (mainboard_id == ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH) {
        auto mesh_layer = process_mesh_layer(phy_id, eid_list, eid_cnt);
        if (!mesh_layer && eid_cnt > 0) {
            return std::nullopt;
        }
        if (mesh_layer) {
            aclshmemi_rank_add_net_layer(rank, *mesh_layer);
        }
    }
    if (include_roce) {
        auto roce_layer = process_roce_layer(phy_id, family);
        if (roce_layer) {
            aclshmemi_rank_add_net_layer(rank, *roce_layer);
        }
    }

    aclshmemi_root_info_add_rank(root_info, rank);
    return root_info;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_card_product_t::process_mesh_layer(
    int phy_id, const dcmi_urma_eid_info_t* eid_list, size_t eid_cnt)
{
    if (eid_cnt == 0) {
        return std::nullopt;
    }

    auto& hal = aclshmemi_hal_t::instance();
    auto server_id = hal.get_server_id(phy_id);
    if (!server_id) {
        return std::nullopt;
    }

    char instance_id[64] = {0};
    snprintf_s(instance_id, sizeof(instance_id), sizeof(instance_id) - 1, "%s", server_id->c_str());

    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "TOPO_FILE_DESC");

    for (size_t i = 0; i < eid_cnt; ++i) {
        int port_id = aclshmemi_eid_parser_t::get_port_id(eid_list[i].eid);
        if (port_id > 9) {
            continue;
        }
        int die_id = aclshmemi_eid_parser_t::get_die_id(eid_list[i].eid);

        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, eid_list[i].eid);

        char port[16] = {0};
        char plane_id[16] = {0};
        snprintf_s(port, sizeof(port), sizeof(port) - 1, "%d/%d", die_id, port_id);
        snprintf_s(plane_id, sizeof(plane_id), sizeof(plane_id) - 1, "plane_%d", die_id);
        aclshmemi_addr_add_port(addr, port);
        aclshmemi_addr_set_plane_id(addr, plane_id);

        aclshmemi_net_layer_add_addr(layer, addr);
    }

    return layer;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_card_product_t::process_mesh2p_layer(
    int phy_id, const dcmi_urma_eid_info_t* eid_list, size_t eid_cnt)
{
    if (eid_cnt == 0) {
        return std::nullopt;
    }

    auto& hal = aclshmemi_hal_t::instance();
    auto server_id = hal.get_server_id(phy_id);
    if (!server_id) {
        return std::nullopt;
    }

    char instance_id[64] = {0};
    snprintf_s(instance_id, sizeof(instance_id), sizeof(instance_id) - 1, "%s_%d", server_id->c_str(), phy_id / 2);

    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "TOPO_FILE_DESC");

    for (size_t i = 0; i < eid_cnt; ++i) {
        int port_id = aclshmemi_eid_parser_t::get_port_id(eid_list[i].eid);
        if (port_id <= 9) {
            continue;
        }

        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, eid_list[i].eid);

        const int ports[] = {4, 5, 6, 8};
        for (int j = 0; j < 4; ++j) {
            char port[16] = {0};
            snprintf_s(port, sizeof(port), sizeof(port) - 1, "0/%d", ports[j]);
            aclshmemi_addr_add_port(addr, port);
        }
        aclshmemi_addr_set_plane_id(addr, "plane_0");
        aclshmemi_net_layer_add_addr(layer, addr);
    }

    return layer;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_card_product_t::process_roce_layer(int phy_id, int family)
{
    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 3, "cluster");
    aclshmemi_net_layer_set_net_type(layer, "CLOS");

    aclshmemi_addr_t addr;
    std::string ip_addr;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    char fallback_ip[INET6_ADDRSTRLEN] = {0};
    if (get_npu_host_rdma_ip(phy_id, family, fallback_ip, sizeof(fallback_ip)) != 0) {
        return std::nullopt;
    }
    ip_addr = fallback_ip;
#else
    auto xml_ip_addr = aclshmemi_get_roce_ip_from_xml(phy_id, family);
    if (xml_ip_addr) {
        ip_addr = *xml_ip_addr;
    } else {
        // Preserve the existing PCIe/NIC discovery path if virtual topology is unavailable.
        char fallback_ip[INET6_ADDRSTRLEN] = {0};
        if (get_npu_host_rdma_ip(phy_id, family, fallback_ip, sizeof(fallback_ip)) != 0) {
            return std::nullopt;
        }
        ip_addr = fallback_ip;
    }
#endif
    aclshmemi_addr_set_ip(addr, ip_addr);
    aclshmemi_addr_set_plane_id(addr, ROCE_PLANE_ID);
    aclshmemi_addr_add_port(addr, "d2h");

    aclshmemi_net_layer_add_addr(layer, addr);
    return layer;
}

namespace {

const UBEntity* find_server_entity(const UEList& list, const aclshmemi_server_entity_t& rule)
{
    const UBEntity* best = nullptr;
    int max_fe = -1;
    for (uint32_t i = 0; i < std::min<uint32_t>(list.ueNum, MAX_UE_PER_NPU); ++i) {
        const auto& ue = list.ueList[i];
        if (ue.eidNum == 0) {
            continue;
        }
        // Match HCOMM GetUBEntityByFilter: assume the first EID represents the entity's protocol and die.
        const auto& eid = ue.eidList[0].eid;
        if (rule.kind == aclshmemi_entity_kind_t::UBOE) {
            if (aclshmemi_eid_parser_t::is_uboe(eid)) {
                return &ue;
            }
            continue;
        }
        if (rule.kind == aclshmemi_entity_kind_t::UB_RTP) {
            if (aclshmemi_eid_parser_t::is_ub_rtp(eid)) {
                return &ue;
            }
            continue;
        }
        if (aclshmemi_eid_parser_t::is_scale_out(eid) || aclshmemi_eid_parser_t::get_ub_die_id(eid) != rule.die) {
            continue;
        }
        const int fe = aclshmemi_eid_parser_t::get_ub_entity_id(ue);
        if (fe == rule.fe) {
            return &ue;
        }
        if (rule.fe == -1 && fe > max_fe) {
            max_fe = fe;
            best = &ue;
        }
    }
    return best;
}

std::optional<std::string> server_instance_id(aclshmemi_instance_scope_t scope, int phy_id, const dcmi_spod_info& spod)
{
    using Scope = aclshmemi_instance_scope_t;
    if (scope == Scope::OS) {
        return aclshmemi_hal_t::instance().get_server_id(phy_id);
    }
    if (scope == Scope::CLUSTER) {
        return "cluster";
    }
    std::string id = "sp_" + std::to_string(spod.super_pod_id);
    if (scope == Scope::SERVER || scope == Scope::BOARD) {
        id += "_srv_" + std::to_string(spod.server_index);
    }
    if (scope == Scope::BOARD) {
        id += "_board" + std::to_string(phy_id / 8);
    }
    return id;
}

void add_server_entity(aclshmemi_net_layer_t& layer, const UBEntity& ue, const aclshmemi_server_entity_t& rule)
{
    using Kind = aclshmemi_entity_kind_t;
    const uint32_t count = std::min<uint32_t>(ue.eidNum, MAX_EID_PER_UE);
    if (rule.kind == Kind::UBOE || rule.kind == Kind::UB_RTP) {
        // HCOMM selects the last configured scale-out address on this entity.
        std::optional<aclshmemi_addr_t> selected;
        for (uint32_t i = 0; i < count; ++i) {
            const auto& eid = ue.eidList[i].eid;
            aclshmemi_addr_t addr;
            if (rule.kind == Kind::UBOE) {
                if (eid.raw[6] == 254) {
                    continue;
                }
                aclshmemi_addr_set_ip(addr, aclshmemi_eid_parser_t::get_cna(eid));
                aclshmemi_addr_set_plane_id(addr, "plane_uboe");
            } else {
                aclshmemi_addr_set_eid(addr, eid);
                aclshmemi_addr_set_plane_id(addr, "plane_ub_rtp");
            }
            aclshmemi_addr_add_port(addr, rule.port);
            selected = std::move(addr);
        }
        if (selected) {
            aclshmemi_net_layer_add_addr(layer, *selected);
        }
        return;
    }
    if (rule.kind == Kind::CLOS) {
        const int group = aclshmemi_eid_parser_t::get_server_port_group_idx(ue);
        if (group < 0) {
            return;
        }
        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, ue.eidList[group].eid);
        const int die = aclshmemi_eid_parser_t::get_ub_die_id(ue.eidList[group].eid);
        for (uint32_t i = 0; i < count; ++i) {
            if (!aclshmemi_eid_parser_t::is_ub_port_group(ue.eidList[i].eid)) {
                aclshmemi_addr_add_port(
                    addr, std::to_string(die) + "/" +
                              std::to_string(aclshmemi_eid_parser_t::get_ub_port_id(ue.eidList[i].eid)));
            }
        }
        aclshmemi_addr_set_plane_id(addr, "plane_clos_" + std::to_string(die));
        aclshmemi_net_layer_add_addr(layer, addr);
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const auto& eid = ue.eidList[i].eid;
        if (aclshmemi_eid_parser_t::is_ub_port_group(eid)) {
            continue;
        }
        const std::string die = std::to_string(aclshmemi_eid_parser_t::get_ub_die_id(eid));
        const std::string port = std::to_string(aclshmemi_eid_parser_t::get_ub_port_id(eid));
        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, eid);
        aclshmemi_addr_add_port(addr, die + "/" + port);
        aclshmemi_addr_set_plane_id(addr, rule.kind == Kind::MESH ? "plane_" + die : "plane_clos_" + die + "_" + port);
        aclshmemi_net_layer_add_addr(layer, addr);
    }
}

} // namespace

std::optional<aclshmemi_root_info_t> aclshmemi_server_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    auto& hal = aclshmemi_hal_t::instance();
    UEList ue_list{};
    dcmi_spod_info spod{};
    if (hal.get_ue_list(phy_id, &ue_list) != 0) {
        return std::nullopt;
    }
    if (mainboard_id != ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX &&
        mainboard_id != ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP &&
        mainboard_id != ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE && hal.get_spod_info(phy_id, &spod) != 0) {
        return std::nullopt;
    }
    const auto rules = aclshmemi_server_layers(mainboard_id, spod.super_pod_type);
    if (rules.empty()) {
        return std::nullopt;
    }
    aclshmemi_root_info_t root;
    aclshmemi_root_info_init(root);
    auto topo_path = aclshmemi_topo_file_path(hal.get_driver_install_path(), mainboard_id, spod.super_pod_type);
    if (!topo_path) {
        return std::nullopt;
    }
    aclshmemi_root_info_set_topo_file_path(root, *topo_path);
    aclshmemi_rank_t rank;
    const uint64_t local_id =
        static_cast<uint64_t>(phy_id) +
        (spod.super_pod_type == ACLSHMEMI_TOPO_TYPE_SERVER_16FM ? static_cast<uint64_t>(spod.server_index) * 8 : 0);
    if (local_id > static_cast<uint64_t>(INT32_MAX)) {
        SHM_LOG_ERROR("Rootinfo local_id overflow for phy_id=" << phy_id);
        return std::nullopt;
    }
    aclshmemi_rank_init(rank, phy_id, static_cast<int>(local_id));
    for (const auto& rule : rules) {
        const auto instance_id = server_instance_id(rule.scope, phy_id, spod);
        if (!instance_id) {
            return std::nullopt;
        }
        aclshmemi_net_layer_t layer;
        aclshmemi_net_layer_init(layer, rule.level, *instance_id);
        aclshmemi_net_layer_set_net_type(layer, rule.net_type);
        for (const auto& entity_rule : rule.entities) {
            const auto* ue = find_server_entity(ue_list, entity_rule);
            if (ue != nullptr) {
                add_server_entity(layer, *ue, entity_rule);
            }
        }
        if (!layer.addr_list.empty() || rule.keep_empty) {
            aclshmemi_rank_add_net_layer(rank, layer);
        }
    }
    if (rank.level_list.empty()) {
        SHM_LOG_WARN(
            "Generated rootinfo has no network layers, mainboard_id=0x" << std::hex << mainboard_id << std::dec
                                                                        << ", phy_id=" << phy_id);
    }
    aclshmemi_root_info_add_rank(root, rank);
    return root;
}

std::optional<aclshmemi_root_info_t> aclshmemi_pod_flex_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    if (mainboard_id != ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX && mainboard_id != ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX_RTP) {
        return std::nullopt;
    }
    // Flex has its own product identity but shares the table-driven generation engine.
    aclshmemi_server_product_t generator;
    return generator.get_root_info(phy_id, mainboard_id);
}

std::optional<aclshmemi_net_layer_t> aclshmemi_pod_product_t::process_mesh_layer(
    int phy_id, const UEList& ue_list, const dcmi_spod_info& spod_info)
{
    aclshmemi_net_layer_t layer;
    char instance_id[64] = {0};
    if (!format_rootinfo_string(
            "aclshmemi_pod_product_t::process_mesh_layer instance_id", instance_id, sizeof(instance_id), "sp%u_srv%u",
            spod_info.super_pod_id, spod_info.server_index)) {
        return std::nullopt;
    }
    aclshmemi_net_layer_init(layer, 0, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "MESH");

    constexpr int NPU_NUM = 8;
    constexpr int MAX_MESH_PORT_ID = 9;
    const int mesh_die_id = (phy_id % NPU_NUM) < 4 ? 0 : 1;
    int mesh_entity_id = aclshmemi_eid_parser_t::get_max_entity_id(ue_list, mesh_die_id);

    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        int fe = aclshmemi_eid_parser_t::get_ub_entity_id(ue_list.ueList[i]);
        if (ue_list.ueList[i].eidNum == 0) {
            continue;
        }

        int die = aclshmemi_eid_parser_t::get_ub_die_id(ue_list.ueList[i].eidList[0].eid);
        if (fe != mesh_entity_id || die != mesh_die_id) {
            continue;
        }

        for (uint32_t j = 0; j < ue_list.ueList[i].eidNum; ++j) {
            int phy_port_id = aclshmemi_eid_parser_t::get_ub_port_id(ue_list.ueList[i].eidList[j].eid);
            if (phy_port_id > MAX_MESH_PORT_ID) {
                continue;
            }

            aclshmemi_addr_t addr;
            aclshmemi_addr_set_eid(addr, ue_list.ueList[i].eidList[j].eid);

            char port[16] = {0};
            if (!format_rootinfo_string(
                    "aclshmemi_pod_product_t::process_mesh_layer port", port, sizeof(port), "%d/%d", die,
                    phy_port_id)) {
                return std::nullopt;
            }
            aclshmemi_addr_add_port(addr, port);
            aclshmemi_addr_set_plane_id(addr, "plane_0");

            aclshmemi_net_layer_add_addr(layer, addr);
        }
    }

    return layer;
}

static const aclshmemi_pod_ub_entity_rule_t* get_pod_ub_rule(
    int npu_id, int level, uint32_t mainboard_id, int die, int fe_id)
{
    constexpr size_t RULE_COUNT = sizeof(g_pod_ub_rules) / sizeof(g_pod_ub_rules[0]);
    constexpr int NPU_NUM = 8;

    for (size_t i = 0; i < RULE_COUNT; ++i) {
        if (g_pod_ub_rules[i].level != level) {
            continue;
        }

        int npu_in_rule = 0;
        for (size_t j = 0; j < g_pod_ub_rules[i].npus.size(); ++j) {
            if (g_pod_ub_rules[i].npus[j] == (npu_id % NPU_NUM)) {
                npu_in_rule = 1;
                break;
            }
        }
        if (npu_in_rule == 0) {
            continue;
        }

        if (mainboard_id == g_pod_ub_rules[i].mainboard_id && die == g_pod_ub_rules[i].die_id &&
            fe_id == g_pod_ub_rules[i].ue_id) {
            return &g_pod_ub_rules[i];
        }
    }
    return nullptr;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_pod_product_t::process_clos_layer(
    int phy_id, uint32_t mainboard_id, const UEList& ue_list, const dcmi_spod_info& spod_info)
{
    aclshmemi_net_layer_t layer;
    char instance_id[64] = {0};
    if (!format_rootinfo_string(
            "aclshmemi_pod_product_t::process_clos_layer instance_id", instance_id, sizeof(instance_id), "superpod_%u",
            spod_info.super_pod_id)) {
        return std::nullopt;
    }
    aclshmemi_net_layer_init(layer, 1, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "CLOS");

    std::optional<aclshmemi_addr_t> primary_addr;
    std::optional<aclshmemi_addr_t> secondary_addr;
    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        int fe = aclshmemi_eid_parser_t::get_ub_entity_id(ue_list.ueList[i]);
        int port_group_idx = aclshmemi_eid_parser_t::get_server_port_group_idx(ue_list.ueList[i]);
        if (port_group_idx < 0) {
            continue;
        }

        int die = aclshmemi_eid_parser_t::get_pod_die_id(ue_list.ueList[i].eidList[port_group_idx].eid);

        const aclshmemi_pod_ub_entity_rule_t* rule = get_pod_ub_rule(phy_id, 1, mainboard_id, die, fe);
        if (rule == nullptr) {
            continue;
        }

        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, ue_list.ueList[i].eidList[port_group_idx].eid);

        int port_num = 0;
        for (uint32_t j = 0; j < ue_list.ueList[i].eidNum; ++j) {
            if (aclshmemi_eid_parser_t::is_ub_port_group(ue_list.ueList[i].eidList[j].eid)) {
                continue;
            }
            int port_id = aclshmemi_eid_parser_t::get_ub_port_id(ue_list.ueList[i].eidList[j].eid);
            char port[16] = {0};
            if (!format_rootinfo_string(
                    "aclshmemi_pod_product_t::process_clos_layer port", port, sizeof(port), "%d/%d", die, port_id)) {
                return std::nullopt;
            }
            aclshmemi_addr_add_port(addr, port);
            port_num++;
        }

        constexpr int PRIMARY_PORT_NUM = 6;
        if (port_num >= PRIMARY_PORT_NUM) {
            aclshmemi_addr_set_plane_id(addr, "plane_0");
            primary_addr = addr;
        } else {
            aclshmemi_addr_set_plane_id(addr, "plane_1");
            secondary_addr = addr;
        }
    }
    if (primary_addr) {
        aclshmemi_net_layer_add_addr(layer, *primary_addr);
    }
    if (secondary_addr) {
        aclshmemi_net_layer_add_addr(layer, *secondary_addr);
    }

    return layer;
}
std::optional<aclshmemi_root_info_t> aclshmemi_pod_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    auto& hal = aclshmemi_hal_t::instance();

    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    std::string driver_path = hal.get_driver_install_path();
    auto topo_path = aclshmemi_topo_file_path(driver_path, mainboard_id, ACLSHMEMI_TOPO_TYPE_IGNORE);
    if (!topo_path) {
        return std::nullopt;
    }
    aclshmemi_root_info_set_topo_file_path(root_info, *topo_path);

    UEList ue_list;
    memset_s(&ue_list, sizeof(ue_list), 0, sizeof(ue_list));
    int ret = hal.get_ue_list(phy_id, &ue_list);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_pod_product_t::get_root_info failed: get_ue_list failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    dcmi_spod_info spod_info;
    memset_s(&spod_info, sizeof(spod_info), 0, sizeof(spod_info));
    ret = hal.get_spod_info(phy_id, &spod_info);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_pod_product_t::get_root_info failed: get_spod_info failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    constexpr int NPU_NUM = 8;
    int local_id = (phy_id % NPU_NUM) + ((spod_info.server_index % NPU_NUM) * NPU_NUM);

    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, phy_id, local_id);

    auto mesh_layer = process_mesh_layer(phy_id, ue_list, spod_info);
    if (!mesh_layer) {
        SHM_LOG_ERROR("aclshmemi_pod_product_t::get_root_info failed: process_mesh_layer failed");
        return std::nullopt;
    }
    aclshmemi_rank_add_net_layer(rank, *mesh_layer);

    auto clos_layer = process_clos_layer(phy_id, mainboard_id, ue_list, spod_info);
    if (!clos_layer) {
        SHM_LOG_ERROR("aclshmemi_pod_product_t::get_root_info failed: process_clos_layer failed");
        return std::nullopt;
    }
    aclshmemi_rank_add_net_layer(rank, *clos_layer);

    aclshmemi_root_info_add_rank(root_info, rank);
    return root_info;
}

std::optional<aclshmemi_root_info_t> aclshmemi_ubx_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    if (mainboard_id != ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX) {
        return std::nullopt;
    }
    aclshmemi_server_product_t server;
    auto root = server.get_root_info(phy_id, mainboard_id);
    if (!root || root->ranks.empty() || root->ranks.front().level_list.empty()) {
        return std::nullopt;
    }
    return root;
}

} // namespace topo
} // namespace shm
