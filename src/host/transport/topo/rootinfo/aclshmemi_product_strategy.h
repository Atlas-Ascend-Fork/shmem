/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEMI_PRODUCT_STRATEGY_HPP
#define ACLSHMEMI_PRODUCT_STRATEGY_HPP

#include <memory>
#include <cstdint>
#include <string>
#include <optional>
#include <vector>
#include "aclshmemi_types.h"
#include "aclshmemi_hal.h"
#include "aclshmemi_eid_parser.h"

namespace shm {
namespace topo {

constexpr size_t ACLSHMEMI_ROOT_INFO_SIZE = 4096;

constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH = 0x68;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH = 0x6a;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH = 0x6c;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH = 0x25;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_UBOE = 0x27;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP = 0x29;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE = 0x2b;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_SERVER_550EL_100 = 0x46;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_SERVER_550EL_200 = 0x48;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1 = 0x23;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX = 0x44;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_POD = 0x07;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_POD_2D = 0x03;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX = 0x2d;
constexpr uint32_t ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX_RTP = 0x2f;

// Product topology types, matching HCOMM topo_addr_info/src/hal.h.
constexpr uint32_t ACLSHMEMI_TOPO_TYPE_IGNORE = 99;
constexpr uint32_t ACLSHMEMI_TOPO_TYPE_SERVER_8P = 0;
constexpr uint32_t ACLSHMEMI_TOPO_TYPE_SERVER_16FM = 3;

std::optional<std::string> aclshmemi_topo_file_path(
    const std::string& driver_path, uint32_t mainboard_id, uint32_t spod_type);

struct aclshmemi_pod_ub_entity_rule_t {
    uint32_t mainboard_id;
    int level;
    int die_id;
    int ue_id;
    std::vector<int> ports;
    std::vector<int> npus;
    std::string plane_id;
};

class aclshmemi_product_strategy_t {
public:
    virtual ~aclshmemi_product_strategy_t() = default;

    virtual std::optional<aclshmemi_root_info_t> get_root_info(int phy_id, uint32_t mainboard_id) = 0;
    virtual std::optional<aclshmemi_root_info_t> get_root_info_without_roce(int phy_id, uint32_t mainboard_id);
    virtual size_t get_root_info_size() const = 0;

    static std::unique_ptr<aclshmemi_product_strategy_t> create(uint32_t mainboard_id);
};

class aclshmemi_card_product_t : public aclshmemi_product_strategy_t {
public:
    std::optional<aclshmemi_root_info_t> get_root_info(int phy_id, uint32_t mainboard_id) override;
    std::optional<aclshmemi_root_info_t> get_root_info_without_roce(int phy_id, uint32_t mainboard_id) override;
    size_t get_root_info_size() const override { return ACLSHMEMI_ROOT_INFO_SIZE; }

private:
    std::optional<aclshmemi_root_info_t> build_root_info(int phy_id, uint32_t mainboard_id, bool include_roce);
    std::optional<aclshmemi_net_layer_t> process_mesh_layer(
        int phy_id, const dcmi_urma_eid_info_t* eid_list, size_t eid_cnt);
    std::optional<aclshmemi_net_layer_t> process_mesh2p_layer(
        int phy_id, const dcmi_urma_eid_info_t* eid_list, size_t eid_cnt);
    std::optional<aclshmemi_net_layer_t> process_roce_layer(int phy_id);
};

class aclshmemi_server_product_t : public aclshmemi_product_strategy_t {
public:
    std::optional<aclshmemi_root_info_t> get_root_info(int phy_id, uint32_t mainboard_id) override;
    size_t get_root_info_size() const override { return ACLSHMEMI_ROOT_INFO_SIZE; }
};

class aclshmemi_pod_flex_product_t : public aclshmemi_product_strategy_t {
public:
    std::optional<aclshmemi_root_info_t> get_root_info(int phy_id, uint32_t mainboard_id) override;
    size_t get_root_info_size() const override { return ACLSHMEMI_ROOT_INFO_SIZE; }
};

class aclshmemi_pod_product_t : public aclshmemi_product_strategy_t {
public:
    std::optional<aclshmemi_root_info_t> get_root_info(int phy_id, uint32_t mainboard_id) override;
    size_t get_root_info_size() const override { return ACLSHMEMI_ROOT_INFO_SIZE; }

private:
    std::optional<aclshmemi_net_layer_t> process_mesh_layer(
        int phy_id, const UEList& ue_list, const dcmi_spod_info& spod_info);
    std::optional<aclshmemi_net_layer_t> process_clos_layer(
        int phy_id, uint32_t mainboard_id, const UEList& ue_list, const dcmi_spod_info& spod_info);
};

class aclshmemi_ubx_product_t : public aclshmemi_product_strategy_t {
public:
    std::optional<aclshmemi_root_info_t> get_root_info(int phy_id, uint32_t mainboard_id) override;
    size_t get_root_info_size() const override { return ACLSHMEMI_ROOT_INFO_SIZE; }
};

} // namespace topo
} // namespace shm

#endif // ACLSHMEMI_PRODUCT_STRATEGY_HPP
