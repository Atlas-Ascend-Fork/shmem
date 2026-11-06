/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclshmemi_topo_addr_info_manager.h"

namespace shm {
namespace topo {

aclshmemi_topo_addr_info_manager_t& aclshmemi_topo_addr_info_manager_t::instance()
{
    static aclshmemi_topo_addr_info_manager_t inst;
    return inst;
}

std::optional<size_t> aclshmemi_topo_addr_info_manager_t::get_size(int phy_id)
{
    auto mainboard_id = hal_.get_mainboard_id(phy_id);
    if (!mainboard_id) {
        return std::nullopt;
    }

    auto strategy = aclshmemi_product_strategy_t::create(*mainboard_id);
    if (!strategy) {
        return std::nullopt;
    }

    return strategy->get_root_info_size();
}

std::optional<std::string> aclshmemi_topo_addr_info_manager_t::get_topo_file_path(int phy_id)
{
    auto mainboard_id = hal_.get_mainboard_id(phy_id);
    if (!mainboard_id) {
        return std::nullopt;
    }

    uint32_t spod_type = ACLSHMEMI_TOPO_TYPE_IGNORE;
    switch (*mainboard_id) {
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_UBOE:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE: {
            dcmi_spod_info spod{};
            if (hal_.get_spod_info(phy_id, &spod) == 0) {
                spod_type = spod.super_pod_type;
            }
            break;
        }
        default:
            break;
    }
    return aclshmemi_topo_file_path(hal_.get_driver_install_path(), *mainboard_id, spod_type);
}

std::optional<std::string> aclshmemi_topo_addr_info_manager_t::get_root_info_json(int phy_id)
{
    auto mainboard_id = hal_.get_mainboard_id(phy_id);
    if (!mainboard_id) {
        return std::nullopt;
    }

    auto strategy = aclshmemi_product_strategy_t::create(*mainboard_id);
    if (!strategy) {
        return std::nullopt;
    }

    auto root_info = strategy->get_root_info(phy_id, *mainboard_id);
    if (!root_info) {
        return std::nullopt;
    }

    return aclshmemi_root_info_to_string(*root_info);
}

} // namespace topo
} // namespace shm
