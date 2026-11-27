/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_SERVER_RULES_H
#define ACLSHMEMI_SERVER_RULES_H

#include <vector>
#include "aclshmemi_product_strategy.h"

namespace shm {
namespace topo {

enum class aclshmemi_instance_scope_t { OS, SERVER, SUPERPOD, BOARD, CLUSTER };
enum class aclshmemi_entity_kind_t { MESH, CLOS_PORTS, CLOS, UBOE, UB_RTP };

struct aclshmemi_server_entity_t {
    int die;
    int fe; // -1 selects the largest non-scale-out FE on this die.
    aclshmemi_entity_kind_t kind;
    const char* port = "";
};

struct aclshmemi_server_layer_t {
    int level;
    const char* net_type;
    aclshmemi_instance_scope_t scope;
    std::vector<aclshmemi_server_entity_t> entities;
    bool keep_empty = false; // Preserve legacy layers even when no address is available.
};

// Product rules ported from HCOMM master 6d9ed308, product_server.c:g_netInfoList.
// These rules cover URMA/UB layers; RoCE retains SHMEM's separate backend flow.
inline std::vector<aclshmemi_server_layer_t> aclshmemi_server_layers(uint32_t board, uint32_t spod_type)
{
    using Scope = aclshmemi_instance_scope_t;
    using Kind = aclshmemi_entity_kind_t;
    switch (board) {
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_UBX:
            return {
                {0, "TOPO_FILE_DESC", Scope::OS, {{1, 3, Kind::MESH}, {1, 2, Kind::CLOS_PORTS}, {1, 2, Kind::CLOS}}}};
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH:
            return {
                {0, "MESH", Scope::SERVER, {{1, -1, Kind::MESH}}, true},
                {1, "CLOS", Scope::SUPERPOD, {{0, 3, Kind::CLOS}}, true}};
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_UBOE:
            if (spod_type == ACLSHMEMI_TOPO_TYPE_SERVER_16FM) {
                return {
                    {0, "TOPO_FILE_DESC", Scope::SUPERPOD, {{0, -1, Kind::MESH}, {1, -1, Kind::MESH}}},
                    {2, "CLOS", Scope::CLUSTER, {{0, 0, Kind::UBOE, "1/8"}}}};
            }
            return {
                {0, "TOPO_FILE_DESC", Scope::SERVER, {{1, -1, Kind::MESH}}},
                {1, "CLOS", Scope::SUPERPOD, {{0, 3, Kind::CLOS}}},
                {2, "CLOS", Scope::CLUSTER, {{0, 0, Kind::UBOE, "1/8"}}}};
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE:
            return {
                {0, "TOPO_FILE_DESC", Scope::OS, {{1, -1, Kind::MESH}}},
                {2, "CLOS", Scope::CLUSTER, {{0, 0, Kind::UBOE, "1/8"}}}};
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH_NOSP:
            return {{0, "TOPO_FILE_DESC", Scope::OS, {{1, -1, Kind::MESH}}}};
        case ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX:
        case ACLSHMEMI_MAIN_BOARD_ID_POD_FLEX_RTP:
            return {
                {0, "TOPO_FILE_DESC", Scope::BOARD, {{0, -1, Kind::MESH}}},
                {1, "CLOS", Scope::SUPERPOD, {{1, 2, Kind::CLOS}}},
                {2, "CLOS", Scope::CLUSTER, {{0, 0, Kind::UB_RTP, "0/8"}}}};
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_550EL_100:
            return {
                {0, "CLOS", Scope::SERVER, {{1, 2, Kind::CLOS}}}, {1, "CLOS", Scope::SUPERPOD, {{1, 2, Kind::CLOS}}}};
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_550EL_200:
            return {
                {0, "CLOS", Scope::SERVER, {{0, 2, Kind::CLOS}, {1, 2, Kind::CLOS}}},
                {1, "CLOS", Scope::SUPERPOD, {{0, 2, Kind::CLOS}, {1, 2, Kind::CLOS}}}};
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1:
            // Upstream leaves netType/instanceIdFunc unset for this legacy entry.
            // Preserve SHMEM's defined Mesh/SuperPod semantics rather than calling a null function.
            return {
                {0, "MESH", Scope::SERVER, {{1, -1, Kind::MESH}}, true},
                {1, "CLOS", Scope::SUPERPOD, {{0, 3, Kind::CLOS}, {1, 2, Kind::CLOS}}, true}};
        default:
            return {};
    }
}

} // namespace topo
} // namespace shm
#endif
