/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <string>

#include "utils/under_api/dl_hccp_api.h"

namespace {
/* 写入临时文件, content 为文件全部内容 */
bool WriteTempFile(const std::string& path, const std::string& content)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return false;
    }
    file.write(content.data(), static_cast<std::streamsize>(content.size()));
    file.close();
    return file.good();
}

/* 生成 n 字节的非标记填充内容 */
std::string MakePadding(size_t n)
{
    std::string padding(n, '\0');
    for (size_t i = 0; i < n; ++i) {
        padding[i] = static_cast<char>('a' + (i % ('z' - 'a' + 1)));
    }
    return padding;
}

/* 自定义 cookie 流: 依次吐出 data 内容, 数据耗尽后按 failAtEnd 返回持续读错误(EIO)或正常 EOF。
 * 用于构造普通文件无法模拟的"中流 I/O 错误"场景 */
struct CookieCtx {
    const char* data = nullptr;
    size_t size = 0;
    size_t pos = 0;
    bool failAtEnd = false;
};

ssize_t CookieRead(void* cookie, char* buf, size_t size)
{
    auto* ctx = static_cast<CookieCtx*>(cookie);
    if (ctx->pos >= ctx->size) {
        if (ctx->failAtEnd) {
            errno = EIO;
            return -1; /* 读错误: ferror 置位且未到 EOF */
        }
        return 0; /* 正常 EOF */
    }
    const size_t n = std::min(size, ctx->size - ctx->pos);
    std::memcpy(buf, ctx->data + ctx->pos, n);
    ctx->pos += n;
    return static_cast<ssize_t>(n);
}

int CookieClose(void*) { return 0; }

std::FILE* OpenCookieStream(CookieCtx& ctx)
{
    cookie_io_functions_t funcs = {};
    funcs.read = CookieRead;
    funcs.close = CookieClose;
    return fopencookie(&ctx, "rb", funcs);
}
} // namespace

/* 分支一: 镜像无法定位/读取时返回 0(调用方降级为 V1) */
TEST(DlHccpApiStrideTest, UnknownImageReturnsZero)
{
    EXPECT_EQ(shm::DlHccpApi::DetectStrideFromImage(nullptr), 0);
    EXPECT_EQ(shm::DlHccpApi::DetectStrideFromImage("/nonexistent/path/libra.so"), 0);
}

/* 空文件无法判定布局, 返回 0 */
TEST(DlHccpApiStrideTest, EmptyImageReturnsZero)
{
    const std::string path = "/tmp/libra_stride_test_empty.bin";
    ASSERT_TRUE(WriteTempFile(path, ""));
    EXPECT_EQ(shm::DlHccpApi::DetectStrideFromImage(path.c_str()), 0);
    std::remove(path.c_str());
}

/* 分支二: 镜像完整扫描无 V2 标记, 返回 V1(224)。
 * 文件大于单次扫描分块(4K), 覆盖多块读取路径 */
TEST(DlHccpApiStrideTest, ImageWithoutMarkerReturnsV1Stride)
{
    const std::string path = "/tmp/libra_stride_test_v1.bin";
    ASSERT_TRUE(WriteTempFile(path, MakePadding(8192)));
    EXPECT_EQ(shm::DlHccpApi::DetectStrideFromImage(path.c_str()), sizeof(shm::HccpSocketConnectInfo));
    std::remove(path.c_str());
}

/* 分支三: 镜像含 V2 标记, 返回 V2(232) */
TEST(DlHccpApiStrideTest, ImageWithMarkerReturnsV2Stride)
{
    const std::string path = "/tmp/libra_stride_test_v2.bin";
    const std::string content = MakePadding(1024) + "tagChkDis" + MakePadding(1024);
    ASSERT_TRUE(WriteTempFile(path, content));
    EXPECT_EQ(shm::DlHccpApi::DetectStrideFromImage(path.c_str()), sizeof(shm::HccpSocketConnectInfoV2));
    std::remove(path.c_str());
}

/* V2 标记串横跨扫描分块边界(1K/4K/32K)时仍应命中, 覆盖跨块重叠读取逻辑 */
TEST(DlHccpApiStrideTest, MarkerSpanningChunkBoundaryDetected)
{
    const std::string path = "/tmp/libra_stride_test_v2_boundary.bin";
    std::string content = MakePadding(65536);
    constexpr const char* marker = "tagChkDis";
    /* 各偏移使标记串恰好横跨 1K/4K/32K 的块边界 */
    for (const size_t offset : {size_t{1020}, size_t{4090}, size_t{32760}}) {
        content.replace(offset, 9, marker);
    }
    ASSERT_TRUE(WriteTempFile(path, content));
    EXPECT_EQ(shm::DlHccpApi::DetectStrideFromImage(path.c_str()), sizeof(shm::HccpSocketConnectInfoV2));
    std::remove(path.c_str());
}

/* 回归: 成功读取一块后持续读错误(ferror 置位且未到 EOF)应退出到探测失败(0), 而不是空转死循环 */
TEST(DlHccpApiStrideTest, ReadErrorAfterFirstChunkReturnsZero)
{
    const std::string content = MakePadding(8192); /* 大于单块 4K: 第一块成功, 之后持续 EIO */
    CookieCtx ctx{content.data(), content.size(), 0, true};
    std::FILE* file = OpenCookieStream(ctx);
    ASSERT_NE(file, nullptr);
    EXPECT_EQ(shm::DlHccpApi::DetectStrideFromFile(file), 0);
    std::fclose(file);
}

/* 正常 EOF(数据读完返回 0)路径行为不变 */
TEST(DlHccpApiStrideTest, CookieStreamCleanEofReturnsV1Stride)
{
    const std::string content = MakePadding(8192);
    CookieCtx ctx{content.data(), content.size(), 0, false};
    std::FILE* file = OpenCookieStream(ctx);
    ASSERT_NE(file, nullptr);
    EXPECT_EQ(shm::DlHccpApi::DetectStrideFromFile(file), sizeof(shm::HccpSocketConnectInfo));
    std::fclose(file);
}
