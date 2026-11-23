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
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "acc_common_util.h"
#include "acc_tcp_ssl_helper.h"
#include "openssl_api_dl.h"

using namespace shm::acc;

namespace {
/*
 * These tests exercise LoadCaCert without a real libssl. OPENSSLAPIDL exposes every OpenSSL
 * entry point as a public static function pointer filled in by dlsym, so the tests point those
 * pointers at counting stubs instead. Handles are opaque (only ever passed back to other stubs),
 * so arbitrary non-null addresses stand in for X509 / EVP_PKEY / SSL_CTX.
 */
int g_x509Alloc = 0;
int g_x509Free = 0;
int g_pkeyBits = 4096; // above MIN_PRIVATE_KEY_CONTENT_BIT_LEN
int g_loadVerifyRet = 1;
bool g_pemReturnsNull = false;

X509* FakeCert()
{
    static int cert = 0;
    return reinterpret_cast<X509*>(&cert);
}

EVP_PKEY* FakePkey()
{
    static int pkey = 0;
    return reinterpret_cast<EVP_PKEY*>(&pkey);
}

SSL_CTX* FakeCtx()
{
    static int ctx = 0;
    return reinterpret_cast<SSL_CTX*>(&ctx);
}

/* Distinct handles so the X509CmpCurrentTime stub can tell the two ends of the validity apart. */
ASN1_TIME* FakeNotAfter()
{
    static int t = 0;
    return reinterpret_cast<ASN1_TIME*>(&t);
}

ASN1_TIME* FakeNotBefore()
{
    static int t = 0;
    return reinterpret_cast<ASN1_TIME*>(&t);
}

X509* StubPemReadX509(FILE*, X509**, pem_password_cb*, void*)
{
    if (g_pemReturnsNull) {
        return nullptr;
    }
    ++g_x509Alloc;
    return FakeCert();
}

X509* StubX509Free(X509* cert)
{
    if (cert != nullptr) {
        ++g_x509Free;
    }
    return nullptr;
}

void StubSslCtxSetVerify(SSL_CTX*, int, int (*)(int, X509_STORE_CTX*)) {}

int StubLoadVerifyLocations(SSL_CTX*, const char*, const char*) { return g_loadVerifyRet; }

/*
 * X509_cmp_current_time returns <0 when the time is in the past and >0 when it is in the future.
 * A currently-valid cert therefore has notAfter in the future and notBefore in the past, which is
 * what CertVerify checks for.
 */
int StubX509CmpCurrentTime(const ASN1_TIME* t) { return (t == FakeNotBefore()) ? -1 : 1; }

ASN1_TIME* StubX509GetNotAfter(const X509*) { return FakeNotAfter(); }

ASN1_TIME* StubX509GetNotBefore(const X509*) { return FakeNotBefore(); }

EVP_PKEY* StubX509GetPubkey(X509*) { return FakePkey(); }

int StubEvpPkeyBits(const EVP_PKEY*) { return g_pkeyBits; }

void StubEvpPkeyFree(EVP_PKEY*) {}
} // namespace

class AccTcpSslHelperTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_x509Alloc = 0;
        g_x509Free = 0;
        g_pkeyBits = 4096;
        g_loadVerifyRet = 1;
        g_pemReturnsNull = false;

        SaveOpensslPtrs();
        InstallStubs();

        /* LoadCaFileList runs FileUtil::Realpath, so the CA files must really exist on disk */
        caDir_ = "acc_ssl_ut_" + std::to_string(getpid());
        tmpTop_ = "/tmp";
        ASSERT_EQ(mkdir((tmpTop_ + "/" + caDir_).c_str(), 0700), 0);

        helper_ = AccMakeRef<AccTcpSslHelper>();
        ASSERT_NE(helper_, nullptr);
    }

    void TearDown() override
    {
        helper_ = nullptr;
        for (const auto& name : createdFiles_) {
            (void)remove((tmpTop_ + "/" + caDir_ + "/" + name).c_str());
        }
        (void)rmdir((tmpTop_ + "/" + caDir_).c_str());
        RestoreOpensslPtrs();
    }

    void MakeCaFiles(const std::set<std::string>& names)
    {
        for (const auto& name : names) {
            std::ofstream out(tmpTop_ + "/" + caDir_ + "/" + name);
            ASSERT_TRUE(out.is_open());
            out << "placeholder, never parsed because PemReadX509 is stubbed\n";
            createdFiles_.insert(name);
        }
        SetCaPaths(tmpTop_, caDir_, names);
    }

    /*
     * Friendship is not inherited and TEST_F derives from this fixture, so every touch of a
     * private member has to go through a fixture method like these two.
     */
    AccResult CallLoadCaCert() { return helper_->LoadCaCert(FakeCtx()); }

    void SetCaPaths(const std::string& top, const std::string& caDir, const std::set<std::string>& names)
    {
        helper_->tlsTopPath = top;
        helper_->tlsCaPath = caDir;
        helper_->tlsCaFile = names;
    }

    void SaveOpensslPtrs()
    {
        savedPemReadX509_ = OPENSSLAPIDL::pemReadX509;
        savedX509Free_ = OPENSSLAPIDL::x509Free;
        savedSetVerify_ = OPENSSLAPIDL::sslCtxSetVerify;
        savedLoadVerify_ = OPENSSLAPIDL::loadVerifyLocations;
        savedCmpTime_ = OPENSSLAPIDL::x509CmpCurrentTime;
        savedNotAfter_ = OPENSSLAPIDL::x509GetNotAfter;
        savedNotBefore_ = OPENSSLAPIDL::x509GetNotBefore;
        savedGetPubkey_ = OPENSSLAPIDL::x509GetPubkey;
        savedPkeyBits_ = OPENSSLAPIDL::evpPkeyBits;
        savedPkeyFree_ = OPENSSLAPIDL::evpPkeyFree;
    }

    void InstallStubs()
    {
        OPENSSLAPIDL::pemReadX509 = StubPemReadX509;
        OPENSSLAPIDL::x509Free = StubX509Free;
        OPENSSLAPIDL::sslCtxSetVerify = StubSslCtxSetVerify;
        OPENSSLAPIDL::loadVerifyLocations = StubLoadVerifyLocations;
        OPENSSLAPIDL::x509CmpCurrentTime = StubX509CmpCurrentTime;
        OPENSSLAPIDL::x509GetNotAfter = StubX509GetNotAfter;
        OPENSSLAPIDL::x509GetNotBefore = StubX509GetNotBefore;
        OPENSSLAPIDL::x509GetPubkey = StubX509GetPubkey;
        OPENSSLAPIDL::evpPkeyBits = StubEvpPkeyBits;
        OPENSSLAPIDL::evpPkeyFree = StubEvpPkeyFree;
    }

    void RestoreOpensslPtrs()
    {
        OPENSSLAPIDL::pemReadX509 = savedPemReadX509_;
        OPENSSLAPIDL::x509Free = savedX509Free_;
        OPENSSLAPIDL::sslCtxSetVerify = savedSetVerify_;
        OPENSSLAPIDL::loadVerifyLocations = savedLoadVerify_;
        OPENSSLAPIDL::x509CmpCurrentTime = savedCmpTime_;
        OPENSSLAPIDL::x509GetNotAfter = savedNotAfter_;
        OPENSSLAPIDL::x509GetNotBefore = savedNotBefore_;
        OPENSSLAPIDL::x509GetPubkey = savedGetPubkey_;
        OPENSSLAPIDL::evpPkeyBits = savedPkeyBits_;
        OPENSSLAPIDL::evpPkeyFree = savedPkeyFree_;
    }

    AccTcpSslHelperPtr helper_ = nullptr;
    std::string tmpTop_;
    std::string caDir_;
    std::set<std::string> createdFiles_;

    FuncPemReadX509 savedPemReadX509_ = nullptr;
    FuncX509Free savedX509Free_ = nullptr;
    FuncSslCtxSetVerify savedSetVerify_ = nullptr;
    FuncLoadVerifyLocations savedLoadVerify_ = nullptr;
    FuncX509CmpCurrentTime savedCmpTime_ = nullptr;
    FuncX509GetNotAfter savedNotAfter_ = nullptr;
    FuncX509GetNotBefore savedNotBefore_ = nullptr;
    FuncX509GetPubkey savedGetPubkey_ = nullptr;
    FuncEvpPkeyBits savedPkeyBits_ = nullptr;
    FuncEvpPkeyFree savedPkeyFree_ = nullptr;
};

/* Success path: every CA parsed is released. */
TEST_F(AccTcpSslHelperTest, LoadCaCertFreesAllCertsOnSuccess)
{
    MakeCaFiles({"ca1.pem", "ca2.pem", "ca3.pem"});

    EXPECT_EQ(CallLoadCaCert(), ACC_OK);
    EXPECT_EQ(g_x509Alloc, 3);
    EXPECT_EQ(g_x509Free, g_x509Alloc);
}

/* Early return from a failed CertVerify must still release the cert already parsed. */
TEST_F(AccTcpSslHelperTest, LoadCaCertFreesCertWhenVerifyFails)
{
    g_pkeyBits = 1024; // below the 3072-bit floor, so CertVerify rejects it
    MakeCaFiles({"ca1.pem"});

    EXPECT_NE(CallLoadCaCert(), ACC_OK);
    EXPECT_EQ(g_x509Alloc, 1);
    EXPECT_EQ(g_x509Free, g_x509Alloc);
}

/* Early return from a failed SslCtxLoadVerifyLocations must also release it. */
TEST_F(AccTcpSslHelperTest, LoadCaCertFreesCertWhenLoadVerifyLocationsFails)
{
    g_loadVerifyRet = 0;
    MakeCaFiles({"ca1.pem"});

    EXPECT_NE(CallLoadCaCert(), ACC_OK);
    EXPECT_EQ(g_x509Alloc, 1);
    EXPECT_EQ(g_x509Free, g_x509Alloc);
}

/* A null cert from PemReadX509 must not be handed to X509Free. */
TEST_F(AccTcpSslHelperTest, LoadCaCertHandlesNullCertWithoutFreeing)
{
    g_pemReturnsNull = true;
    MakeCaFiles({"ca1.pem"});

    EXPECT_NE(CallLoadCaCert(), ACC_OK);
    EXPECT_EQ(g_x509Alloc, 0);
    EXPECT_EQ(g_x509Free, 0);
}

/* A missing CA file fails in LoadCaFileList, before anything is parsed. */
TEST_F(AccTcpSslHelperTest, LoadCaCertFailsOnMissingCaFile)
{
    SetCaPaths(tmpTop_, caDir_, {"does_not_exist.pem"});

    EXPECT_NE(CallLoadCaCert(), ACC_OK);
    EXPECT_EQ(g_x509Alloc, 0);
    EXPECT_EQ(g_x509Free, 0);
}

TEST(AccCommonUtilAddressTest, HexadecimalHostnamesAreNotIPv6)
{
    for (const std::string& name : {"a", "face", "dead", "cafe", "FACE", "beef", "abcd", "0"}) {
        SCOPED_TRACE(name);
        EXPECT_FALSE(AccCommonUtil::IsValidIPv6(name));
    }
}

TEST(AccCommonUtilAddressTest, AcceptsCompleteIPv6Literals)
{
    for (const std::string& address :
         {"::", "::1", "2001::", "2001:db8::1", "2001:db8:0:1:2:3:4:5", "FFFF:FFFF:FFFF:FFFF:FFFF:FFFF:FFFF:FFFF",
          "::ffff:127.0.0.1"}) {
        SCOPED_TRACE(address);
        EXPECT_TRUE(AccCommonUtil::IsValidIPv6(address));
    }
}

TEST(AccCommonUtilAddressTest, RejectsIncompleteOrMalformedIPv6Literals)
{
    for (const std::string& address :
         {"", "1:2", "1:2:3:4:5:6:7", "1:2:3:4:5:6:7:8:9", "1:::2", "1::2::3", ":1", "1:", "2001:db8::xyz",
          "ordinary-shmem-host", "127.0.0.1", "[::1]", " ::1", "::1 ", "fe80::1%eth0"}) {
        SCOPED_TRACE(address);
        EXPECT_FALSE(AccCommonUtil::IsValidIPv6(address));
    }
    EXPECT_FALSE(AccCommonUtil::IsValidIPv6(std::string("::1\0suffix", 10)));
}

TEST(AccCommonUtilAddressTest, KeepsIPv4Classification)
{
    EXPECT_TRUE(AccCommonUtil::IsValidIPv4("127.0.0.1"));
    EXPECT_TRUE(AccCommonUtil::IsValidIPv4("192.0.2.1"));
    EXPECT_FALSE(AccCommonUtil::IsValidIPv4("::1"));
    EXPECT_FALSE(AccCommonUtil::IsValidIPv4("face"));
    EXPECT_FALSE(AccCommonUtil::IsValidIPv4("256.0.0.1"));
}
