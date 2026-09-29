"""Compile the production ownership check without CANN or NPU dependencies.

Run directly with python3. Only logging and device-buffer cleanup are stubbed;
the method body is extracted from device_udma_transport_manager.cpp.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SharedJettyOwnershipTest(unittest.TestCase):
    def test_production_check_in_direct_and_relay_builds(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++17 compiler required")
        root = Path(__file__).resolve().parents[5]
        source = (root / "src/host/transport/device_udma/device_udma_transport_manager.cpp").read_text()
        start = source.index("Result UdmaTransportManager::CheckPerQpEndpointSplit(")
        end = source.index("\nvoid UdmaTransportManager::PrintSharedJettyGrouping(", start)
        method = source[start:end]
        prefix = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
using Result = int;
constexpr int ACLSHMEM_SUCCESS = 0, ACLSHMEM_INNER_ERROR = -1;
constexpr bool ACLSHMEM_UDMA_RELAY_ENABLED = TEST_RELAY;
std::ostringstream errors, infos;
#define SHM_LOG_ERROR(message) (errors << message << '\n')
#define SHM_LOG_INFO(message) (infos << message << '\n')
class UdmaTransportManager {
public:
 bool shared_jetty_enabled_ = true;
 uint32_t qp_num_ = 16;
 uint32_t frees = 0;
 void FreeQueueStateBuffers() { ++frees; }
 Result CheckPerQpEndpointSplit(const std::map<std::pair<uint32_t,uint32_t>,size_t>&);
};
'''
        main = r'''
int main() {
 UdmaTransportManager manager;
 using Bindings = std::map<std::pair<uint32_t,uint32_t>,size_t>;
 // Exact reported shape, including a third peer reusing the same state.
 assert(manager.CheckPerQpEndpointSplit({{{8,0},112},{{9,0},112},{{10,0},112}})==0);
 assert(manager.frees==0 && errors.str().empty());
 Bindings bindings;
 for(uint32_t qp=0;qp<16;++qp) {
  bindings[{1,qp}]=qp;
  for(uint32_t peer=8;peer<128;++peer) bindings[{peer,qp}]=112+qp;
 }
 assert(manager.CheckPerQpEndpointSplit(bindings)==0);
 assert(manager.frees==0 && errors.str().empty());
 assert(manager.CheckPerQpEndpointSplit({{{9,0},112},{{9,1},112}})==(TEST_RELAY?0:-1));
 assert(manager.CheckPerQpEndpointSplit({{{8,0},112},{{9,0},112},{{10,2},112}})==(TEST_RELAY?0:-1));
 assert(manager.frees==(TEST_RELAY?0:2));
 if(!TEST_RELAY) {
  assert(errors.str().find("first=(dst_pe=8, qp=0), conflicting=(dst_pe=10, qp=2)")!=std::string::npos);
  assert(infos.str().find("same-QP cross-peer sharing allowed")!=std::string::npos);
 }
 manager.shared_jetty_enabled_=false;
 assert(manager.CheckPerQpEndpointSplit({{{8,0},112},{{9,1},112}})==0);
 manager.shared_jetty_enabled_=true; manager.qp_num_=1;
 assert(manager.CheckPerQpEndpointSplit({{{8,0},112},{{9,0},112}})==0);
 manager.qp_num_=16;
 assert(manager.CheckPerQpEndpointSplit({})==0);
 assert(manager.frees==(TEST_RELAY?0:2));
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp, binary = Path(directory) / "ownership.cpp", Path(directory) / "ownership"
            cpp.write_text(prefix + method + main)
            for relay in (0, 1):
                with self.subTest(relay=relay):
                    build = subprocess.run(
                        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                         f"-DTEST_RELAY={relay}", str(cpp), "-o", str(binary)],
                        capture_output=True, text=True)
                    self.assertEqual(build.returncode, 0, build.stderr)
                    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=20)
                    self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
