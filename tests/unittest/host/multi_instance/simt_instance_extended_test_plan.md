# SIMT set/get 扩展测试

每个 VF 的所有 lanes 一致调用 set，再通过原有 typed/byte、GM/UB 接口执行 A→B→A→B。
检查源、接收区、未触及区及 guard；两个实例使用相反的逻辑 rank 映射，检查 getter 地址间距及实例查询。
覆盖同 VF 切换、4 AIV、字节长度 0/1/3/31/32/33/127/129 与偏移 1/3，
以及 int8/int16/int32/int64/uint32/uint64/half/float/bfloat16。
每类均覆盖 thread/warp/block 和 nbi 变体。Host 选择不影响 SIMT 选择。
