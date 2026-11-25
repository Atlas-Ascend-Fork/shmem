# SIMT set/get Extended Tests

All lanes of each VF consistently call set, and then perform A→B→A→B using the existing typed/byte and GM/UB APIs.
Check the source, receive buffer, untouched region, and guard; the two instances use opposite logical rank mappings; check that the getter addresses are non-zero and different, and that the instance ranks match.
Cover same-VF switching, 4 AIVs, byte lengths 0/1/3/31/32/33/127/129 and offsets 1/3,
as well as int8/int16/int32/int64/uint32/uint64/half/float/bfloat16.
Each type covers the thread/warp/block and nbi variants. The host selection does not affect the SIMT selection.
