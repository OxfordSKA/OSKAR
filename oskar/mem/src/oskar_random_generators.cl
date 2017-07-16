inline uint mulhilo_(uint a, uint b, uint* hip) {
    ulong product = ((ulong)a)*((ulong)b);
    *hip = product >> 32;
    return (uint)product;
}
inline uint2 r4inc_(uint2 key) {
    key.x += ((uint)0x9E3779B9); key.y += ((uint)0xBB67AE85); return key;
}
inline uint4 r4iter_(uint4 ctr, uint2 key) {
    uint4 out;
    uint hi0, hi1;
    uint lo0 = mulhilo_(((uint)0xD2511F53), ctr.x, &hi0);
    uint lo1 = mulhilo_(((uint)0xCD9E8D57), ctr.z, &hi1);
    out.x = hi1 ^ ctr.y ^ key.x;
    out.y = lo1;
    out.z = hi0 ^ ctr.w ^ key.y;
    out.w = lo0;
    return out;
}
inline uint4 rnd4_(uint4 ctr, uint2 key) {
    ctr = r4iter_(ctr, key);
    key = r4inc_(key); ctr = r4iter_(ctr, key);
    key = r4inc_(key); ctr = r4iter_(ctr, key);
    key = r4inc_(key); ctr = r4iter_(ctr, key);
    key = r4inc_(key); ctr = r4iter_(ctr, key);
    key = r4inc_(key); ctr = r4iter_(ctr, key);
    key = r4inc_(key); ctr = r4iter_(ctr, key);
    key = r4inc_(key); ctr = r4iter_(ctr, key);
    key = r4inc_(key); ctr = r4iter_(ctr, key);
    key = r4inc_(key); ctr = r4iter_(ctr, key);
    return ctr;
}
uint4 rnd_uint4(const uint seed,
        const uint c0, const uint c1, const uint c2, const uint c3) {
    return rnd4_((uint4)(c0, c1, c2, c3), (uint2)(seed, 0xCAFEF00D));
}
