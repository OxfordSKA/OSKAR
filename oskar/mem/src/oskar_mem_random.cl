uint4 rnd_uint4(const uint seed,
        const uint c0, const uint c1, const uint c2, const uint c3);

inline REAL int_to_range_0_to_1_REAL(const uint in)
{
    const REAL factor = (REAL) 1.0 / ((REAL) 1.0 + 0xFFFFFFFFu);
    const REAL half_factor = (REAL) 0.5 * factor;
    return (in * factor) + half_factor;
}

inline REAL int_to_range_minus_1_to_1_REAL(const uint in)
{
    const REAL factor = (REAL) 1.0 / ((REAL) 1.0 + 0x7FFFFFFFu);
    const REAL half_factor = (REAL) 0.5 * factor;
    return (((int)in) * factor) + half_factor;
}

inline REAL2 box_muller_REAL(const uint u0, const uint u1)
{
    REAL2 rnd;
    REAL sin_t, cos_t, r, t;
    t = (REAL) 3.14159265358979323846264338327950288;
    t *= int_to_range_minus_1_to_1_REAL(u0);
    sin_t = sincos(t, &cos_t);
    r = sqrt((REAL) -2.0 * log(int_to_range_0_to_1_REAL(u1)));
    rnd.x = sin_t * r;
    rnd.y = cos_t * r;
    return rnd;
}

kernel void mem_random_gaussian_REAL(const int n,
        global REAL* restrict data, const uint seed,
        const uint counter1, const uint counter2, const uint counter3,
        const REAL std)
{
    REAL4 r;
    const int i = get_global_id(0);
    const int i4 = i * 4;
    if (i4 >= n) return;
    uint4 rnd = rnd_uint4(seed, i, counter1, counter2, counter3);
    r.xy = box_muller_REAL(rnd.x, rnd.y);
    r.zw = box_muller_REAL(rnd.z, rnd.w);
    r *= std;

    /* Store random numbers. */
    if (i4 <= n - 4)
    {
        ((global REAL4*) data)[i] = r;
    }
    else
    {
        /* End case only if length not divisible by 4. */
        data[i4] = r.x;
        if (i4 + 1 < n)
            data[i4 + 1] = r.y;
        if (i4 + 2 < n)
            data[i4 + 2] = r.z;
        if (i4 + 3 < n)
            data[i4 + 3] = r.w;
    }
}

kernel void mem_random_uniform_REAL(const int n,
        global REAL* restrict data, const uint seed,
        const uint counter1, const uint counter2, const uint counter3)
{
    REAL4 r;
    const int i = get_global_id(0);
    const int i4 = i * 4;
    if (i4 >= n) return;
    uint4 rnd = rnd_uint4(seed, i, counter1, counter2, counter3);
    r.x = int_to_range_0_to_1_REAL(rnd.x);
    r.y = int_to_range_0_to_1_REAL(rnd.y);
    r.z = int_to_range_0_to_1_REAL(rnd.z);
    r.w = int_to_range_0_to_1_REAL(rnd.w);

    /* Store random numbers. */
    if (i4 <= n - 4)
    {
        ((global REAL4*) data)[i] = r;
    }
    else
    {
        /* End case only if length not divisible by 4. */
        data[i4] = r.x;
        if (i4 + 1 < n)
            data[i4 + 1] = r.y;
        if (i4 + 2 < n)
            data[i4 + 2] = r.z;
        if (i4 + 3 < n)
            data[i4 + 3] = r.w;
    }
}
