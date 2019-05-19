uint4 rnd_uint4(const uint seed,
        const uint c0, const uint c1, const uint c2, const uint c3);

inline Real int_to_range_0_to_1_Real(const uint in)
{
    const Real factor = (Real) 1.0 / ((Real) 1.0 + 0xFFFFFFFFu);
    const Real half_factor = (Real) 0.5 * factor;
    return (in * factor) + half_factor;
}

inline Real int_to_range_minus_1_to_1_Real(const uint in)
{
    const Real factor = (Real) 1.0 / ((Real) 1.0 + 0x7FFFFFFFu);
    const Real half_factor = (Real) 0.5 * factor;
    return (((int)in) * factor) + half_factor;
}

inline Real2 box_muller_Real(const uint u0, const uint u1)
{
    Real2 rnd;
    Real sin_t, cos_t, r, t;
    t = (Real) 3.14159265358979323846264338327950288;
    t *= int_to_range_minus_1_to_1_Real(u0);
    sin_t = sincos(t, &cos_t);
    r = sqrt((Real) -2.0 * log(int_to_range_0_to_1_Real(u1)));
    rnd.x = sin_t * r;
    rnd.y = cos_t * r;
    return rnd;
}

kernel void mem_random_gaussian_Real(const uint n,
        global Real* restrict data, const uint seed,
        const uint counter1, const uint counter2, const uint counter3,
        const Real std)
{
    Real4 r;
    const uint i = get_global_id(0);
    const uint i4 = i * 4;
    if (i4 >= n) return;
    uint4 rnd = rnd_uint4(seed, i, counter1, counter2, counter3);
    r.xy = box_muller_Real(rnd.x, rnd.y);
    r.zw = box_muller_Real(rnd.z, rnd.w);
    r *= std;

    /* Store random numbers. */
    if (i4 <= n - 4)
    {
        ((global Real4*) data)[i] = r;
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

kernel void mem_random_uniform_Real(const uint n,
        global Real* restrict data, const uint seed,
        const uint counter1, const uint counter2, const uint counter3)
{
    Real4 r;
    const uint i = get_global_id(0);
    const uint i4 = i * 4;
    if (i4 >= n) return;
    uint4 rnd = rnd_uint4(seed, i, counter1, counter2, counter3);
    r.x = int_to_range_0_to_1_Real(rnd.x);
    r.y = int_to_range_0_to_1_Real(rnd.y);
    r.z = int_to_range_0_to_1_Real(rnd.z);
    r.w = int_to_range_0_to_1_Real(rnd.w);

    /* Store random numbers. */
    if (i4 <= n - 4)
    {
        ((global Real4*) data)[i] = r;
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
