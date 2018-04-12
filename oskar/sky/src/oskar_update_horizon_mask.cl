kernel void update_horizon_mask_REAL(const int num_sources,
        global const REAL* restrict l,
        global const REAL* restrict m,
        global const REAL* restrict n,
        const REAL l_mul, const REAL m_mul, const REAL n_mul,
        global int* restrict mask)
{
    const int i = get_global_id(0);
    if (i >= num_sources) return;
    mask[i] |= ((l[i] * l_mul + m[i] * m_mul + n[i] * n_mul) > (REAL) 0);
}
