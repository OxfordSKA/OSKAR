kernel void mem_set_value_real_r_REAL(const int n, global REAL* a,
        const int offset, const REAL val)
{
    const int i = get_global_id(0) + offset;
    if (i >= n) return;
    a[i] = val;
}

kernel void mem_set_value_real_c_REAL(const int n, global REAL2* a,
        const int offset, const REAL val)
{
    REAL2 v;
    const int i = get_global_id(0) + offset;
    if (i >= n) return;
    v.x = val;
    v.y = (REAL) 0.;
    a[i] = v;
}

kernel void mem_set_value_real_m_REAL(const int n, global REAL8* a,
        const int offset, const REAL val)
{
    REAL8 v;
    const int i = get_global_id(0) + offset;
    if (i >= n) return;
    v.s0 = val;
    v.s1 = (REAL) 0.;
    v.s2 = (REAL) 0.;
    v.s3 = (REAL) 0.;
    v.s4 = (REAL) 0.;
    v.s5 = (REAL) 0.;
    v.s6 = val;
    v.s7 = (REAL) 0.;
    a[i] = v;
}
