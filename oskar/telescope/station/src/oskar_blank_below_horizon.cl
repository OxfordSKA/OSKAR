kernel void blank_below_horizon_scalar_REAL(const int n,
        global const REAL* restrict mask, global REAL2* restrict jones)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    if (mask[i] < (REAL) 0.0) {
        jones[i].x = (REAL) 0.0;
        jones[i].y = (REAL) 0.0;
    }
}

kernel void blank_below_horizon_matrix_REAL(const int n,
        global const REAL* restrict mask, global REAL8* restrict jones)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    if (mask[i] < (REAL) 0.0) {
        jones[i].s0 = (REAL) 0.0;
        jones[i].s1 = (REAL) 0.0;
        jones[i].s2 = (REAL) 0.0;
        jones[i].s3 = (REAL) 0.0;
        jones[i].s4 = (REAL) 0.0;
        jones[i].s5 = (REAL) 0.0;
        jones[i].s6 = (REAL) 0.0;
        jones[i].s7 = (REAL) 0.0;
    }
}
