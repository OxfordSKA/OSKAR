kernel void gaussian_2d_complex_REAL(const int n,
        global const REAL* restrict x, global const REAL* restrict y,
        const REAL inv_2_var, global REAL2* restrict z)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    const REAL x_ = x[i];
    const REAL y_ = y[i];
    const REAL arg = (x_*x_ + y_*y_) * inv_2_var;
    z[i].x = exp(-arg);
    z[i].y = (REAL) 0.0;
}

kernel void gaussian_2d_matrix_REAL(const int n,
        global const REAL* restrict x, global const REAL* restrict y,
        const REAL inv_2_var, global REAL8* restrict z)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    const REAL x_ = x[i];
    const REAL y_ = y[i];
    const REAL arg = (x_*x_ + y_*y_) * inv_2_var;
    const REAL value = exp(-arg);
    z[i].s0 = value;
    z[i].s1 = (REAL) 0.0;
    z[i].s2 = (REAL) 0.0;
    z[i].s3 = (REAL) 0.0;
    z[i].s4 = (REAL) 0.0;
    z[i].s5 = (REAL) 0.0;
    z[i].s6 = value;
    z[i].s7 = (REAL) 0.0;
}
