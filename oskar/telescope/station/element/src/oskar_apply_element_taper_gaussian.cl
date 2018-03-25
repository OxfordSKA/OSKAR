kernel void apply_element_taper_gaussian_scalar_REAL(const int n,
        const REAL inv_2sigma_sq, global const REAL* restrict theta,
        global REAL2* restrict jones)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    REAL theta_sq = theta[i];
    theta_sq *= theta_sq;
    const REAL f = exp(-theta_sq * inv_2sigma_sq);
    jones[i].x *= f;
    jones[i].y *= f;
}

kernel void apply_element_taper_gaussian_matrix_REAL(const int n,
        const REAL inv_2sigma_sq, global const REAL* restrict theta,
        global REAL8* restrict jones)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    REAL theta_sq = theta[i];
    theta_sq *= theta_sq;
    const REAL f = exp(-theta_sq * inv_2sigma_sq);
    jones[i].s0 *= f;
    jones[i].s1 *= f;
    jones[i].s2 *= f;
    jones[i].s3 *= f;
    jones[i].s4 *= f;
    jones[i].s5 *= f;
    jones[i].s6 *= f;
    jones[i].s7 *= f;
}
