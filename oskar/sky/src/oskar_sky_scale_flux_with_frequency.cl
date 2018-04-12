kernel void scale_flux_with_frequency_REAL(
        const int num_sources, const REAL frequency,
        global REAL* restrict I, global REAL* restrict Q,
        global REAL* restrict U, global REAL* restrict V,
        global REAL* restrict ref_freq,
        global const REAL* restrict sp_index,
        global const REAL* restrict rm)
{
    const int i = get_global_id(0);
    if (i >= num_sources) return;
    REAL cos_b;
    const REAL freq0 = ref_freq[i];
    if (freq0 == (REAL) 0) return;

    /* Compute delta_lambda_sq using factorised difference of two
     * squares. (Numerically superior than an explicit difference.)
     * This is (lambda^2 - lambda0^2) */
    const REAL lambda  = ((REAL) 299792458.) / frequency;
    const REAL lambda0 = ((REAL) 299792458.) / freq0;
    const REAL delta_lambda_sq = (lambda - lambda0) * (lambda + lambda0);

    /* Compute rotation factors, sin(2 beta) and cos(2 beta). */
    const REAL b = ((REAL) 2) * rm[i] * delta_lambda_sq;
    const REAL sin_b = sincos(b, &cos_b);

    /* Compute spectral index scaling factor. */
    const REAL scale = pow(frequency / freq0, sp_index[i]);

    /* Set new values and update reference frequency. */
    const REAL Q_ = scale * Q[i];
    const REAL U_ = scale * U[i];
    I[i] *= scale;
    V[i] *= scale;
    Q[i] = Q_ * cos_b - U_ * sin_b;
    U[i] = Q_ * sin_b + U_ * cos_b;
    ref_freq[i] = frequency;
}
