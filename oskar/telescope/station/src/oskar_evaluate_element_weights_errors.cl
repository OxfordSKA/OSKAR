kernel void evaluate_element_weights_errors_REAL(const int n,
        global const REAL* restrict amp_gain,
        global const REAL* restrict amp_error,
        global const REAL* restrict phase_offset,
        global const REAL* restrict phase_error,
        global REAL2* restrict errors)
{
    REAL2 r;
    REAL re, im;
    const int i = get_global_id(0);
    if (i >= n) return;

    /* Get two random numbers from a normalised Gaussian distribution. */
    r = errors[i];

    /* Evaluate the real and imaginary components of the error weight
     * for the antenna. */
    r.x *= amp_error[i];
    r.x += amp_gain[i];
    r.y *= phase_error[i];
    r.y += phase_offset[i];
    im = sincos(r.y, &re);
    re *= r.x;
    im *= r.x;
    errors[i].x = re;
    errors[i].y = im;
}
