kernel void evaluate_element_weights_dft_REAL(const int n,
        global const REAL* restrict x,
        global const REAL* restrict y,
        global const REAL* restrict z,
        const REAL wavenumber, const REAL x1, const REAL y1, const REAL z1,
        global REAL2* restrict weights)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    REAL re, im;
    const REAL p = wavenumber * (x[i] * x1 + y[i] * y1 + z[i] * z1);
    im = sincos(-p, &re);
    weights[i].x = re;
    weights[i].y = im;
}
