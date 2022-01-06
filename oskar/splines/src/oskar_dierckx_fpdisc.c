#include "splines/oskar_dierckx_fpdisc.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_dierckx_fpdisc(const double *t, int n, int k2, double *b, int nest)
{
    /* Local variables */
    double h[12] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    int i = 0, j = 0, k = 0, l = 0, k1 = 0, ik = 0, jk = 0;
    int lj = 0, lk = 0, lp = 0, nk1 = 0, lmk = 0, nrint = 0;
    double an = 0.0, fac = 0.0, prod = 0.0;

    /* Parameter adjustments */
    --t;
    b -= (1 + nest);

    /* Function Body */
    k1 = k2 - 1;
    k = k1 - 1;
    nk1 = n - k1;
    nrint = nk1 - k;
    an = (double) nrint;
    fac = an / (t[nk1 + 1] - t[k1]);
    for (l = k2; l <= nk1; ++l)
    {
        lmk = l - k1;
        for (j = 1; j <= k1; ++j)
        {
            ik = j + k1;
            lj = l + j;
            lk = lj - k2;
            h[j - 1] = t[l] - t[lk];
            h[ik - 1] = t[l] - t[lj];
        }
        lp = lmk;
        for (j = 1; j <= k2; ++j)
        {
            jk = j;
            prod = h[j - 1];
            for (i = 1; i <= k; ++i)
            {
                ++jk;
                prod = prod * h[jk - 1] * fac;
            }
            lk = lp + k1;
            b[lmk + j * nest] = (t[lk] - t[lp]) / prod;
            ++lp;
        }
    }
}

#ifdef __cplusplus
}
#endif
