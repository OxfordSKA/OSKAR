#include <oskar_dierckx_fpdisc.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_dierckx_fpdisc(const double *t, int n, int k2, double *b, int nest)
{
    /* Local variables */
    double h[12];
    int i, j, k, l, k1, ik, jk, lj, lk, lp, nk1, lmk, nrint;
    double an, fac, prod;

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
