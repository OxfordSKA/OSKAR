#include "splines/oskar_dierckx_fpback.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_dierckx_fpback(const double *a, const double *z, int n, int k,
        double *c, int nest)
{
    /* Local variables */
    int i, j, l, m, i1, k1;
    double store;

    /* Parameter adjustments */
    --c;
    --z;
    a -= (1 + nest);

    /* Function Body */
    k1 = k - 1;
    c[n] = z[n] / a[n + nest];
    i = n - 1;
    if (i == 0) return;
    for (j = 2; j <= n; ++j)
    {
        store = z[i];
        i1 = k1;
        if (j <= k1) i1 = j - 1;
        m = i;
        for (l = 1; l <= i1; ++l)
        {
            ++m;
            store -= c[m] * a[i + (l + 1) * nest];
        }
        c[i] = store / a[i + nest];
        --i;
    }
}

#ifdef __cplusplus
}
#endif
