#include "splines/oskar_dierckx_sphere.h"
#include "splines/oskar_dierckx_fpsphe.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_dierckx_sphere(int iopt, int m, const double* theta,
        const double* phi, const double* r, const double* w, double s,
        int ntest, int npest, double eps, int* nt, double* tt, int* np,
        double* tp, double* c, double* fp, double* wrk1, int lwrk1,
        double* wrk2, int lwrk2, int* iwrk, int kwrk, int* ier)
{
    /* Local variables */
    int i = 0, j = 0, la = 0, lf = 0, ki = 0, lh = 0, kn = 0;
    int lq = 0, ib1 = 0, ib3 = 0;
    int np4 = 0, nt4 = 0, lcc = 0, ncc = 0, lff = 0, lco = 0;
    int lbp = 0, lbt = 0, lcs = 0, lfp = 0, lro = 0, npp = 0;
    int lsp = 0, lst = 0, ntt = 0, ncof = 0, nreg = 0, ncest = 0;
    int maxit = 0, nrint = 0, kwest = 0, lwest = 0;
    double tol = 0.0, pi = 0.0, pi2 = 0.0;

    /* Parameter adjustments */
    --tt;
    --tp;
    --wrk1;
    --wrk2;
    --iwrk;

    /* Function Body */
    /*  we set up the parameters tol and maxit. */
    maxit = 20;
    tol = 0.001;
    /*  before starting computations a data check is made. if the input data
     *  are invalid, control is immediately repassed to the calling program. */
    *ier = 10;
    if (eps <= 0.0 || eps >= 1.0) return;
    if (iopt < -1 || iopt > 1) return;
    if (m < 2) return;
    if (ntest < 8 || npest < 8) return;
    nt4 = ntest - 4;
    np4 = npest - 4;
    ncest = nt4 * np4;
    ntt = ntest - 7;
    npp = npest - 7;
    ncc = npp * (ntt - 1) + 6;
    nrint = ntt + npp;
    nreg = ntt * npp;
    ncof = npp * 3 + 6;
    ib1 = npp << 2;
    ib3 = ib1 + 3;
    if (ncof > ib1) ib1 = ncof;
    if (ncof > ib3) ib3 = ncof;
    lwest = 185 + 52 * npp + 10 * ntt + 14 * ntt * npp + 8 * (m + (ntt - 1) *
            (npp * npp));
    kwest = m + nreg;
    if (lwrk1 < lwest || kwrk < kwest) return;
    pi = 4.0 * atan(1.0);
    pi2 = pi + pi;
    if (iopt >= 0 && s < 0.0) return;
    if (iopt <= 0)
    {
        for (i = 0; i < m; ++i)
        {
            if (w[i] <= 0.0) return;
            if (theta[i] < 0.0 || theta[i] > pi) return;
            if (phi[i] < 0.0 || phi[i] > pi2) return;
        }
        if (iopt < 0)
        {
            ntt = *nt - 8;
            if (ntt < 0 || *nt > ntest) return;
            if (ntt != 0)
            {
                tt[4] = 0.0;
                for (i = 1; i <= ntt; ++i)
                {
                    j = i + 4;
                    if (tt[j] <= tt[j - 1] || tt[j] >= pi) return;
                }
            }
            npp = *np - 8;
            if (npp < 1 || *np > npest) return;
            tp[4] = 0.0;
            for (i = 1; i <= npp; ++i)
            {
                j = i + 4;
                if (tp[j] <= tp[j - 1] || tp[j] >= pi2) return;
            }
        }
    }
    *ier = 0;
    /*  we partition the working space and determine the spline approximation */
    kn = 1;
    ki = kn + m;
    lq = 2;
    la = lq + ncc * ib3;
    lf = la + ncc * ib1;
    lff = lf + ncc;
    lfp = lff + ncest;
    lco = lfp + nrint;
    lh = lco + nrint;
    lbt = lh + ib3;
    lbp = lbt + ntest * 5;
    lro = lbp + npest * 5;
    lcc = lro + npest;
    lcs = lcc + npest;
    lst = lcs + npest;
    lsp = lst + (m << 2);
    oskar_dierckx_fpsphe(iopt, m, theta, phi, r, w, s, ntest, npest, eps,
            tol, maxit, ncc, nt, &tt[1], np, &tp[1], c, fp, &wrk1[1],
            &wrk1[lfp], &wrk1[lco], &wrk1[lf], &wrk1[lff], &wrk1[lro],
            &wrk1[lcc], &wrk1[lcs], &wrk1[la], &wrk1[lq], &wrk1[lbt],
            &wrk1[lbp], &wrk1[lst], &wrk1[lsp], &wrk1[lh], &iwrk[ki],
            &iwrk[kn], &wrk2[1], lwrk2, ier);
}

#ifdef __cplusplus
}
#endif
