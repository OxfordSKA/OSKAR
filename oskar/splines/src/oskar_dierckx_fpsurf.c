#include "splines/oskar_dierckx_fpsurf.h"
#include "splines/oskar_dierckx_fpback.h"
#include "splines/oskar_dierckx_fpdisc.h"
#include "splines/oskar_dierckx_fporde.h"
#include "splines/oskar_dierckx_fprank.h"
#include "splines/oskar_dierckx_fpbspl.h"
#include "splines/oskar_dierckx_fprota.h"
#include "splines/oskar_dierckx_fpgivs.h"
#include "splines/oskar_dierckx_fprati.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NOLINTNEXTLINE(readability-function-size) */
void oskar_dierckx_fpsurf(int iopt, int m, double *x, double *y,
        const double *z, const double *w, double xb, double xe, double yb,
        double ye, int kxx, int kyy, double s, int nxest, int nyest,
        double eta, double tol, int maxit, int nmax, int nc, int *nx0,
        double *tx, int *ny0, double *ty, double *c, double *fp, double *fp0,
        double *fpint, double *coord, double *f, double *ff, double *a,
        double *q, double *bx, double *by, double *spx, double *spy, double *h,
        int *index, int *nummer, double *wrk, int lwrk, int *ier)
{
    /* System generated locals */
    int a_offset = 0, q_offset = 0, bx_offset = 0, by_offset = 0;
    double r1 = 0.0;

    /* Local variables */
    int i = 0, iband = 0, iband1 = 0, iband3 = 0, iband4 = 0, ibb = 0;
    int ichang = 0, ich1 = 0, ich3 = 0, ii = 0;
    int in = 0, irot = 0, iter = 0, i1 = 0, i2 = 0, i3 = 0, j = 0;
    int jrot = 0, jxy = 0, j1 = 0, kx = 0, kx1 = 0, kx2 = 0, ky = 0, ky1 = 0;
    int ky2 = 0, l = 0, la = 0, lf = 0, lh = 0, lwest = 0, lx = 0, ly = 0;
    int l1 = 0, l2 = 0, n = 0, ncof = 0, nk1x = 0;
    int nk1y = 0, nminx = 0, nminy = 0, nreg = 0, nrint = 0, num = 0, num1 = 0;
    int nx = 0, nxe = 0, nxx = 0, ny = 0;
    int nye = 0, nyy = 0, n1 = 0, rank = 0;
    double acc = 0.0, arg = 0.0, cos_ = 0.0, dmax = 0.0;
    double fac1 = 0.0, fac2 = 0.0, fpmax = 0.0, fpms = 0.0, f1 = 0.0, f2 = 0.0;
    double f3 = 0.0, eps = 0.0, hxi = 0.0, p = 0.0, pinv = 0.0, piv = 0.0;
    double p1 = 0.0, p2 = 0.0, p3 = 0.0, sigma = 0.0, sin_ = 0.0, sq = 0.0;
    double store = 0.0, wi = 0.0;
    double rn = 0.0, hx[6], hy[6], x0 = 0.0, x1 = 0.0, y0 = 0.0, y1 = 0.0;
    double zi = 0.0, con1 = 0.0, con4 = 0.0, con9 = 0.0;

    /* Parameter adjustments */
    --nummer;
    --w;
    --z;
    --y;
    --x;
    --ty;
    --tx;
    spy -= (1 + m);
    spx -= (1 + m);
    by_offset = 1 + nmax;
    by -= by_offset;
    bx_offset = 1 + nmax;
    bx -= bx_offset;
    --h;
    q_offset = 1 + nc;
    q -= q_offset;
    a_offset = 1 + nc;
    a -= a_offset;
    --ff;
    --f;
    --c;
    --coord;
    --fpint;
    --index;
    --wrk;

    /* Function Body */
    /*  set constants */
    con1 = 0.1;
    con9 = 0.9;
    con4 = 0.04;
    /*
     * part 1: determination of the number of knots and their position.
     * ****************************************************************
     * given a set of knots we compute the least-squares spline sinf(x,y),
     * and the corresponding weighted sum of squared residuals fp=f(p=inf).
     * if iopt=-1  sinf(x,y) is the requested approximation.
     * if iopt=0 or iopt=1 we check whether we can accept the knots:
     *   if fp <=s we will continue with the current set of knots.
     *   if fp > s we will increase the number of knots and compute the
     *      corresponding least-squares spline until finally  fp<=s.
     * the initial choice of knots depends on the value of s and iopt.
     *   if iopt=0 we first compute the least-squares polynomial of degree
     *     kx in x and ky in y; nx=nminx=2*kx+2 and ny=nminy=2*ky+2.
     *     fp0=f(0) denotes the corresponding weighted sum of squared
     *     residuals
     *   if iopt=1 we start with the knots found at the last call of the
     *     routine, except for the case that s>=fp0; then we can compute
     *     the least-squares polynomial directly.
     * eventually the independent variables x and y (and the corresponding
     * parameters) will be switched if this can reduce the bandwidth of the
     * system to be solved. */

    /*  ichang denotes whether(1) or not(-1) the directions have been inter-
     *  changed. */
    ichang = -1;
    x0 = xb;
    x1 = xe;
    y0 = yb;
    y1 = ye;
    kx = kxx;
    ky = kyy;
    kx1 = kx + 1;
    ky1 = ky + 1;
    nxe = nxest;
    nye = nyest;
    eps = sqrt(eta);
    if (iopt >= 0)
    {
        /*  calculation of acc, the absolute tolerance for the root
         *  of f(p)=s. */
        acc = tol * s;
        if (iopt != 0 && *fp0 > s)
        {
            nx = *nx0;
            ny = *ny0;
        }
        else
        {
            /*  initialization for the least-squares polynomial. */
            nminx = kx1 << 1;
            nminy = ky1 << 1;
            nx = nminx;
            ny = nminy;
            *ier = -2;
        }
    }
    else
    {
        nx = *nx0;
        ny = *ny0;
    }
    /*  main loop for the different sets of knots. m is a safe upper bound
     *  for the number of trials. */
    for (iter = 1; iter <= m; ++iter)
    {
        /*  find the position of the additional knots which are needed for
         *  the b-spline representation of s(x,y). */
        l = nx;
        for (i = 1; i <= kx1; ++i)
        {
            tx[i] = x0;
            tx[l] = x1;
            --l;
        }
        l = ny;
        for (i = 1; i <= ky1; ++i)
        {
            ty[i] = y0;
            ty[l] = y1;
            --l;
        }
        /*  find nrint, the total number of knot intervals and nreg, the
         *  number of panels in which the approximation domain is subdivided
         *  by the intersection of knots. */
        nxx = nx - (kx1 << 1) + 1;
        nyy = ny - (ky1 << 1) + 1;
        nrint = nxx + nyy;
        nreg = nxx * nyy;
        /*  find the bandwidth of the observation matrix a.
         *  if necessary, interchange the variables x and y, in order to obtain
         *  a minimal bandwidth. */
        iband1 = kx * (ny - ky1) + ky;
        l = ky * (nx - kx1) + kx;
        if (iband1 > l)
        {
            iband1 = l;
            ichang = -ichang;
            for (i = 1; i <= m; ++i)
            {
                store = x[i];
                x[i] = y[i];
                y[i] = store;
            }
            store = x0;
            x0 = y0;
            y0 = store;
            store = x1;
            x1 = y1;
            y1 = store;
            n = (nx < ny) ? nx : ny;
            for (i = 1; i <= n; ++i)
            {
                store = tx[i];
                tx[i] = ty[i];
                ty[i] = store;
            }
            n1 = n + 1;
            if (nx < ny)
            {
                for (i = n1; i <= ny; ++i)
                {
                    tx[i] = ty[i];
                }
            }
            else if (nx > ny)
            {
                for (i = n1; i <= nx; ++i)
                {
                    ty[i] = tx[i];
                }
            }
            l = nx;
            nx = ny;
            ny = l;
            l = nxe;
            nxe = nye;
            nye = l;
            l = nxx;
            nxx = nyy;
            nyy = l;
            l = kx;
            kx = ky;
            ky = l;
            kx1 = kx + 1;
            ky1 = ky + 1;
        }
        iband = iband1 + 1;
        /*  arrange the data points according to the panel they belong to. */
        oskar_dierckx_fporde(&x[1], &y[1], m, kx, ky, &tx[1], nx, &ty[1], ny,
                &nummer[1], &index[1], nreg);
        /*  find ncof, the number of b-spline coefficients. */
        nk1x = nx - kx1;
        nk1y = ny - ky1;
        ncof = nk1x * nk1y;
        /*  initialize the observation matrix a. */
        for (i = 1; i <= ncof; ++i)
        {
            f[i] = 0.0;
            for (j = 1; j <= iband; ++j)
            {
                a[i + j * nc] = 0.0;
            }
        }
        /*  initialize the sum of squared residuals. */
        *fp = 0.0;
        /*  fetch the data points in the new order. main loop for the
         *  different panels. */
        for (num = 1; num <= nreg; ++num)
        {
            /*  fix certain constants for the current panel; jrot records the
             *  column number of the first non-zero element in a row of the
             *  observation matrix according to a data point of the panel. */
            num1 = num - 1;
            lx = num1 / nyy;
            l1 = lx + kx1;
            ly = num1 - lx * nyy;
            l2 = ly + ky1;
            jrot = lx * nk1y + ly;
            /*  test whether there are still data points in the panel. */
            in = index[num];
            while (in != 0)
            {
                /*  fetch a new data point. */
                wi = w[in];
                zi = z[in] * wi;
                /*  evaluate for the x-direction, the (kx+1) non-zero
                 *  b-splines at x(in). */
                oskar_dierckx_fpbspl_d(&tx[1], kx, x[in], l1, hx);
                /*  evaluate for the y-direction, the (ky+1) non-zero
                 *  b-splines at y(in). */
                oskar_dierckx_fpbspl_d(&ty[1], ky, y[in], l2, hy);
                /*  store the value of these b-splines in spx and spy
                 *  respectively. */
                for (i = 1; i <= kx1; ++i)
                {
                    spx[in + i * m] = hx[i - 1];
                }
                for (i = 1; i <= ky1; ++i)
                {
                    spy[in + i * m] = hy[i - 1];
                }
                /*  initialize the new row of observation matrix. */
                for (i = 1; i <= iband; ++i)
                {
                    h[i] = 0.0;
                }
                /*  calculate the non-zero elements of the new row by making
                 *  the cross products of the non-zero b-splines in x- and
                 *  y-direction. */
                i1 = 0;
                for (i = 1; i <= kx1; ++i)
                {
                    hxi = hx[i - 1];
                    j1 = i1;
                    for (j = 1; j <= ky1; ++j)
                    {
                        ++j1;
                        h[j1] = hxi * hy[j - 1] * wi;
                    }
                    i1 += nk1y;
                }
                /*  rotate the row into triangle by givens transformations . */
                irot = jrot;
                for (i = 1; i <= iband; ++i)
                {
                    ++irot;
                    piv = h[i];
                    if (piv == 0.0) continue;
                    /*  calculate the parameters of the givens
                     *  transformation. */
                    oskar_dierckx_fpgivs(piv, &a[irot + nc], &cos_, &sin_);
                    /*  apply that transformation to the right hand side. */
                    oskar_dierckx_fprota(cos_, sin_, &zi, &f[irot]);
                    if (i == iband) break;
                    /*  apply that transformation to the left hand side. */
                    i2 = 1;
                    i3 = i + 1;
                    for (j = i3; j <= iband; ++j)
                    {
                        ++i2;
                        oskar_dierckx_fprota(cos_, sin_, &h[j],
                                &a[irot + i2 * nc]);
                    }
                }
                /*  add the contribution of the row to the sum of squares
                 *  of residual right hand sides. */
                *fp += zi * zi;
                /*  find the number of the next data point in the panel. */
                in = nummer[in];
            }
        }
        /*  find dmax, the maximum value for the diagonal elements in the
         *  reduced triangle. */
        dmax = 0.0;
        for (i = 1; i <= ncof; ++i)
        {
            if (a[i + nc] <= dmax) continue;
            dmax = a[i + nc];
        }
        /*  check whether the observation matrix is rank deficient. */
        sigma = eps * dmax;
        for (i = 1; i <= ncof; ++i)
        {
            if (a[i + nc] <= sigma)
            {
                i = -1;
                break;
            }
        }
        if (i != -1)
        {
            /*  backward substitution in case of full rank. */
            oskar_dierckx_fpback(&a[a_offset], &f[1], ncof, iband, &c[1], nc);
            rank = ncof;
            for (i = 1; i <= ncof; ++i)
            {
                q[i + nc] = a[i + nc] / dmax;
            }
        }
        else
        {
            /*  in case of rank deficiency, find the minimum norm solution. */
            /*  check whether there is sufficient working space */
            lwest = ncof * iband + ncof + iband;
            if (lwrk < lwest)
            {
                *ier = lwest;
                goto L830;
            }
            for (i = 1; i <= ncof; ++i)
            {
                ff[i] = f[i];
                for (j = 1; j <= iband; ++j)
                {
                    q[i + j * nc] = a[i + j * nc];
                }
            }
            lf = 1;
            lh = lf + ncof;
            la = lh + iband;
            oskar_dierckx_fprank(&q[q_offset], &ff[1], ncof, iband, nc,
                    sigma, &c[1],
                    &sq, &rank, &wrk[la], &wrk[lf], &wrk[lh]);
            for (i = 1; i <= ncof; ++i)
            {
                q[i + nc] /= dmax;
            }
            /*  add to the sum of squared residuals, the contribution of
             *  reducing the rank. */
             *fp += sq;
        }
        if (*ier == -2) *fp0 = *fp;
        /*  test whether the least-squares spline is an acceptable solution. */
        if (iopt < 0)
        {
            if (ncof != rank) *ier = -rank;
            goto L830;
        }
        fpms = *fp - s;
        if (fabs(fpms) <= acc)
        {
            if (*fp <= 0.0)
            {
                *ier = -1;
                *fp = 0.0;
            }
            if (ncof != rank) *ier = -rank;
            goto L830;
        }
        /*  test whether we can accept the choice of knots. */
        if (fpms < 0.0) break; /* Go to part 2. */
        /*  test whether we cannot further increase the number of knots. */
        if (ncof > m)
        {
            *ier = 4;
            goto L830;
        }
        *ier = 0;
        /*  search where to add a new knot.
         *  find for each interval the sum of squared residuals fpint for the
         *  data points having the coordinate belonging to that knot interval.
         *  calculate also coord which is the same sum, weighted by the
         *  position of the data points considered. */
        for (i = 1; i <= nrint; ++i)
        {
            fpint[i] = 0.0;
            coord[i] = 0.0;
        }
        for (num = 1; num <= nreg; ++num)
        {
            num1 = num - 1;
            lx = num1 / nyy;
            l1 = lx + 1;
            ly = num1 - lx * nyy;
            l2 = ly + 1 + nxx;
            jrot = lx * nk1y + ly;
            in = index[num];
            while (in != 0)
            {
                store = 0.0;
                i1 = jrot;
                for (i = 1; i <= kx1; ++i)
                {
                    hxi = spx[in + i * m];
                    j1 = i1;
                    for (j = 1; j <= ky1; ++j)
                    {
                        ++j1;
                        store += hxi * spy[in + j * m] * c[j1];
                    }
                    i1 += nk1y;
                }
                r1 = w[in] * (z[in] - store);
                store = r1 * r1;
                fpint[l1] += store;
                coord[l1] += store * x[in];
                fpint[l2] += store;
                coord[l2] += store * y[in];
                in = nummer[in];
            }
        }
        /*  find the interval for which fpint is maximal on the condition
         *  that there still can be added a knot. */
        for (;;)
        {
            l = 0;
            fpmax = 0.0;
            l1 = 1;
            l2 = nrint;
            if (nx == nxe) l1 = nxx + 1;
            if (ny == nye) l2 = nxx;
            if (l1 > l2)
            {
                *ier = 1;
                goto L830;
            }
            for (i = l1; i <= l2; ++i)
            {
                if (fpmax >= fpint[i]) continue;
                l = i;
                fpmax = fpint[i];
            }
            /*  test whether we cannot further increase the number of knots. */
            if (l == 0)
            {
                *ier = 5;
                goto L830;
            }
            /*  calculate the position of the new knot. */
            arg = coord[l] / fpint[l];
            /*  test in what direction the new knot is going to be added. */
            if (l <= nxx)
            {
                /*  addition in the x-direction. */
                jxy = l + kx1;
                fpint[l] = 0.0;
                fac1 = tx[jxy] - arg;
                fac2 = arg - tx[jxy - 1];
                if (fac1 > 10.0 * fac2 || fac2 > 10.0 * fac1) continue;
                j = nx;
                for (i = jxy; i <= nx; ++i)
                {
                    tx[j + 1] = tx[j];
                    --j;
                }
                tx[jxy] = arg;
                ++nx;
            }
            else
            {
                /*  addition in the y-direction. */
                jxy = l + ky1 - nxx;
                fpint[l] = 0.0;
                fac1 = ty[jxy] - arg;
                fac2 = arg - ty[jxy - 1];
                if (fac1 > 10.0 * fac2 || fac2 > 10.0 * fac1) continue;
                j = ny;
                for (i = jxy; i <= ny; ++i)
                {
                    ty[j + 1] = ty[j];
                    --j;
                }
                ty[jxy] = arg;
                ++ny;
            }
            break;
        }
        /*  restart the computations with the new set of knots. */
    }
    /*  test whether the least-squares polynomial is a solution of our
     *  approximation problem. */
    if (*ier == -2) goto L830;
    /*
     * part 2: determination of the smoothing spline sp(x,y)
     * *****************************************************
     * we have determined the number of knots and their position. we now
     * compute the b-spline coefficients of the smoothing spline sp(x,y).
     * the observation matrix a is extended by the rows of a matrix,
     * expressing that sp(x,y) must be a polynomial of degree kx in x and
     * ky in y. the corresponding weights of these additional rows are set
     * to 1./p.  iteratively we than have to determine the value of p
     * such that f(p)=sum((w(i)*(z(i)-sp(x(i),y(i))))**2) be = s.
     * we already know that the least-squares polynomial corresponds to
     * p=0  and that the least-squares spline corresponds to p=infinity.
     * the iteration process which is proposed here makes use of rational
     * interpolation. since f(p) is a convex and strictly decreasing
     * function of p, it can be approximated by a rational function r(p)=
     * (u*p+v)/(p+w). three values of p(p1,p2,p3) with corresponding values
     * of f(p) (f1=f(p1)-s,f2=f(p2)-s,f3=f(p3)-s) are used to calculate the
     * new value of p such that r(p)=s. convergence is guaranteed by taking
     * f1 > 0 and f3 < 0. */

    kx2 = kx1 + 1;
    /*  test whether there are interior knots in the x-direction. */
    if (nk1x != kx1)
    {
        /*  evaluate the discotinuity jumps of the kx-th order derivative of
         *  the b-splines at the knots tx(l),l=kx+2,...,nx-kx-1. */
        oskar_dierckx_fpdisc(&tx[1], nx, kx2, &bx[bx_offset], nmax);
    }
    ky2 = ky1 + 1;
    /*  test whether there are interior knots in the y-direction. */
    if (nk1y != ky1)
    {
        /*  evaluate the discontinuity jumps of the ky-th order derivative of
         *  the b-splines at the knots ty(l),l=ky+2,...,ny-ky-1. */
        oskar_dierckx_fpdisc(&ty[1], ny, ky2, &by[by_offset], nmax);
    }
    /*  initial value for p. */
    p1 = 0.0;
    f1 = *fp0 - s;
    p3 = -1.0;
    f3 = fpms;
    p = 0.0;
    for (i = 1; i <= ncof; ++i)
    {
        p += a[i + nc];
    }
    rn = (double) ncof;
    p = rn / p;
    /*  find the bandwidth of the extended observation matrix. */
    iband3 = kx1 * nk1y;
    iband4 = iband3 + 1;
    ich1 = 0;
    ich3 = 0;
    /*  iteration process to find the root of f(p)=s. */
    for (iter = 1; iter <= maxit; ++iter)
    {
        pinv = 1.0 / p;
        /*  store the triangularized observation matrix into q. */
        for (i = 1; i <= ncof; ++i)
        {
            ff[i] = f[i];
            for (j = 1; j <= iband; ++j)
            {
                q[i + j * nc] = a[i + j * nc];
            }
            ibb = iband + 1;
            for (j = ibb; j <= iband4; ++j)
            {
                q[i + j * nc] = 0.0;
            }
        }
        if (nk1y != ky1)
        {
            /*  extend the observation matrix with the rows of a matrix,
             *  expressing that for x=cst. sp(x,y) must be a polynomial in y of
             *  degree ky. */
            for (i = ky2; i <= nk1y; ++i)
            {
                ii = i - ky1;
                for (j = 1; j <= nk1x; ++j)
                {
                    /*  initialize the new row. */
                    for (l = 1; l <= iband; ++l)
                    {
                        h[l] = 0.0;
                    }
                    /*  fill in the non-zero elements of the row. jrot records
                     *  the column number of the first non-zero element in the
                     *  row. */
                    for (l = 1; l <= ky2; ++l)
                    {
                        h[l] = by[ii + l * nmax] * pinv;
                    }
                    zi = 0.0;
                    jrot = (j - 1) * nk1y + ii;
                    /*  rotate the new row into triangle by givens
                     *  transformations without square roots. */
                    for (irot = jrot; irot <= ncof; ++irot)
                    {
                        piv = h[1];
                        i2 = (iband1 < ncof - irot) ? iband1 : ncof - irot;
                        if (piv == 0.0)
                        {
                            if (i2 <= 0) break;
                        }
                        else
                        {
                            /*  calculate the parameters of the givens
                             *  transformation. */
                            oskar_dierckx_fpgivs(piv, &q[irot + nc],
                                    &cos_, &sin_);
                            /*  apply that givens transformation to the right
                             *  hand side. */
                            oskar_dierckx_fprota(cos_, sin_, &zi, &ff[irot]);
                            if (i2 == 0) break;
                            /*  apply that givens transformation to the left
                             *  hand side. */
                            for (l = 1; l <= i2; ++l)
                            {
                                l1 = l + 1;
                                oskar_dierckx_fprota(cos_, sin_, &h[l1],
                                        &q[irot + l1 * nc]);
                            }
                        }
                        for (l = 1; l <= i2; ++l)
                        {
                            h[l] = h[l + 1];
                        }
                        h[i2 + 1] = 0.0;
                    }
                }
            }
        }
        if (nk1x != kx1)
        {
            /*  extend the observation matrix with the rows of a matrix
             *  expressing that for y=cst. sp(x,y) must be a polynomial in x of
             *  degree kx. */
            for (i = kx2; i <= nk1x; ++i)
            {
                ii = i - kx1;
                for (j = 1; j <= nk1y; ++j)
                {
                    /*  initialize the new row */
                    for (l = 1; l <= iband4; ++l)
                    {
                        h[l] = 0.0;
                    }
                    /*  fill in the non-zero elements of the row. jrot records
                     *  the column number of the first non-zero element in the
                     *  row. */
                    j1 = 1;
                    for (l = 1; l <= kx2; ++l)
                    {
                        h[j1] = bx[ii + l * nmax] * pinv;
                        j1 += nk1y;
                    }
                    zi = 0.0;
                    jrot = (i - kx2) * nk1y + j;
                    /*  rotate the new row into triangle by givens
                     *  transformations. */
                    for (irot = jrot; irot <= ncof; ++irot)
                    {
                        piv = h[1];
                        i2 = (iband3 < ncof - irot) ? iband3 : ncof - irot;
                        if (piv == 0.0)
                        {
                            if (i2 <= 0) break;
                        }
                        else
                        {
                            /*  calculate the parameters of the givens
                             *  transformation. */
                            oskar_dierckx_fpgivs(piv, &q[irot + nc],
                                    &cos_, &sin_);
                            /*  apply that givens transformation to the right
                             *  hand side. */
                            oskar_dierckx_fprota(cos_, sin_, &zi, &ff[irot]);
                            if (i2 == 0) break;
                            /*  apply that givens transformation to the left
                             *  hand side. */
                            for (l = 1; l <= i2; ++l)
                            {
                                l1 = l + 1;
                                oskar_dierckx_fprota(cos_, sin_, &h[l1],
                                        &q[irot + l1 * nc]);
                            }
                        }
                        for (l = 1; l <= i2; ++l)
                        {
                            h[l] = h[l + 1];
                        }
                        h[i2 + 1] = 0.0;
                    }
                }
            }
        }
        /*  find dmax, the maximum value for the diagonal elements in the
         *  reduced triangle. */
        dmax = 0.0;
        for (i = 1; i <= ncof; ++i)
        {
            if (q[i + nc] <= dmax) continue;
            dmax = q[i + nc];
        }
        /*  check whether the matrix is rank deficient. */
        sigma = eps * dmax;
        for (i = 1; i <= ncof; ++i)
        {
            if (q[i + nc] <= sigma)
            {
                i = -1;
                break;
            }
        }
        if (i != -1)
        {
            /*  backward substitution in case of full rank. */
            oskar_dierckx_fpback(&q[q_offset], &ff[1], ncof, iband4,
                    &c[1], nc);
            rank = ncof;
        }
        else
        {
            /*  in case of rank deficiency, find the minimum norm solution. */
            lwest = ncof * iband4 + ncof + iband4;
            if (lwrk < lwest)
            {
                *ier = lwest;
                goto L830;
            }
            lf = 1;
            lh = lf + ncof;
            la = lh + iband4;
            oskar_dierckx_fprank(&q[q_offset], &ff[1], ncof, iband4, nc,
                    sigma, &c[1], &sq, &rank, &wrk[la], &wrk[lf], &wrk[lh]);
        }
        for (i = 1; i <= ncof; ++i)
        {
            q[i + nc] /= dmax;
        }
        /*  compute f(p). */
        *fp = 0.0;
        for (num = 1; num <= nreg; ++num)
        {
            num1 = num - 1;
            lx = num1 / nyy;
            ly = num1 - lx * nyy;
            jrot = lx * nk1y + ly;
            in = index[num];
            while (in != 0)
            {
                store = 0.0;
                i1 = jrot;
                for (i = 1; i <= kx1; ++i)
                {
                    hxi = spx[in + i * m];
                    j1 = i1;
                    for (j = 1; j <= ky1; ++j)
                    {
                        ++j1;
                        store += hxi * spy[in + j * m] * c[j1];
                    }
                    i1 += nk1y;
                }
                /* Computing 2nd power */
                r1 = w[in] * (z[in] - store);
                *fp += r1 * r1;
                in = nummer[in];
            }
        }
        /*  test whether the approximation sp(x,y) is an acceptable solution. */
        fpms = *fp - s;
        if (fabs(fpms) <= acc)
        {
            if (ncof != rank) *ier = -rank;
            break;
        }
        /*  test whether the maximum allowable number of iterations has been
         *  reached. */
        if (iter == maxit)
        {
            *ier = 3;
            break;
        }
        /*  carry out one more step of the iteration process. */
        p2 = p;
        f2 = fpms;
        if (ich3 == 0)
        {
            if (!(f2 - f3 > acc))
            {
                /*  our initial choice of p is too large. */
                p3 = p2;
                f3 = f2;
                p *= con4;
                if (p <= p1) p = p1 * con9 + p2 * con1;
                continue;
            }
            if (f2 < 0.0) ich3 = 1;
        }
        if (ich1 == 0)
        {
            if (!(f1 - f2 > acc))
            {
                /*  our initial choice of p is too small */
                p1 = p2;
                f1 = f2;
                p /= con4;
                if (p3 < 0.0) continue;
                if (p >= p3) p = p2 * con1 + p3 * con9;
                continue;
            }
            if (f2 > 0.0) ich1 = 1;
        }
        /*  test whether the iteration process proceeds as theoretically
         *  expected. */
        if (f2 >= f1 || f2 <= f3)
        {
            *ier = 2;
            break;
        }
        /*  find the new value of p. */
        p = oskar_dierckx_fprati(&p1, &f1, p2, f2, &p3, &f3);
    }
    /*  test whether x and y are in the original order. */
L830:
    if (ichang > 0)
    {
        /*  if not, interchange x and y once more. */
        l1 = 1;
        for (i = 1; i <= nk1x; ++i)
        {
            l2 = i;
            for (j = 1; j <= nk1y; ++j)
            {
                f[l2] = c[l1];
                ++l1;
                l2 += nk1x;
            }
        }
        for (i = 1; i <= ncof; ++i)
        {
            c[i] = f[i];
        }
        for (i = 1; i <= m; ++i)
        {
            store = x[i];
            x[i] = y[i];
            y[i] = store;
        }
        n = (nx < ny) ? nx : ny;
        for (i = 1; i <= n; ++i)
        {
            store = tx[i];
            tx[i] = ty[i];
            ty[i] = store;
        }
        if (nx < ny)
        {
            for (i = n + 1; i <= ny; ++i)
            {
                tx[i] = ty[i];
            }
        }
        else if (nx > ny)
        {
            for (i = n + 1; i <= nx; ++i)
            {
                ty[i] = tx[i];
            }
        }
        l = nx;
        nx = ny;
        ny = l;
    }
    if (iopt < 0) return;
    *nx0 = nx;
    *ny0 = ny;
}

#ifdef __cplusplus
}
#endif
