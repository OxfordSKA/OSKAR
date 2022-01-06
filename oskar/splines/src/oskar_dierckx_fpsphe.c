#include "splines/oskar_dierckx_fpsphe.h"
#include "splines/oskar_dierckx_fpback.h"
#include "splines/oskar_dierckx_fpdisc.h"
#include "splines/oskar_dierckx_fporde.h"
#include "splines/oskar_dierckx_fprank.h"
#include "splines/oskar_dierckx_fpbspl.h"
#include "splines/oskar_dierckx_fprota.h"
#include "splines/oskar_dierckx_fpgivs.h"
#include "splines/oskar_dierckx_fprpsp.h"
#include "splines/oskar_dierckx_fprati.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NOLINTNEXTLINE(readability-function-size) */
void oskar_dierckx_fpsphe(int iopt, int m, const double* theta,
        const double* phi, const double* r, const double* w, double s,
        int ntest, int npest, double eta, double tol, int maxit, int ncc,
        int* nt, double* tt, int* np, double* tp, double* c, double* fp,
        double* sup, double* fpint, double* coord, double* f, double* ff,
        double* row, double* coco, double* cosi, double* a, double* q,
        double* bt, double* bp, double* spt, double* spp, double* h,
        int* index, int* nummer, double* wrk, int lwrk, int* ier)
{
    /* System generated locals */
    int a_dim1 = 0, a_offset = 0, q_dim1 = 0, q_offset = 0;
    int bt_dim1 = 0, bt_offset = 0, bp_dim1 = 0, bp_offset = 0;
    double r1 = 0.0;

    /* Local variables */
    int i = 0, j = 0, l = 0, i1 = 0, i2 = 0, i3 = 0, j1 = 0, j2 = 0;
    int l1 = 0, l2 = 0, l3 = 0, l4 = 0, la = 0, ii = 0, ij = 0, il = 0, in = 0;
    int lf = 0, lh = 0, ll = 0, lp = 0, lt = 0, np4 = 0, nt4 = 0, nt6 = 0;
    int jlt = 0, npp = 0, num = 0, nrr = 0;
    int ntt = 0, ich1 = 0, ich3 = 0, num1 = 0, ncof = 0, nreg = 0, rank = 0;
    int iter = 0, irot = 0, jrot = 0;
    int iband = 0, ncoff = 0, nrint = 0;
    int iband1 = 0, lwest = 0, iband3 = 0, iband4 = 0;
    double p = 0.0, c1 = 0.0, d1 = 0.0, d2 = 0.0, f1 = 0.0, f2 = 0.0, f3 = 0.0;
    double p1 = 0.0, p2 = 0.0, p3 = 0.0, aa = 0.0, cn = 0.0, co = 0.0;
    double fn = 0.0, ri = 0.0, si = 0.0;
    double wi = 0.0, rn = 0.0, sq = 0.0, acc = 0.0, arg = 0.0;
    double hti = 0.0, htj = 0.0, eps = 0.0, piv = 0.0, fac1 = 0.0, fac2 = 0.0;
    double facc = 0.0, facs = 0.0, dmax = 0.0, fpms = 0.0, pinv = 0.0;
    double sigma = 0.0, fpmax = 0.0, store = 0.0, pi = 0.0, pi2 = 0.0;
    double ht[4], hp[4];

    /* Parameter adjustments */
    --nummer;
    spp -= (1 + m);
    spt -= (1 + m);
    --w;
    --r;
    --phi;
    --theta;
    bt_dim1 = ntest;
    bt_offset = 1 + bt_dim1;
    bt -= bt_offset;
    --tt;
    bp_dim1 = npest;
    bp_offset = 1 + bp_dim1;
    bp -= bp_offset;
    --cosi;
    --coco;
    --row;
    --tp;
    --h;
    --ff;
    --c;
    q_dim1 = ncc;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    a_dim1 = ncc;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --f;
    --coord;
    --fpint;
    --index;
    --wrk;

    /* Function Body */
    pi = 4.0 * atan(1.0);
    pi2 = pi + pi;
    eps = sqrt(eta);
    /*  calculation of acc, the absolute tolerance for the root of f(p)=s. */
    acc = tol * s;
    if (iopt < 0) {
        goto L70;
    }
    if (iopt != 0 && s < *sup)
    {
        if (*np - 11 >= 0) {
            goto L70;
        } else {
            goto L60;
        }
    }

    /*  if iopt=0 we begin by computing the weighted least-squares polynomial
     *  of the form
     *     s(theta,phi) = c1*f1(theta) + cn*fn(theta)
     *  where f1(theta) and fn(theta) are the cubic polynomials satisfying
     *     f1(0) = 1, f1(pi) = f1'(0) = f1'(pi) = 0 ; fn(theta) = 1-f1(theta).
     *  the corresponding weighted sum of squared residuals gives the upper
     *  bound sup for the smoothing factor s. */
    *sup = 0.0;
    d1 = 0.0;
    d2 = 0.0;
    c1 = 0.0;
    cn = 0.0;
    fac1 = pi * (1.0 + 0.5);
    fac2 = (1.0 + 1.0) / (pi * (pi * pi));
    aa = 0.0;
    for (i = 1; i <= m; ++i)
    {
        wi = w[i];
        ri = r[i] * wi;
        arg = theta[i];
        fn = fac2 * arg * arg * (fac1 - arg);
        f1 = (1.0 - fn) * wi;
        fn *= wi;
        if (fn != 0.0)
        {
            oskar_dierckx_fpgivs(fn, &d1, &co, &si);
            oskar_dierckx_fprota(co, si, &f1, &aa);
            oskar_dierckx_fprota(co, si, &ri, &cn);
        }
        if (f1 != 0.0)
        {
            oskar_dierckx_fpgivs(f1, &d2, &co, &si);
            oskar_dierckx_fprota(co, si, &ri, &c1);
        }
        *sup += ri * ri;
    }
    if (d2 != 0.0) c1 /= d2;
    if (d1 != 0.0) cn = (cn - aa * c1) / d1;
    /*  find the b-spline representation of this least-squares polynomial */
    *nt = 8;
    *np = 8;
    for (i = 1; i <= 4; ++i)
    {
        c[i] = c1;
        c[i + 4] = c1;
        c[i + 8] = cn;
        c[i + 12] = cn;
        tt[i] = 0.0;
        tt[i + 4] = pi;
        tp[i] = 0.0;
        tp[i + 4] = pi2;
    }
    *fp = *sup;
    /*  test whether the least-squares polynomial is an acceptable solution */
    fpms = *sup - s;
    if (fpms < acc)
    {
        *ier = -2;
        return;
    }
    /*  test whether we cannot further increase the number of knots. */
L60:
    if (npest < 11 || ntest < 9)
    {
        *ier = 1;
        return;
    }
    /*  find the initial set of interior knots of the spherical spline in
     *  case iopt = 0. */
    *np = 11;
    tp[5] = pi * 0.5;
    tp[6] = pi;
    tp[7] = tp[5] + pi;
    *nt = 9;
    tt[5] = tp[5];
L70:
    /*
     *  part 1 : computation of least-squares spherical splines.
     *  ********************************************************
     *  if iopt < 0 we compute the least-squares spherical spline according
     *  to the given set of knots.
     *  if iopt >=0 we compute least-squares spherical splines with increas-
     *  ing numbers of knots until the corresponding sum f(p=inf)<=s.
     *  the initial set of knots then depends on the value of iopt:
     *    if iopt=0 we start with one interior knot in the theta-direction
     *              (pi/2) and three in the phi-direction (pi/2,pi,3*pi/2).
     *    if iopt>0 we start with the set of knots found at the last call
     *              of the routine. */

    /*  main loop for the different sets of knots. m is a safe upper bound
     *  for the number of trials. */
    for (iter = 1; iter <= m; ++iter)
    {
        /*  find the position of the additional knots which are needed for
         *  the b-spline representation of s(theta,phi). */
        l1 = 4;
        l2 = l1;
        l3 = *np - 3;
        l4 = l3;
        tp[l2] = 0.0;
        tp[l3] = pi2;
        for (i = 1; i <= 3; ++i)
        {
            ++l1;
            --l2;
            ++l3;
            --l4;
            tp[l2] = tp[l4] - pi2;
            tp[l3] = tp[l1] + pi2;
        }
        l = *nt;
        for (i = 1; i <= 4; ++i)
        {
            tt[i] = 0.0;
            tt[l] = pi;
            --l;
        }
        /*  find nrint, the total number of knot intervals and nreg, the number
         *  of panels in which the approximation domain is subdivided by the
         *  intersection of knots. */
        ntt = *nt - 7;
        npp = *np - 7;
        nrr = npp / 2;
        nrint = ntt + npp;
        nreg = ntt * npp;
        /*  arrange the data points according to the panel they belong to. */
        oskar_dierckx_fporde(&theta[1], &phi[1], m, 3, 3, &tt[1], *nt,
                &tp[1], *np, &nummer[1], &index[1], nreg);
        /*  find the b-spline coefficients coco and cosi of the cubic spline
         *  approximations sc(phi) and ss(phi) for cos(phi) and sin(phi). */
        for (i = 1; i <= npp; ++i)
        {
            coco[i] = 0.0;
            cosi[i] = 0.0;
            for (j = 1; j <= npp; ++j)
            {
                a[i + j * a_dim1] = 0.0;
            }
        }
        /*  the coefficients coco and cosi are obtained from the conditions
         *  sc(tp(i))=cos(tp(i)),resp. ss(tp(i))=sin(tp(i)),i=4,5,...np-4. */
        for (i = 1; i <= npp; ++i)
        {
            l2 = i + 3;
            arg = tp[l2];
            oskar_dierckx_fpbspl_d(&tp[1], 3, arg, l2, hp);
            for (j = 1; j <= npp; ++j)
            {
                row[j] = 0.0;
            }
            ll = i;
            for (j = 1; j <= 3; ++j)
            {
                if (ll > npp) ll = 1;
                row[ll] += hp[j - 1];
                ++ll;
            }
            facc = cos(arg);
            facs = sin(arg);
            for (j = 1; j <= npp; ++j)
            {
                piv = row[j];
                if (piv == 0.0) continue;
                oskar_dierckx_fpgivs(piv, &a[j + a_dim1], &co, &si);
                oskar_dierckx_fprota(co, si, &facc, &coco[j]);
                oskar_dierckx_fprota(co, si, &facs, &cosi[j]);
                if (j == npp) break;
                j1 = j + 1;
                i2 = 1;
                for (l = j1; l <= npp; ++l)
                {
                    ++i2;
                    oskar_dierckx_fprota(co, si, &row[l], &a[j + i2 * a_dim1]);
                }
            }
        }
        oskar_dierckx_fpback(&a[a_offset], &coco[1], npp, npp, &coco[1], ncc);
        oskar_dierckx_fpback(&a[a_offset], &cosi[1], npp, npp, &cosi[1], ncc);
        /*  find ncof, the dimension of the spherical spline and ncoff, the
         *  number of coefficients in the standard b-spline representation. */
        nt4 = *nt - 4;
        np4 = *np - 4;
        ncoff = nt4 * np4;
        ncof = npp * (ntt - 1) + 6;
        /*  find the bandwidth of the observation matrix a. */
        iband = npp << 2;
        if (ntt == 4)
        {
            iband = (npp + 1) * 3;
        }
        else if (ntt < 4)
        {
            iband = ncof;
        }
        iband1 = iband - 1;
        /*  initialize the observation matrix a. */
        for (i = 1; i <= ncof; ++i)
        {
            f[i] = 0.0;
            for (j = 1; j <= iband; ++j)
            {
                a[i + j * a_dim1] = 0.0;
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
            lt = num1 / npp;
            l1 = lt + 4;
            lp = num1 - lt * npp + 1;
            l2 = lp + 3;
            ++lt;
            jrot = (lt > 2) ? (lt - 3) * npp + 3 : 0;
            /*  test whether there are still data points in the
             *  current panel. */
            in = index[num];
            while (in != 0)
            {
                /*  fetch a new data point. */
                wi = w[in];
                ri = r[in] * wi;
                /*  evaluate for the theta-direction, the 4 non-zero b-splines
                 *  at theta(in) */
                oskar_dierckx_fpbspl_d(&tt[1], 3, theta[in], l1, ht);
                /*  evaluate for the phi-direction, the 4 non-zero b-splines
                 *  at phi(in) */
                oskar_dierckx_fpbspl_d(&tp[1], 3, phi[in], l2, hp);
                /*  store the value of these b-splines in spt and spp resp. */
                for (i = 1; i <= 4; ++i)
                {
                    spp[in + i * m] = hp[i - 1];
                    spt[in + i * m] = ht[i - 1];
                }
                /*  initialize the new row of observation matrix. */
                for (i = 1; i <= iband; ++i)
                {
                    h[i] = 0.0;
                }
                /*  calculate the non-zero elements of the new row by making
                 *  the cross products of the non-zero b-splines in theta- and
                 *  phi-direction and by taking into account the conditions of
                 *  the spherical splines. */
                for (i = 1; i <= npp; ++i)
                {
                    row[i] = 0.0;
                }
                /*  take into account the condition (3) of the spherical
                 *  splines. */
                ll = lp;
                for (i = 1; i <= 4; ++i)
                {
                    if (ll > npp) ll = 1;
                    row[ll] += hp[i - 1];
                    ++ll;
                }
                /*  take into account the other conditions of the spherical
                 *  splines. */
                if (!(lt > 2 && lt < ntt - 1))
                {
                    facc = 0.0;
                    facs = 0.0;
                    for (i = 1; i <= npp; ++i)
                    {
                        facc += row[i] * coco[i];
                        facs += row[i] * cosi[i];
                    }
                }
                /*  fill in the non-zero elements of the new row. */
                j1 = 0;
                for (j = 1; j <= 4; ++j)
                {
                    jlt = j + lt;
                    htj = ht[j - 1];
                    if (!(jlt > 2 && jlt <= nt4))
                    {
                        ++j1;
                        h[j1] += htj;
                        continue;
                    }
                    if (!(jlt == 3 || jlt == nt4))
                    {
                        for (i = 1; i <= npp; ++i)
                        {
                            ++j1;
                            h[j1] = row[i] * htj;
                        }
                        continue;
                    }
                    if (jlt != 3)
                    {
                        h[j1 + 1] = facc * htj;
                        h[j1 + 2] = facs * htj;
                        h[j1 + 3] = htj;
                        j1 += 2;
                        continue;
                    }
                    h[1] += htj;
                    h[2] = facc * htj;
                    h[3] = facs * htj;
                    j1 = 3;
                }
                for (i = 1; i <= iband; ++i)
                {
                    h[i] *= wi;
                }
                /*  rotate the row into triangle by givens transformations. */
                irot = jrot;
                for (i = 1; i <= iband; ++i)
                {
                    ++irot;
                    piv = h[i];
                    if (piv == 0.0) continue;
                    /*  calculate the parameters of the givens transformation. */
                    oskar_dierckx_fpgivs(piv, &a[irot + a_dim1], &co, &si);
                    /*  apply that transformation to the right hand side. */
                    oskar_dierckx_fprota(co, si, &ri, &f[irot]);
                    if (i == iband) break;
                    /*  apply that transformation to the left hand side. */
                    i2 = 1;
                    i3 = i + 1;
                    for (j = i3; j <= iband; ++j)
                    {
                        ++i2;
                        oskar_dierckx_fprota(co, si, &h[j],
                                &a[irot + i2 * a_dim1]);
                    }
                }
                /*  add the contribution of the row to the sum of squares of
                 *  residual right hand sides. */
                *fp += ri * ri;
                /*  find the number of the next data point in the panel. */
                in = nummer[in];
            }
        }
        /*  find dmax, the maximum value for the diagonal elements in the
         *  reduced triangle. */
        dmax = 0.0;
        for (i = 1; i <= ncof; ++i)
        {
            if (a[i + a_dim1] <= dmax) continue;
            dmax = a[i + a_dim1];
        }
        /*  check whether the observation matrix is rank deficient. */
        sigma = eps * dmax;
        for (i = 1; i <= ncof; ++i)
        {
            if (a[i + a_dim1] <= sigma)
            {
                i = -1;
                break;
            }
        }
        if (i != -1)
        {
            /*  backward substitution in case of full rank. */
            oskar_dierckx_fpback(&a[a_offset], &f[1], ncof, iband, &c[1], ncc);
            rank = ncof;
            for (i = 1; i <= ncof; ++i)
            {
                q[i + q_dim1] = a[i + a_dim1] / dmax;
            }
        }
        else
        {
            /*  in case of rank deficiency, find the minimum norm solution. */
            lwest = ncof * iband + ncof + iband;
            if (lwrk < lwest)
            {
                *ier = lwest;
                return;
            }
            lf = 1;
            lh = lf + ncof;
            la = lh + iband;
            for (i = 1; i <= ncof; ++i)
            {
                ff[i] = f[i];
                for (j = 1; j <= iband; ++j)
                {
                    q[i + j * q_dim1] = a[i + j * a_dim1];
                }
            }
            oskar_dierckx_fprank(&q[q_offset], &ff[1], ncof, iband, ncc,
                    sigma, &c[1], &sq, &rank, &wrk[la], &wrk[lf], &wrk[lh]);
            for (i = 1; i <= ncof; ++i)
            {
                q[i + q_dim1] /= dmax;
            }
            /*  add to the sum of squared residuals, the contribution of
             *  reducing the rank. */
            *fp += sq;
        }
        /*  find the coefficients in the standard b-spline representation of
         *  the spherical spline. */
        oskar_dierckx_fprpsp(*nt, *np, &coco[1], &cosi[1], &c[1], &ff[1],
                ncoff);
        /*  test whether the least-squares spline is an acceptable solution. */
        if (iopt < 0) {
            if (*fp <= 0.0) {
                goto L970;
            } else {
                goto L980;
            }
        }
        fpms = *fp - s;
        if (fabs(fpms) <= acc) {
            if (*fp <= 0.0) {
                goto L970;
            } else {
                goto L980;
            }
        }
        /*  if f(p=inf) < s, accept the choice of knots. */
        if (fpms < 0.0) break; /* Go to part 2. */
        /*  test whether we cannot further increase the number of knots. */
        if (ncof > m)
        {
            *ier = 4;
            return;
        }
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
            lt = num1 / npp;
            l1 = lt + 1;
            lp = num1 - lt * npp;
            l2 = lp + 1 + ntt;
            jrot = lt * np4 + lp;
            in = index[num];
            while (in != 0)
            {
                store = 0.0;
                i1 = jrot;
                for (i = 1; i <= 4; ++i)
                {
                    hti = spt[in + i * m];
                    j1 = i1;
                    for (j = 1; j <= 4; ++j)
                    {
                        ++j1;
                        store += hti * spp[in + j * m] * c[j1];
                    }
                    i1 += np4;
                }
                r1 = w[in] * (r[in] - store);
                store = r1 * r1;
                fpint[l1] += store;
                coord[l1] += store * theta[in];
                fpint[l2] += store;
                coord[l2] += store * phi[in];
                in = nummer[in];
            }
        }
        /*  find the interval for which fpint is maximal on the condition that
         *  there still can be added a knot. */
        l1 = 1;
        l2 = nrint;
        if (ntest < *nt + 1) l1 = ntt + 1;
        if (npest < *np + 2) l2 = ntt;
        /*  test whether we cannot further increase the number of knots. */
        if (l1 > l2)
        {
            *ier = 1;
            return;
        }
        for (;;)
        {
            fpmax = 0.0;
            l = 0;
            for (i = l1; i <= l2; ++i)
            {
                if (fpmax >= fpint[i]) continue;
                l = i;
                fpmax = fpint[i];
            }
            if (l == 0)
            {
                *ier = 5;
                return;
            }
            /*  calculate the position of the new knot. */
            arg = coord[l] / fpint[l];
            /*  test in what direction the new knot is going to be added. */
            if (l > ntt)
            {
                /*  addition in the phi-direction */
                l4 = l + 4 - ntt;
                if (!(arg < pi))
                {
                    arg -= pi;
                    l4 -= nrr;
                }
                fpint[l] = 0.0;
                fac1 = tp[l4] - arg;
                fac2 = arg - tp[l4 - 1];
                if (fac1 > 10.0 * fac2 || fac2 > 10.0 * fac1) continue;
                ll = nrr + 4;
                j = ll;
                for (i = l4; i <= ll; ++i)
                {
                    tp[j + 1] = tp[j];
                    --j;
                }
                tp[l4] = arg;
                *np += 2;
                ++nrr;
                for (i = 5; i <= ll; ++i)
                {
                    j = i + nrr;
                    tp[j] = tp[i] + pi;
                }
            }
            else
            {
                /*  addition in the theta-direction */
                l4 = l + 4;
                fpint[l] = 0.0;
                fac1 = tt[l4] - arg;
                fac2 = arg - tt[l4 - 1];
                if (fac1 > 10.0 * fac2 || fac2 > 10.0 * fac1) continue;
                j = *nt;
                for (i = l4; i <= *nt; ++i)
                {
                    tt[j + 1] = tt[j];
                    --j;
                }
                tt[l4] = arg;
                ++(*nt);
            }
            break;
        }
        /*  restart the computations with the new set of knots. */
    }

    /*
     * part 2: determination of the smoothing spherical spline.
     * ********************************************************
     * we have determined the number of knots and their position. we now
     * compute the coefficients of the smoothing spline sp(theta,phi).
     * the observation matrix a is extended by the rows of a matrix, expres-
     * sing that sp(theta,phi) must be a constant function in the variable
     * phi and a cubic polynomial in the variable theta. the corresponding
     * weights of these additional rows are set to 1/(p). iteratively
     * we than have to determine the value of p such that f(p) = sum((w(i)*
     * (r(i)-sp(theta(i),phi(i))))**2)  be = s.
     * we already know that the least-squares polynomial corresponds to p=0,
     * and that the least-squares spherical spline corresponds to p=infin.
     * the iteration process makes use of rational interpolation. since f(p)
     * is a convex and strictly decreasing function of p, it can be
     * approximated by a rational function of the form r(p) = (u*p+v)/(p+w).
     * three values of p (p1,p2,p3) with corresponding values of f(p) (f1=
     * f(p1)-s,f2=f(p2)-s,f3=f(p3)-s) are used to calculate the new value
     * of p such that r(p)=s. convergence is guaranteed by taking f1>0,f3<0.
     */

    /*  evaluate the discontinuity jumps of the 3rd order derivative of
     *  the b-splines at the knots tt(l),l=5,...,nt-4. */
    oskar_dierckx_fpdisc(&tt[1], *nt, 5, &bt[bt_offset], ntest);
    /*  evaluate the discontinuity jumps of the 3rd order derivative of
     *  the b-splines at the knots tp(l),l=5,...,np-4. */
    oskar_dierckx_fpdisc(&tp[1], *np, 5, &bp[bp_offset], npest);
    /*  initial value for p. */
    p1 = 0.0;
    f1 = *sup - s;
    p3 = -1.0;
    f3 = fpms;
    p = 0.0;
    for (i = 1; i <= ncof; ++i)
    {
        p += a[i + a_dim1];
    }
    rn = (double) ncof;
    p = rn / p;
    /*  find the bandwidth of the extended observation matrix. */
    iband4 = (ntt <= 4) ? ncof : iband + 3;
    iband3 = iband4 - 1;
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
            for (j = 1; j <= iband4; ++j)
            {
                q[i + j * q_dim1] = 0.0;
            }
            for (j = 1; j <= iband; ++j)
            {
                q[i + j * q_dim1] = a[i + j * a_dim1];
            }
        }
        /*  extend the observation matrix with the rows of a matrix, expressing
         *  that for theta=cst. sp(theta,phi) must be a constant function. */
        nt6 = *nt - 6;
        for (i = 5; i <= np4; ++i)
        {
            ii = i - 4;
            for (l = 1; l <= npp; ++l)
            {
                row[l] = 0.0;
            }
            ll = ii;
            for (l = 1; l <= 5; ++l)
            {
                if (ll > npp) ll = 1;
                row[ll] += bp[ii + l * bp_dim1];
                ++ll;
            }
            facc = 0.0;
            facs = 0.0;
            for (l = 1; l <= npp; ++l)
            {
                facc += row[l] * coco[l];
                facs += row[l] * cosi[l];
            }
            for (j = 1; j <= nt6; ++j)
            {
                /*  initialize the new row. */
                for (l = 1; l <= iband; ++l)
                {
                    h[l] = 0.0;
                }
                /*  fill in the non-zero elements of the row. jrot records the
                 *  column number of the first non-zero element in the row. */
                jrot = (j - 2) * npp + 4;
                if (j > 1 && j < nt6)
                {
                    for (l = 1; l <= npp; ++l)
                    {
                        h[l] = row[l];
                    }
                }
                else
                {
                    h[1] = facc;
                    h[2] = facs;
                    if (j == 1) jrot = 2;
                }
                for (l = 1; l <= iband; ++l)
                {
                    h[l] *= pinv;
                }
                ri = 0.0;
                /*  rotate the new row into triangle by givens transformations. */
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
                        oskar_dierckx_fpgivs(piv, &q[irot + q_dim1], &co, &si);
                        /*  apply givens transformation to right hand side. */
                        oskar_dierckx_fprota(co, si, &ri, &ff[irot]);
                        if (i2 == 0) break;
                        /*  apply givens transformation to left hand side. */
                        for (l = 1; l <= i2; ++l)
                        {
                            l1 = l + 1;
                            oskar_dierckx_fprota(co, si, &h[l1],
                                    &q[irot + l1 * q_dim1]);
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
        /*  extend the observation matrix with the rows of a matrix expressing
         *  that for phi=cst. sp(theta,phi) must be a cubic polynomial. */
        for (i = 5; i <= nt4; ++i)
        {
            ii = i - 4;
            for (j = 1; j <= npp; ++j)
            {
                /*  initialize the new row */
                for (l = 1; l <= iband4; ++l)
                {
                    h[l] = 0.0;
                }
                /*  fill in the non-zero elements of the row. jrot records the
                 *  column number of the first non-zero element in the row. */
                j1 = 1;
                for (l = 1; l <= 5; ++l)
                {
                    il = ii + l;
                    ij = npp;
                    if (il == 3 || il == nt4)
                    {
                        j1 = j1 + 3 - j;
                        j2 = j1 - 2;
                        ij = 0;
                        if (il == 3)
                        {
                            j1 = 1;
                            j2 = 2;
                            ij = j + 2;
                        }
                        h[j2] = bt[ii + l * bt_dim1] * coco[j];
                        h[j2 + 1] = bt[ii + l * bt_dim1] * cosi[j];
                    }
                    h[j1] += bt[ii + l * bt_dim1];
                    j1 += ij;
                }
                for (l = 1; l <= iband4; ++l)
                {
                    h[l] *= pinv;
                }
                ri = 0.0;
                jrot = 1;
                if (ii > 2) jrot = j + 3 + (ii - 3) * npp;
                /*  rotate the new row into triangle by givens transformations. */
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
                        oskar_dierckx_fpgivs(piv, &q[irot + q_dim1], &co, &si);
                        /*  apply givens transformation to right hand side. */
                        oskar_dierckx_fprota(co, si, &ri, &ff[irot]);
                        if (i2 == 0) break;
                        /*  apply givens transformation to left hand side. */
                        for (l = 1; l <= i2; ++l)
                        {
                            l1 = l + 1;
                            oskar_dierckx_fprota(co, si, &h[l1],
                                    &q[irot + l1 * q_dim1]);
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
        /*  find dmax, the maximum value for the diagonal elements in the
         *  reduced triangle. */
        dmax = 0.0;
        for (i = 1; i <= ncof; ++i)
        {
            if (q[i + q_dim1] <= dmax) continue;
            dmax = q[i + q_dim1];
        }
        /*  check whether the matrix is rank deficient. */
        sigma = eps * dmax;
        for (i = 1; i <= ncof; ++i)
        {
            if (q[i + q_dim1] <= sigma)
            {
                i = -1;
                break;
            }
        }
        if (i != -1)
        {
            /*  backward substitution in case of full rank. */
            oskar_dierckx_fpback(&q[q_offset], &ff[1], ncof, iband4,
                    &c[1], ncc);
            rank = ncof;
        }
        else
        {
            /*  in case of rank deficiency, find the minimum norm solution. */
            lwest = ncof * iband4 + ncof + iband4;
            if (lwrk < lwest)
            {
                *ier = lwest;
                return;
            }
            lf = 1;
            lh = lf + ncof;
            la = lh + iband4;
            oskar_dierckx_fprank(&q[q_offset], &ff[1], ncof, iband4, ncc,
                    sigma, &c[1], &sq, &rank, &wrk[la], &wrk[lf], &wrk[lh]);
        }
        for (i = 1; i <= ncof; ++i)
        {
            q[i + q_dim1] /= dmax;
        }
        /*  find the coefficients in the standard b-spline representation of
         *  the spherical spline. */
        oskar_dierckx_fprpsp(*nt, *np, &coco[1], &cosi[1], &c[1], &ff[1],
                ncoff);
        /*  compute f(p). */
        *fp = 0.0;
        for (num = 1; num <= nreg; ++num)
        {
            num1 = num - 1;
            lt = num1 / npp;
            lp = num1 - lt * npp;
            jrot = lt * np4 + lp;
            in = index[num];
            while (in != 0)
            {
                store = 0.0;
                i1 = jrot;
                for (i = 1; i <= 4; ++i)
                {
                    hti = spt[in + i * m];
                    j1 = i1;
                    for (j = 1; j <= 4; ++j)
                    {
                        ++j1;
                        store += hti * spp[in + j * m] * c[j1];
                    }
                    i1 += np4;
                }
                r1 = w[in] * (r[in] - store);
                *fp += r1 * r1;
                in = nummer[in];
            }
        }
        /*  test whether the approximation sp(theta,phi) is an acceptable
         *  solution */
        fpms = *fp - s;
        if (fabs(fpms) <= acc) {
            goto L980;
        }
        /*  test whether the maximum allowable number of iterations has been
         *  reached. */
        if (iter == maxit)
        {
            *ier = 3;
            return;
        }
        /*  carry out one more step of the iteration process. */
        p2 = p;
        f2 = fpms;
        if (ich3 == 0)
        {
            if (! (f2 - f3 > acc))
            {
                /*  our initial choice of p is too large. */
                p3 = p2;
                f3 = f2;
                p *= 0.04;
                if (p <= p1) p = p1 * 0.9 + p2 * 0.1;
                continue;
            }
            if (f2 < 0.0) ich3 = 1;
        }
        if (ich1 == 0)
        {
            if (! (f1 - f2 > acc))
            {
                /*  our initial choice of p is too small */
                p1 = p2;
                f1 = f2;
                p /= 0.04;
                if (p3 < 0.0) continue;
                if (p >= p3) p = p2 * 0.1 + p3 * 0.9;
                continue;
            }
            if (f2 > 0.0) ich1 = 1;
        }

        /*  test whether the iteration process proceeds as theoretically
         *  expected. */
        if (f2 >= f1 || f2 <= f3)
        {
            *ier = 2;
            return;
        }
        /*  find the new value of p. */
        p = oskar_dierckx_fprati(&p1, &f1, p2, f2, &p3, &f3);
    }
L970:
    *ier = -1;
    *fp = 0.0;
L980:
    if (ncof != rank) *ier = -rank;
}

#ifdef __cplusplus
}
#endif
