#include "splines/oskar_dierckx_fprank.h"
#include "splines/oskar_dierckx_fprota.h"
#include "splines/oskar_dierckx_fpgivs.h"

#define min(a,b) ((a) <= (b) ? (a) : (b))

#ifdef __cplusplus
extern "C" {
#endif

void oskar_dierckx_fprank(double *a, double *f, int n, int m, int na,
        double tol, double *c, double *sq, int *rank, double *aa,
        double *ff, double *h)
{
    /* System generated locals */
    int a_dim1 = 0, aa_dim1 = 0, t = 0;

    /* Local variables */
    int i = 0, j = 0, k = 0, i1 = 0, i2 = 0, j1 = 0, j2 = 0, j3 = 0;
    int m1 = 0, ii = 0, ij = 0, jj = 0, kk = 0, nl = 0;
    double yi = 0.0, fac = 0.0, cos_ = 0.0, sin_ = 0.0, piv = 0.0;
    double stor1 = 0.0, stor2 = 0.0, stor3 = 0.0, store = 0.0;

    /* Parameter adjustments */
    --ff;
    --c;
    --f;
    --h;
    aa_dim1 = n;
    aa -= (1 + aa_dim1);
    a_dim1 = na;
    a -= (1 + a_dim1);

    /* Function Body */
    m1 = m - 1;
    /*  the rank deficiency nl is considered to be the number of sufficient
     *  small diagonal elements of a. */
    nl = 0;
    *sq = 0.0;
    for (i = 1; i <= n; ++i)
    {
        if (a[i + a_dim1] > tol) continue;
        /*  if a sufficient small diagonal element is found, we put it to
         *  zero. the remainder of the row corresponding to that zero diagonal
         *  element is then rotated into triangle by givens rotations.
         *  the rank deficiency is increased by one. */
        ++nl;
        if (i == n) continue;
        yi = f[i];
        for (j = 1; j <= m1; ++j)
        {
            h[j] = a[i + (j + 1) * a_dim1];
        }
        h[m] = 0.0;
        i1 = i + 1;
        for (ii = i1; ii <= n; ++ii)
        {
            /* Computing MIN */
            t = n - ii;
            i2 = min(t,m1);
            piv = h[1];
            if (piv == 0.0)
            {
                if (i2 == 0) break;
                for (j = 1; j <= i2; ++j)
                {
                    h[j] = h[j + 1];
                }
            }
            else
            {
                oskar_dierckx_fpgivs(piv, &a[ii + a_dim1], &cos_, &sin_);
                oskar_dierckx_fprota(cos_, sin_, &yi, &f[ii]);
                if (i2 == 0) break;
                for (j = 1; j <= i2; ++j)
                {
                    j1 = j + 1;
                    oskar_dierckx_fprota(cos_, sin_, &h[j1], &a[ii + j1 * a_dim1]);
                    h[j] = h[j1];
                }
            }
            h[i2 + 1] = 0.0;
        }
        /*  add to the sum of squared residuals the contribution of deleting
         *  the row with small diagonal element. */
        *sq += yi * yi;
    }
    /*  rank denotes the rank of a. */
    *rank = n - nl;
    /*  let b denote the (rank*n) upper trapezoidal matrix which can be
     *  obtained from the (n*n) upper triangular matrix a by deleting
     *  the rows and interchanging the columns corresponding to a zero
     *  diagonal element. if this matrix is factorized using givens
     *  transformations as  b = (r) (u)  where
     *    r is a (rank*rank) upper triangular matrix,
     *    u is a (rank*n) orthonormal matrix
     *  then the minimal least-squares solution c is given by c = b' v,
     *  where v is the solution of the system  (r) (r)' v = g  and
     *  g denotes the vector obtained from the old right hand side f, by
     *  removing the elements corresponding to a zero diagonal element of a. */
    /*  initialization. */
    for (i = 1; i <= *rank; ++i)
    {
        for (j = 1; j <= m; ++j)
        {
            aa[i + j * aa_dim1] = 0.0;
        }
    }
    /*  form in aa the upper triangular matrix obtained from a by
     *  removing rows and columns with zero diagonal elements. form in ff
     *  the new right hand side by removing the elements of the old right
     *  hand side corresponding to a deleted row. */
    ii = 0;
    for (i = 1; i <= n; ++i)
    {
        if (a[i + a_dim1] <= tol) continue;
        ++ii;
        ff[ii] = f[i];
        aa[ii + aa_dim1] = a[i + a_dim1];
        jj = ii;
        kk = 1;
        j = i;
        /* Computing MIN */
        t = j - 1;
        j1 = min(t,m1);
        if (j1 == 0) continue;
        for (k = 1; k <= j1; ++k)
        {
            --j;
            if (a[j + a_dim1] <= tol) continue;
            ++kk;
            --jj;
            aa[jj + kk * aa_dim1] = a[j + (k + 1) * a_dim1];
        }
    }
    /*  form successively in h the columns of a with a zero diagonal element. */
    ii = 0;
    for (i = 1; i <= n; ++i)
    {
        ++ii;
        if (a[i + a_dim1] > tol) continue;
        --ii;
        if (ii == 0) continue;
        jj = 1;
        j = i;
        /* Computing MIN */
        t = j - 1;
        j1 = min(t,m1);
        for (k = 1; k <= j1; ++k)
        {
            --j;
            if (a[j + a_dim1] <= tol) continue;
            h[jj] = a[j + (k + 1) * a_dim1];
            ++jj;
        }
        for (kk = jj; kk <= m; ++kk)
        {
            h[kk] = 0.0;
        }
        /*  rotate this column into aa by givens transformations. */
        jj = ii;
        for (i1 = 1; i1 <= ii; ++i1)
        {
            /* Computing MIN */
            t = jj - 1;
            j1 = min(t,m1);
            piv = h[1];
            if (piv != 0.0)
            {
                oskar_dierckx_fpgivs(piv, &aa[jj + aa_dim1], &cos_, &sin_);
                if (j1 == 0) break;
                kk = jj;
                for (j2 = 1; j2 <= j1; ++j2)
                {
                    j3 = j2 + 1;
                    --kk;
                    oskar_dierckx_fprota(cos_, sin_, &h[j3],
                            &aa[kk + j3 * aa_dim1]);
                    h[j2] = h[j3];
                }
            }
            else
            {
                if (j1 == 0) break;
                for (j2 = 1; j2 <= j1; ++j2)
                {
                    j3 = j2 + 1;
                    h[j2] = h[j3];
                }
            }
            --jj;
            h[j3] = 0.0;
        }
    }
    /*  solve the system (aa) (f1) = ff */
    ff[*rank] /= aa[*rank + aa_dim1];
    i = *rank - 1;
    if (i != 0)
    {
        for (j = 2; j <= *rank; ++j)
        {
            store = ff[i];
            /* Computing MIN */
            t = j - 1;
            i1 = min(t,m1);
            k = i;
            for (ii = 1; ii <= i1; ++ii)
            {
                ++k;
                stor1 = ff[k];
                stor2 = aa[i + (ii + 1) * aa_dim1];
                store -= stor1 * stor2;
            }
            stor1 = aa[i + aa_dim1];
            ff[i] = store / stor1;
            --i;
        }
    }
    /*  solve the system  (aa)' (f2) = f1 */
    ff[1] /= aa[aa_dim1 + 1];
    if (*rank != 1)
    {
        for (j = 2; j <= *rank; ++j)
        {
            store = ff[j];
            /* Computing MIN */
            t = j - 1;
            i1 = min(t,m1);
            k = j;
            for (ii = 1; ii <= i1; ++ii)
            {
                --k;
                stor1 = ff[k];
                stor2 = aa[k + (ii + 1) * aa_dim1];
                store -= stor1 * stor2;
            }
            stor1 = aa[j + aa_dim1];
            ff[j] = store / stor1;
        }
    }
    /*  premultiply f2 by the transpose of a. */
    k = 0;
    for (i = 1; i <= n; ++i)
    {
        store = 0.0;
        if (a[i + a_dim1] > tol) ++k;
        j1 = min(i,m);
        kk = k;
        ij = i + 1;
        for (j = 1; j <= j1; ++j)
        {
            --ij;
            if (a[ij + a_dim1] <= tol) continue;
            stor1 = a[ij + j * a_dim1];
            stor2 = ff[kk];
            store += stor1 * stor2;
            --kk;
        }
        c[i] = store;
    }
    /*  add to the sum of squared residuals the contribution of putting
     *  to zero the small diagonal elements of matrix (a). */
    stor3 = 0.0;
    for (i = 1; i <= n; ++i)
    {
        if (a[i + a_dim1] > tol) continue;
        store = f[i];
        /* Computing MIN */
        t = n - i;
        i1 = min(t,m1);
        if (i1 != 0)
        {
            for (j = 1; j <= i1; ++j)
            {
                ij = i + j;
                stor1 = c[ij];
                stor2 = a[i + (j + 1) * a_dim1];
                store -= stor1 * stor2;
            }
        }
        stor1 = a[i + a_dim1];
        stor2 = c[i];
        stor1 *= stor2;
        stor3 += stor1 * (stor1 - store - store);
    }
    fac = stor3;
    *sq += fac;
}

#ifdef __cplusplus
}
#endif
