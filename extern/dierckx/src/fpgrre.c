/* fpgrre.f -- translated by f2c (version 20090411).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

extern int fpback_(real *, real *, integer *, integer *, 
    real *, integer *);

extern int fpdisc_(real *, integer *, integer *, real *, 
    integer *), fpbspl_(real *, integer *, integer *, real *, integer 
    *, real *);

extern int fprota_(real *, real *, real *, real *), 
    fpgivs_(real *, real *, real *, real *);




int fpgrre_(integer *ifsx, integer *ifsy, integer *ifbx, 
	integer *ifby, real *x, integer *mx, real *y, integer *my, real *z__, 
	integer *mz, integer *kx, integer *ky, real *tx, integer *nx, real *
	ty, integer *ny, real *p, real *c__, integer *nc, real *fp, real *fpx,
	 real *fpy, integer *mm, integer *mynx, integer *kx1, integer *kx2, 
	integer *ky1, integer *ky2, real *spx, real *spy, real *right, real *
	q, real *ax, real *ay, real *bx, real *by, integer *nrx, integer *nry)
{
    /* System generated locals */
    integer spx_dim1, spx_offset, spy_dim1, spy_offset, ax_dim1, ax_offset, 
	    bx_dim1, bx_offset, ay_dim1, ay_offset, by_dim1, by_offset, i__1, 
	    i__2, i__3, i__4;
    real r__1;

    /* Local variables */
    real h__[7];
    integer i__, j, k, l, i1, i2, i3, k1, k2, l1, l2, n1, ic, iq, it, iz;
    real fac, arg, cos__, sin__, piv;
    integer nk1x, nk1y;
    real half;
    integer ncof;
    real term, pinv;
    integer irot, numx, numy, numx1, numy1, nrold;
    integer ibandx, ibandy;
    integer number;
    integer nroldx, nroldy;

/*  the b-spline coefficients of the smoothing spline are calculated as
 *  the least-squares solution of the over-determined linear system of
 *  equations  (ay) c (ax)' = q       where
 *
 *               |   (spx)    |            |   (spy)    |
 *        (ax) = | ---------- |     (ay) = | ---------- |
 *               | (1/p) (bx) |            | (1/p) (by) |
 *
 *                                | z  ' 0 |
 *                            q = | ------ |
 *                                | 0  ' 0 |
 *
 *  with c      : the (ny-ky-1) x (nx-kx-1) matrix which contains the
 *                b-spline coefficients.
 *       z      : the my x mx matrix which contains the function values.
 *       spx,spy: the mx x (nx-kx-1) and  my x (ny-ky-1) observation
 *                matrices according to the least-squares problems in
 *                the x- and y-direction.
 *       bx,by  : the (nx-2*kx-1) x (nx-kx-1) and (ny-2*ky-1) x (ny-ky-1)
 *                matrices which contain the discontinuity jumps of the
 *                derivatives of the b-splines in the x- and y-direction.
 *  ..subroutine references..
 *    fpback,fpbspl,fpgivs,fpdisc,fprota
 */

    /* Parameter adjustments */
    --nrx;
    --x;
    --nry;
    --y;
    --z__;
    --fpx;
    --tx;
    --fpy;
    --ty;
    --c__;
    --right;
    --q;
    spx_dim1 = *mx;
    spx_offset = 1 + spx_dim1;
    spx -= spx_offset;
    bx_dim1 = *nx;
    bx_offset = 1 + bx_dim1;
    bx -= bx_offset;
    ax_dim1 = *nx;
    ax_offset = 1 + ax_dim1;
    ax -= ax_offset;
    spy_dim1 = *my;
    spy_offset = 1 + spy_dim1;
    spy -= spy_offset;
    by_dim1 = *ny;
    by_offset = 1 + by_dim1;
    by -= by_offset;
    ay_dim1 = *ny;
    ay_offset = 1 + ay_dim1;
    ay -= ay_offset;

    /* Function Body */
    half = .5f;
    nk1x = *nx - *kx1;
    nk1y = *ny - *ky1;
    if (*p > 0.f) {
	pinv = 1.f / *p;
    }
/*  it depends on the value of the flags ifsx,ifsy,ifbx and ifby and on */
/*  the value of p whether the matrices (spx),(spy),(bx) and (by) still */
/*  must be determined. */
    if (*ifsx != 0) {
	goto L50;
    }
/*  calculate the non-zero elements of the matrix (spx) which is the */
/*  observation matrix according to the least-squares spline approximat- */
/*  ion problem in the x-direction. */
    l = *kx1;
    l1 = *kx2;
    number = 0;
    i__1 = *mx;
    for (it = 1; it <= i__1; ++it) {
	arg = x[it];
L10:
	if (arg < tx[l1] || l == nk1x) {
	    goto L20;
	}
	l = l1;
	l1 = l + 1;
	++number;
	goto L10;
L20:
	fpbspl_(&tx[1], nx, kx, &arg, &l, h__);
	i__2 = *kx1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    spx[it + i__ * spx_dim1] = h__[i__ - 1];
/* L30: */
	}
	nrx[it] = number;
/* L40: */
    }
    *ifsx = 1;
L50:
    if (*ifsy != 0) {
	goto L100;
    }
/*  calculate the non-zero elements of the matrix (spy) which is the */
/*  observation matrix according to the least-squares spline approximat- */
/*  ion problem in the y-direction. */
    l = *ky1;
    l1 = *ky2;
    number = 0;
    i__1 = *my;
    for (it = 1; it <= i__1; ++it) {
	arg = y[it];
L60:
	if (arg < ty[l1] || l == nk1y) {
	    goto L70;
	}
	l = l1;
	l1 = l + 1;
	++number;
	goto L60;
L70:
	fpbspl_(&ty[1], ny, ky, &arg, &l, h__);
	i__2 = *ky1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    spy[it + i__ * spy_dim1] = h__[i__ - 1];
/* L80: */
	}
	nry[it] = number;
/* L90: */
    }
    *ifsy = 1;
L100:
    if (*p <= 0.f) {
	goto L120;
    }
/*  calculate the non-zero elements of the matrix (bx). */
    if (*ifbx != 0 || *nx == *kx1 << 1) {
	goto L110;
    }
    fpdisc_(&tx[1], nx, kx2, &bx[bx_offset], nx);
    *ifbx = 1;
/*  calculate the non-zero elements of the matrix (by). */
L110:
    if (*ifby != 0 || *ny == *ky1 << 1) {
	goto L120;
    }
    fpdisc_(&ty[1], ny, ky2, &by[by_offset], ny);
    *ifby = 1;
/*  reduce the matrix (ax) to upper triangular form (rx) using givens */
/*  rotations. apply the same transformations to the rows of matrix q */
/*  to obtain the my x (nx-kx-1) matrix g. */
/*  store matrix (rx) into (ax) and g into q. */
L120:
    l = *my * nk1x;
/*  initialization. */
    i__1 = l;
    for (i__ = 1; i__ <= i__1; ++i__) {
	q[i__] = 0.f;
/* L130: */
    }
    i__1 = nk1x;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *kx2;
	for (j = 1; j <= i__2; ++j) {
	    ax[i__ + j * ax_dim1] = 0.f;
/* L140: */
	}
    }
    l = 0;
    nrold = 0;
/*  ibandx denotes the bandwidth of the matrices (ax) and (rx). */
    ibandx = *kx1;
    i__2 = *mx;
    for (it = 1; it <= i__2; ++it) {
	number = nrx[it];
L150:
	if (nrold == number) {
	    goto L180;
	}
	if (*p <= 0.f) {
	    goto L260;
	}
	ibandx = *kx2;
/*  fetch a new row of matrix (bx). */
	n1 = nrold + 1;
	i__1 = *kx2;
	for (j = 1; j <= i__1; ++j) {
	    h__[j - 1] = bx[n1 + j * bx_dim1] * pinv;
/* L160: */
	}
/*  find the appropriate column of q. */
	i__1 = *my;
	for (j = 1; j <= i__1; ++j) {
	    right[j] = 0.f;
/* L170: */
	}
	irot = nrold;
	goto L210;
/*  fetch a new row of matrix (spx). */
L180:
	h__[ibandx - 1] = 0.f;
	i__1 = *kx1;
	for (j = 1; j <= i__1; ++j) {
	    h__[j - 1] = spx[it + j * spx_dim1];
/* L190: */
	}
/*  find the appropriate column of q. */
	i__1 = *my;
	for (j = 1; j <= i__1; ++j) {
	    ++l;
	    right[j] = z__[l];
/* L200: */
	}
	irot = number;
/*  rotate the new row of matrix (ax) into triangle. */
L210:
	i__1 = ibandx;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    ++irot;
	    piv = h__[i__ - 1];
	    if (piv == 0.f) {
		goto L240;
	    }
/*  calculate the parameters of the givens transformation. */
	    fpgivs_(&piv, &ax[irot + ax_dim1], &cos__, &sin__);
/*  apply that transformation to the rows of matrix q. */
	    iq = (irot - 1) * *my;
	    i__3 = *my;
	    for (j = 1; j <= i__3; ++j) {
		++iq;
		fprota_(&cos__, &sin__, &right[j], &q[iq]);
/* L220: */
	    }
/*  apply that transformation to the columns of (ax). */
	    if (i__ == ibandx) {
		goto L250;
	    }
	    i2 = 1;
	    i3 = i__ + 1;
	    i__3 = ibandx;
	    for (j = i3; j <= i__3; ++j) {
		++i2;
		fprota_(&cos__, &sin__, &h__[j - 1], &ax[irot + i2 * ax_dim1])
			;
/* L230: */
	    }
L240:
	    ;
	}
L250:
	if (nrold == number) {
	    goto L270;
	}
L260:
	++nrold;
	goto L150;
L270:
	;
    }
/*  reduce the matrix (ay) to upper triangular form (ry) using givens */
/*  rotations. apply the same transformations to the columns of matrix g */
/*  to obtain the (ny-ky-1) x (nx-kx-1) matrix h. */
/*  store matrix (ry) into (ay) and h into c. */
    ncof = nk1x * nk1y;
/*  initialization. */
    i__2 = ncof;
    for (i__ = 1; i__ <= i__2; ++i__) {
	c__[i__] = 0.f;
/* L280: */
    }
    i__2 = nk1y;
    for (i__ = 1; i__ <= i__2; ++i__) {
	i__1 = *ky2;
	for (j = 1; j <= i__1; ++j) {
	    ay[i__ + j * ay_dim1] = 0.f;
/* L290: */
	}
    }
    nrold = 0;
/*  ibandy denotes the bandwidth of the matrices (ay) and (ry). */
    ibandy = *ky1;
    i__1 = *my;
    for (it = 1; it <= i__1; ++it) {
	number = nry[it];
L300:
	if (nrold == number) {
	    goto L330;
	}
	if (*p <= 0.f) {
	    goto L410;
	}
	ibandy = *ky2;
/*  fetch a new row of matrix (by). */
	n1 = nrold + 1;
	i__2 = *ky2;
	for (j = 1; j <= i__2; ++j) {
	    h__[j - 1] = by[n1 + j * by_dim1] * pinv;
/* L310: */
	}
/*  find the appropiate row of g. */
	i__2 = nk1x;
	for (j = 1; j <= i__2; ++j) {
	    right[j] = 0.f;
/* L320: */
	}
	irot = nrold;
	goto L360;
/*  fetch a new row of matrix (spy) */
L330:
	h__[ibandy - 1] = 0.f;
	i__2 = *ky1;
	for (j = 1; j <= i__2; ++j) {
	    h__[j - 1] = spy[it + j * spy_dim1];
/* L340: */
	}
/*  find the appropiate row of g. */
	l = it;
	i__2 = nk1x;
	for (j = 1; j <= i__2; ++j) {
	    right[j] = q[l];
	    l += *my;
/* L350: */
	}
	irot = number;
/*  rotate the new row of matrix (ay) into triangle. */
L360:
	i__2 = ibandy;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    ++irot;
	    piv = h__[i__ - 1];
	    if (piv == 0.f) {
		goto L390;
	    }
/*  calculate the parameters of the givens transformation. */
	    fpgivs_(&piv, &ay[irot + ay_dim1], &cos__, &sin__);
/*  apply that transformation to the colums of matrix g. */
	    ic = irot;
	    i__3 = nk1x;
	    for (j = 1; j <= i__3; ++j) {
		fprota_(&cos__, &sin__, &right[j], &c__[ic]);
		ic += nk1y;
/* L370: */
	    }
/*  apply that transformation to the columns of matrix (ay). */
	    if (i__ == ibandy) {
		goto L400;
	    }
	    i2 = 1;
	    i3 = i__ + 1;
	    i__3 = ibandy;
	    for (j = i3; j <= i__3; ++j) {
		++i2;
		fprota_(&cos__, &sin__, &h__[j - 1], &ay[irot + i2 * ay_dim1])
			;
/* L380: */
	    }
L390:
	    ;
	}
L400:
	if (nrold == number) {
	    goto L420;
	}
L410:
	++nrold;
	goto L300;
L420:
	;
    }
/*  backward substitution to obtain the b-spline coefficients as the */
/*  solution of the linear system    (ry) c (rx)' = h. */
/*  first step: solve the system  (ry) (c1) = h. */
    k = 1;
    i__1 = nk1x;
    for (i__ = 1; i__ <= i__1; ++i__) {
	fpback_(&ay[ay_offset], &c__[k], &nk1y, &ibandy, &c__[k], ny);
	k += nk1y;
/* L450: */
    }
/*  second step: solve the system  c (rx)' = (c1). */
    k = 0;
    i__1 = nk1y;
    for (j = 1; j <= i__1; ++j) {
	++k;
	l = k;
	i__2 = nk1x;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    right[i__] = c__[l];
	    l += nk1y;
/* L460: */
	}
	fpback_(&ax[ax_offset], &right[1], &nk1x, &ibandx, &right[1], nx);
	l = k;
	i__2 = nk1x;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    c__[l] = right[i__];
	    l += nk1y;
/* L470: */
	}
/* L480: */
    }
/*  calculate the quantities */
/*    res(i,j) = (z(i,j) - s(x(i),y(j)))**2 , i=1,2,..,mx;j=1,2,..,my */
/*    fp = sumi=1,mx(sumj=1,my(res(i,j))) */
/*    fpx(r) = sum''i(sumj=1,my(res(i,j))) , r=1,2,...,nx-2*kx-1 */
/*                  tx(r+kx) <= x(i) <= tx(r+kx+1) */
/*    fpy(r) = sumi=1,mx(sum''j(res(i,j))) , r=1,2,...,ny-2*ky-1 */
/*                  ty(r+ky) <= y(j) <= ty(r+ky+1) */
    *fp = 0.f;
    i__1 = *nx;
    for (i__ = 1; i__ <= i__1; ++i__) {
	fpx[i__] = 0.f;
/* L490: */
    }
    i__1 = *ny;
    for (i__ = 1; i__ <= i__1; ++i__) {
	fpy[i__] = 0.f;
/* L500: */
    }
    nk1y = *ny - *ky1;
    iz = 0;
    nroldx = 0;
/*  main loop for the different grid points. */
    i__1 = *mx;
    for (i1 = 1; i1 <= i__1; ++i1) {
	numx = nrx[i1];
	numx1 = numx + 1;
	nroldy = 0;
	i__2 = *my;
	for (i2 = 1; i2 <= i__2; ++i2) {
	    numy = nry[i2];
	    numy1 = numy + 1;
	    ++iz;
/*  evaluate s(x,y) at the current grid point by making the sum of the */
/*  cross products of the non-zero b-splines at (x,y), multiplied with */
/*  the appropiate b-spline coefficients. */
	    term = 0.f;
	    k1 = numx * nk1y + numy;
	    i__3 = *kx1;
	    for (l1 = 1; l1 <= i__3; ++l1) {
		k2 = k1;
		fac = spx[i1 + l1 * spx_dim1];
		i__4 = *ky1;
		for (l2 = 1; l2 <= i__4; ++l2) {
		    ++k2;
		    term += fac * spy[i2 + l2 * spy_dim1] * c__[k2];
/* L510: */
		}
		k1 += nk1y;
/* L520: */
	    }
/*  calculate the squared residual at the current grid point. */
/* Computing 2nd power */
	    r__1 = z__[iz] - term;
	    term = r__1 * r__1;
/*  adjust the different parameters. */
	    *fp += term;
	    fpx[numx1] += term;
	    fpy[numy1] += term;
	    fac = term * half;
	    if (numy == nroldy) {
		goto L530;
	    }
	    fpy[numy1] -= fac;
	    fpy[numy] += fac;
L530:
	    nroldy = numy;
	    if (numx == nroldx) {
		goto L540;
	    }
	    fpx[numx1] -= fac;
	    fpx[numx] += fac;
L540:
	    ;
	}
	nroldx = numx;
/* L550: */
    }
    return 0;
} /* fpgrre_ */

