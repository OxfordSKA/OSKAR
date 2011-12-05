/**
 * @details
 * Subroutine bispev evaluates on a grid (x(i),y(j)),i=1,...,mx; j=1,...
 * ,my a bivariate spline s(x,y) of degrees kx and ky, given in the
 * b-spline representation.
 *
 * calling sequence:
 *    call bispev(tx,nx,ty,ny,c,kx,ky,x,mx,y,my,z,wrk,lwrk,
 *   * iwrk,kwrk,ier)
 *
 * input parameters:
 *  tx    : float array, length nx, which contains the position of the
 *          knots in the x-direction.
 *  nx    : int, giving the total number of knots in the x-direction
 *  ty    : float array, length ny, which contains the position of the
 *          knots in the y-direction.
 *  ny    : int, giving the total number of knots in the y-direction
 *  c     : float array, length (nx-kx-1)*(ny-ky-1), which contains the
 *          b-spline coefficients.
 *  kx,ky : int values, giving the degrees of the spline.
 *  x     : float array of dimension (mx).
 *          before entry x(i) must be set to the x co-ordinate of the
 *          i-th grid point along the x-axis.
 *          tx(kx+1)<=x(i-1)<=x(i)<=tx(nx-kx), i=2,...,mx.
 *  mx    : on entry mx must specify the number of grid points along
 *          the x-axis. mx >=1.
 *  y     : float array of dimension (my).
 *          before entry y(j) must be set to the y co-ordinate of the
 *          j-th grid point along the y-axis.
 *          ty(ky+1)<=y(j-1)<=y(j)<=ty(ny-ky), j=2,...,my.
 *  my    : on entry my must specify the number of grid points along
 *          the y-axis. my >=1.
 *  wrk   : float array of dimension lwrk. used as workspace.
 *  lwrk  : int, specifying the dimension of wrk.
 *          lwrk >= mx*(kx+1)+my*(ky+1)
 *  iwrk  : int array of dimension kwrk. used as workspace.
 *  kwrk  : int, specifying the dimension of iwrk. kwrk >= mx+my.
 *
 * output parameters:
 *  z     : float array of dimension (mx*my).
 *          on succesful exit z(my*(i-1)+j) contains the value of s(x,y)
 *          at the point (x(i),y(j)),i=1,...,mx;j=1,...,my.
 *  ier   : int error flag
 *   ier=0 : normal return
 *   ier=10: invalid input data (see restrictions)
 *
 * restrictions:
 *  mx >=1, my >=1, lwrk>=mx*(kx+1)+my*(ky+1), kwrk>=mx+my
 *  tx(kx+1) <= x(i-1) <= x(i) <= tx(nx-kx), i=2,...,mx
 *  ty(ky+1) <= y(j-1) <= y(j) <= ty(ny-ky), j=2,...,my
 *
 * other subroutines required:
 *   fpbisp,fpbspl
 *
 * references :
 *   de boor c : on calculating with b-splines, j. approximation theory
 *               6 (1972) 50-62.
 *   cox m.g.  : the numerical evaluation of b-splines, j. inst. maths
 *               applics 10 (1972) 134-149.
 *   dierckx p. : curve and surface fitting with splines, monographs on
 *                numerical analysis, oxford university press, 1993.
 *
 * author :
 *   p.dierckx
 *   dept. computer science, k.u.leuven
 *   celestijnenlaan 200a, b-3001 heverlee, belgium.
 *   e-mail : Paul.Dierckx@cs.kuleuven.ac.be
 *
 * latest update : march 1987
 */

extern void fpbisp(const float *tx, int nx, const float *ty, int ny, 
	const float *c, int kx, int ky, const float *x, int mx, const float *y, 
	int my, float *z, float *wx, float *wy, int *lx, int *ly);

void bispev(const float *tx, int nx, const float *ty, int ny, 
	const float *c, int kx, int ky, const float *x, int mx, const float *y, 
	int my, float *z, float *wrk, int lwrk, int *iwrk, int kwrk, int *ier)
{
    /* Local variables */
    int i, iw, lwest;

    /* before starting computations a data check is made. if the input data
     * are invalid control is immediately repassed to the calling program. */

    *ier = 10;
    lwest = (kx + 1) * mx + (ky + 1) * my;
    if (lwrk < lwest) return;
    if (kwrk < mx + my) return;
    if (mx < 1) return;
    if (my < 1) return;
    for (i = 1; i < mx; ++i)
        if (x[i] < x[i - 1]) return;
    for (i = 1; i < my; ++i)
        if (y[i] < y[i - 1]) return;
    *ier = 0;
    iw = mx * (kx + 1);
    fpbisp(tx, nx, ty, ny, c, kx, ky, x, mx, y, my,
        z, wrk, &wrk[iw], iwrk, &iwrk[mx]);
}

