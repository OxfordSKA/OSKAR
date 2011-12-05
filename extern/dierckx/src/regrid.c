/**
 * @details
 * Given the set of values z(i,j) on the rectangular grid (x(i),y(j)),
 * i=1,...,mx;j=1,...,my, subroutine regrid determines a smooth bivar-
 * iate spline approximation s(x,y) of degrees kx and ky on the rect-
 * angle xb <= x <= xe, yb <= y <= ye.
 * if iopt = -1 regrid calculates the least-squares spline according
 * to a given set of knots.
 * if iopt >= 0 the total numbers nx and ny of these knots and their
 * position tx(j),j=1,...,nx and ty(j),j=1,...,ny are chosen automatic-
 * ally by the routine. the smoothness of s(x,y) is then achieved by
 * minimalizing the discontinuity jumps in the derivatives of s(x,y)
 * across the boundaries of the subpanels (tx(i),tx(i+1))*(ty(j),ty(j+1).
 * the amounth of smoothness is determined by the condition that f(p) =
 * sum ((z(i,j)-s(x(i),y(j))))**2) be <= s, with s a given non-negative
 * constant, called the smoothing factor.
 * the fit is given in the b-spline representation (b-spline coefficients
 * c((ny-ky-1)*(i-1)+j),i=1,...,nx-kx-1;j=1,...,ny-ky-1) and can be eval-
 * uated by means of subroutine bispev.

 * calling sequence:
 *     call regrid(iopt,mx,x,my,y,z,xb,xe,yb,ye,kx,ky,s,nxest,nyest,
 *    *  nx,tx,ny,ty,c,fp,wrk,lwrk,iwrk,kwrk,ier)

 * parameters:
 *  iopt  : int flag. on entry iopt must specify whether a least-
 *          squares spline (iopt=-1) or a smoothing spline (iopt=0 or 1)
 *          must be determined.
 *          if iopt=0 the routine will start with an initial set of knots
 *          tx(i)=xb,tx(i+kx+1)=xe,i=1,...,kx+1;ty(i)=yb,ty(i+ky+1)=ye,i=
 *          1,...,ky+1. if iopt=1 the routine will continue with the set
 *          of knots found at the last call of the routine.
 *          attention: a call with iopt=1 must always be immediately pre-
 *                     ceded by another call with iopt=1 or iopt=0 and
 *                     s.ne.0.
 *          unchanged on exit.
 *  mx    : int. on entry mx must specify the number of grid points
 *          along the x-axis. mx > kx . unchanged on exit.
 *  x     : float array of dimension at least (mx). before entry, x(i)
 *          must be set to the x-co-ordinate of the i-th grid point
 *          along the x-axis, for i=1,2,...,mx. these values must be
 *          supplied in strictly ascending order. unchanged on exit.
 *  my    : int. on entry my must specify the number of grid points
 *          along the y-axis. my > ky . unchanged on exit.
 *  y     : float array of dimension at least (my). before entry, y(j)
 *          must be set to the y-co-ordinate of the j-th grid point
 *          along the y-axis, for j=1,2,...,my. these values must be
 *          supplied in strictly ascending order. unchanged on exit.
 *  z     : float array of dimension at least (mx*my).
 *          before entry, z(my*(i-1)+j) must be set to the data value at
 *          the grid point (x(i),y(j)) for i=1,...,mx and j=1,...,my.
 *          unchanged on exit.
 *  xb,xe : float values. on entry xb,xe,yb and ye must specify the bound-
 *  yb,ye   aries of the rectangular approximation domain.
 *          xb<=x(i)<=xe,i=1,...,mx; yb<=y(j)<=ye,j=1,...,my.
 *          unchanged on exit.
 *  kx,ky : int values. on entry kx and ky must specify the degrees
 *          of the spline. 1<=kx,ky<=5. it is recommended to use bicubic
 *          (kx=ky=3) splines. unchanged on exit.
 *  s     : float. on entry (in case iopt>=0) s must specify the smoothing
 *          factor. s >=0. unchanged on exit.
 *          for advice on the choice of s see further comments
 *  nxest : int. unchanged on exit.
 *  nyest : int. unchanged on exit.
 *          on entry, nxest and nyest must specify an upper bound for the
 *          number of knots required in the x- and y-directions respect.
 *          these numbers will also determine the storage space needed by
 *          the routine. nxest >= 2*(kx+1), nyest >= 2*(ky+1).
 *          in most practical situation nxest = mx/2, nyest=my/2, will
 *          be sufficient. always large enough are nxest=mx+kx+1, nyest=
 *          my+ky+1, the number of knots needed for interpolation (s=0).
 *          see also further comments.
 *  nx    : int.
 *          unless ier=10 (in case iopt >=0), nx will contain the total
 *          number of knots with respect to the x-variable, of the spline
 *          approximation returned. if the computation mode iopt=1 is
 *          used, the value of nx should be left unchanged between sub-
 *          sequent calls.
 *          in case iopt=-1, the value of nx should be specified on entry
 *  tx    : float array of dimension nmax.
 *          on succesful exit, this array will contain the knots of the
 *          spline with respect to the x-variable, i.e. the position of
 *          the interior knots tx(kx+2),...,tx(nx-kx-1) as well as the
 *          position of the additional knots tx(1)=...=tx(kx+1)=xb and
 *          tx(nx-kx)=...=tx(nx)=xe needed for the b-spline representat.
 *          if the computation mode iopt=1 is used, the values of tx(1),
 *          ...,tx(nx) should be left unchanged between subsequent calls.
 *          if the computation mode iopt=-1 is used, the values tx(kx+2),
 *          ...tx(nx-kx-1) must be supplied by the user, before entry.
 *          see also the restrictions (ier=10).
 *  ny    : int.
 *          unless ier=10 (in case iopt >=0), ny will contain the total
 *          number of knots with respect to the y-variable, of the spline
 *          approximation returned. if the computation mode iopt=1 is
 *          used, the value of ny should be left unchanged between sub-
 *          sequent calls.
 *          in case iopt=-1, the value of ny should be specified on entry
 *  ty    : float array of dimension nmax.
 *          on succesful exit, this array will contain the knots of the
 *          spline with respect to the y-variable, i.e. the position of
 *          the interior knots ty(ky+2),...,ty(ny-ky-1) as well as the
 *          position of the additional knots ty(1)=...=ty(ky+1)=yb and
 *          ty(ny-ky)=...=ty(ny)=ye needed for the b-spline representat.
 *          if the computation mode iopt=1 is used, the values of ty(1),
 *          ...,ty(ny) should be left unchanged between subsequent calls.
 *          if the computation mode iopt=-1 is used, the values ty(ky+2),
 *          ...ty(ny-ky-1) must be supplied by the user, before entry.
 *          see also the restrictions (ier=10).
 *  c     : float array of dimension at least (nxest-kx-1)*(nyest-ky-1).
 *          on succesful exit, c contains the coefficients of the spline
 *          approximation s(x,y)
 *  fp    : float. unless ier=10, fp contains the sum of squared
 *          residuals of the spline approximation returned.
 *  wrk   : float array of dimension (lwrk). used as workspace.
 *          if the computation mode iopt=1 is used the values of wrk(1),
 *          ...,wrk(4) should be left unchanged between subsequent calls.
 *  lwrk  : int. on entry lwrk must specify the actual dimension of
 *          the array wrk as declared in the calling (sub)program.
 *          lwrk must not be too small.
 *           lwrk >= 4+nxest*(my+2*kx+5)+nyest*(2*ky+5)+mx*(kx+1)+
 *            my*(ky+1) +u
 *           where u is the larger of my and nxest.
 *  iwrk  : int array of dimension (kwrk). used as workspace.
 *          if the computation mode iopt=1 is used the values of iwrk(1),
 *          ...,iwrk(3) should be left unchanged between subsequent calls
 *  kwrk  : int. on entry kwrk must specify the actual dimension of
 *          the array iwrk as declared in the calling (sub)program.
 *          kwrk >= 3+mx+my+nxest+nyest.
 *  ier   : int. unless the routine detects an error, ier contains a
 *          non-positive value on exit, i.e.
 *   ier=0  : normal return. the spline returned has a residual sum of
 *            squares fp such that abs(fp-s)/s <= tol with tol a relat-
 *            ive tolerance set to 0.001 by the program.
 *   ier=-1 : normal return. the spline returned is an interpolating
 *            spline (fp=0).
 *   ier=-2 : normal return. the spline returned is the least-squares
 *            polynomial of degrees kx and ky. in this extreme case fp
 *            gives the upper bound for the smoothing factor s.
 *   ier=1  : error. the required storage space exceeds the available
 *            storage space, as specified by the parameters nxest and
 *            nyest.
 *            probably causes : nxest or nyest too small. if these param-
 *            eters are already large, it may also indicate that s is
 *            too small
 *            the approximation returned is the least-squares spline
 *            according to the current set of knots. the parameter fp
 *            gives the corresponding sum of squared residuals (fp>s).
 *   ier=2  : error. a theoretically impossible result was found during
 *            the iteration proces for finding a smoothing spline with
 *            fp = s. probably causes : s too small.
 *            there is an approximation returned but the corresponding
 *            sum of squared residuals does not satisfy the condition
 *            abs(fp-s)/s < tol.
 *   ier=3  : error. the maximal number of iterations maxit (set to 20
 *            by the program) allowed for finding a smoothing spline
 *            with fp=s has been reached. probably causes : s too small
 *            there is an approximation returned but the corresponding
 *            sum of squared residuals does not satisfy the condition
 *            abs(fp-s)/s < tol.
 *   ier=10 : error. on entry, the input data are controlled on validity
 *            the following restrictions must be satisfied.
 *            -1<=iopt<=1, 1<=kx,ky<=5, mx>kx, my>ky, nxest>=2*kx+2,
 *            nyest>=2*ky+2, kwrk>=3+mx+my+nxest+nyest,
 *            lwrk >= 4+nxest*(my+2*kx+5)+nyest*(2*ky+5)+mx*(kx+1)+
 *             my*(ky+1) +max(my,nxest),
 *            xb<=x(i-1)<x(i)<=xe,i=2,..,mx,yb<=y(j-1)<y(j)<=ye,j=2,..,my
 *            if iopt=-1: 2*kx+2<=nx<=min(nxest,mx+kx+1)
 *                        xb<tx(kx+2)<tx(kx+3)<...<tx(nx-kx-1)<xe
 *                        2*ky+2<=ny<=min(nyest,my+ky+1)
 *                        yb<ty(ky+2)<ty(ky+3)<...<ty(ny-ky-1)<ye
 *                    the schoenberg-whitney conditions, i.e. there must
 *                    be subset of grid co-ordinates xx(p) and yy(q) such
 *                    that   tx(p) < xx(p) < tx(p+kx+1) ,p=1,...,nx-kx-1
 *                           ty(q) < yy(q) < ty(q+ky+1) ,q=1,...,ny-ky-1
 *            if iopt>=0: s>=0
 *                        if s=0 : nxest>=mx+kx+1, nyest>=my+ky+1
 *            if one of these conditions is found to be violated,control
 *            is immediately repassed to the calling program. in that
 *            case there is no approximation returned.

 * further comments:
 *   regrid does not allow individual weighting of the data-values.
 *   so, if these were determined to widely different accuracies, then
 *   perhaps the general data set routine surfit should rather be used
 *   in spite of efficiency.
 *   by means of the parameter s, the user can control the tradeoff
 *   between closeness of fit and smoothness of fit of the approximation.
 *   if s is too large, the spline will be too smooth and signal will be
 *   lost ; if s is too small the spline will pick up too much noise. in
 *   the extreme cases the program will return an interpolating spline if
 *   s=0 and the least-squares polynomial (degrees kx,ky) if s is
 *   very large. between these extremes, a properly chosen s will result
 *   in a good compromise between closeness of fit and smoothness of fit.
 *   to decide whether an approximation, corresponding to a certain s is
 *   satisfactory the user is highly recommended to inspect the fits
 *   graphically.
 *   recommended values for s depend on the accuracy of the data values.
 *   if the user has an idea of the statistical errors on the data, he
 *   can also find a proper estimate for s. for, by assuming that, if he
 *   specifies the right s, regrid will return a spline s(x,y) which
 *   exactly reproduces the function underlying the data he can evaluate
 *   the sum((z(i,j)-s(x(i),y(j)))**2) to find a good estimate for this s
 *   for example, if he knows that the statistical errors on his z(i,j)-
 *   values is not greater than 0.1, he may expect that a good s should
 *   have a value not larger than mx*my*(0.1)**2.
 *   if nothing is known about the statistical error in z(i,j), s must
 *   be determined by trial and error, taking account of the comments
 *   above. the best is then to start with a very large value of s (to
 *   determine the least-squares polynomial and the corresponding upper
 *   bound fp0 for s) and then to progressively decrease the value of s
 *   ( say by a factor 10 in the beginning, i.e. s=fp0/10,fp0/100,...
 *   and more carefully as the approximation shows more detail) to
 *   obtain closer fits.
 *   to economize the search for a good s-value the program provides with
 *   different modes of computation. at the first call of the routine, or
 *   whenever he wants to restart with the initial set of knots the user
 *   must set iopt=0.
 *   if iopt=1 the program will continue with the set of knots found at
 *   the last call of the routine. this will save a lot of computation
 *   time if regrid is called repeatedly for different values of s.
 *   the number of knots of the spline returned and their location will
 *   depend on the value of s and on the complexity of the shape of the
 *   function underlying the data. if the computation mode iopt=1
 *   is used, the knots returned may also depend on the s-values at
 *   previous calls (if these were smaller). therefore, if after a number
 *   of trials with different s-values and iopt=1, the user can finally
 *   accept a fit as satisfactory, it may be worthwhile for him to call
 *   regrid once more with the selected value for s but now with iopt=0.
 *   indeed, regrid may then return an approximation of the same quality
 *   of fit but with fewer knots and therefore better if data reduction
 *   is also an important objective for the user.
 *   the number of knots may also depend on the upper bounds nxest and
 *   nyest. indeed, if at a certain stage in regrid the number of knots
 *   in one direction (say nx) has reached the value of its upper bound
 *   (nxest), then from that moment on all subsequent knots are added
 *   in the other (y) direction. this may indicate that the value of
 *   nxest is too small. on the other hand, it gives the user the option
 *   of limiting the number of knots the routine locates in any direction
 *   for example, by setting nxest=2*kx+2 (the lowest allowable value for
 *   nxest), the user can indicate that he wants an approximation which
 *   is a simple polynomial of degree kx in the variable x.

 *  other subroutines required:
 *    fpback,fpbspl,fpregr,fpdisc,fpgivs,fpgrre,fprati,fprota,fpchec,
 *    fpknot

 *  references:
 *   dierckx p. : a fast algorithm for smoothing data on a rectangular
 *                grid while using spline functions, siam j.numer.anal.
 *                19 (1982) 1286-1304.
 *   dierckx p. : a fast algorithm for smoothing data on a rectangular
 *                grid while using spline functions, report tw53, dept.
 *                computer science,k.u.leuven, 1980.
 *   dierckx p. : curve and surface fitting with splines, monographs on
 *                numerical analysis, oxford university press, 1993.

 *  author:
 *    p.dierckx
 *    dept. computer science, k.u. leuven
 *    celestijnenlaan 200a, b-3001 heverlee, belgium.
 *    e-mail : Paul.Dierckx@cs.kuleuven.ac.be

 *  creation date : may 1979
 *  latest update : march 1989
 */

extern int fpchec_(float *, int *, float *, int *, 
    int *, int *), fpregr_(int *, float *, int *, float 
    *, int *, float *, int *, float *, float *, float *, float *, 
    int *, int *, float *, int *, int *, float *, 
    int *, int *, int *, float *, int *, float *, float *
    , float *, float *, float *, float *, float *, float *, float *, int 
    *, int *, int *, int *, int *, int *, int 
    *, float *, int *, int *);


void regrid_(int *iopt, int *mx, float *x, int *my,
	 float *y, float *z__, float *xb, float *xe, float *yb, float *ye, int *
	kx, int *ky, float *s, int *nxest, int *nyest, int *nx,
	 float *tx, int *ny, float *ty, float *c__, float *fp, float *wrk, 
	int *lwrk, int *iwrk, int *kwrk, int *ier)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    int i__, j, nc, mz, kx1, kx2, ky1, ky2;
    float tol;
    int lww, kndx, kndy, lfpx, lfpy, jwrk, knrx, knry, maxit, nminx, 
	    nminy, kwest, lwest;

    /* Parameter adjustments */
    --x;
    --z__;
    --y;
    --tx;
    --c__;
    --ty;
    --wrk;
    --iwrk;

    /* Function Body */
    /* we set up the parameters tol and maxit. */
    maxit = 20;
    tol = .001f;
    /* before starting computations a data check is made. if the input data */
    /* are invalid, control is immediately repassed to the calling program. */
    *ier = 10;
    if (*kx <= 0 || *kx > 5) return;
    kx1 = *kx + 1;
    kx2 = kx1 + 1;
    if (*ky <= 0 || *ky > 5) return;
    ky1 = *ky + 1;
    ky2 = ky1 + 1;
    if (*iopt < -1 || *iopt > 1) return;
    nminx = kx1 << 1;
    if (*mx < kx1 || *nxest < nminx) return;
    nminy = ky1 << 1;
    if (*my < ky1 || *nyest < nminy) return;
    mz = *mx * *my;
    nc = (*nxest - kx1) * (*nyest - ky1);
    lwest = *nxest * (*my + (kx2 << 1) + 1) + 4 + *nyest * ((ky2 << 1) + 1) + 
	    *mx * kx1 + *my * ky1 + max(*nxest,*my);
    kwest = *mx + 3 + *my + *nxest + *nyest;
    if (*lwrk < lwest || *kwrk < kwest) return;
    if (*xb > x[1] || *xe < x[*mx]) return;
    i__1 = *mx;
    for (i__ = 2; i__ <= i__1; ++i__)
	    if (x[i__ - 1] >= x[i__]) return;
    if (*yb > y[1] || *ye < y[*my]) return;
    i__1 = *my;
    for (i__ = 2; i__ <= i__1; ++i__)
	    if (y[i__ - 1] >= y[i__]) return;
    if (*iopt >= 0) {
	    goto L50;
    }
    if (*nx < nminx || *nx > *nxest) return;
    j = *nx;
    i__1 = kx1;
    for (i__ = 1; i__ <= i__1; ++i__)
    {
	    tx[i__] = *xb;
	    tx[j] = *xe;
	    --j;
    }
    fpchec_(&x[1], mx, &tx[1], nx, kx, ier);
    if (*ier != 0) return;
    if (*ny < nminy || *ny > *nyest) return;
    j = *ny;
    i__1 = ky1;
    for (i__ = 1; i__ <= i__1; ++i__)
    {
	    ty[i__] = *yb;
	    ty[j] = *ye;
	    --j;
    }
    fpchec_(&y[1], my, &ty[1], ny, ky, ier);
    if (*ier != 0) return;
    else goto L60;
L50:
    if (*s < 0.f) return;
    if (*s == 0.f && (*nxest < *mx + kx1 || *nyest < *my + ky1)) return;
    
    *ier = 0;
    /* we partition the working space and determine the spline approximation */
L60:
    lfpx = 5;
    lfpy = lfpx + *nxest;
    lww = lfpy + *nyest;
    jwrk = *lwrk - 4 - *nxest - *nyest;
    knrx = 4;
    knry = knrx + *mx;
    kndx = knry + *my;
    kndy = kndx + *nxest;
    fpregr_(iopt, &x[1], mx, &y[1], my, &z__[1], &mz, xb, xe, yb, ye, kx, ky, 
	    s, nxest, nyest, &tol, &maxit, &nc, nx, &tx[1], ny, &ty[1], &c__[
	    1], fp, &wrk[1], &wrk[2], &wrk[3], &wrk[4], &wrk[lfpx], &wrk[lfpy]
	    , &iwrk[1], &iwrk[2], &iwrk[3], &iwrk[knrx], &iwrk[knry], &iwrk[
	    kndx], &iwrk[kndy], &wrk[lww], &jwrk, ier);
}

