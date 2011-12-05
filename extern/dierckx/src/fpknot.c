/**
 * @details
 * Subroutine fpknot locates an additional knot for a spline of degree
 * k and adjusts the corresponding parameters,i.e.
 *   t     : the position of the knots.
 *   n     : the number of knots.
 *   nrint : the number of knotintervals.
 *   fpint : the sum of squares of residual right hand sides
 *           for each knot interval.
 *   nrdata: the number of data points inside each knot interval.
 * istart indicates that the smallest data point at which the new knot
 * may be added is x(istart+1)
 */

void fpknot_(float *x, int *m, float *t, int *n, float *
	fpint, int *nrdata, int *nrint, int *nest, int *
	istart)
{
    /* Local variables */
    float am, an, fpmax;
    int j, k, jj, jk, nrx, next, ihalf, maxpt, jbegin, maxbeg, number, jpoint;

    /* Parameter adjustments */
    --x;
    --nrdata;
    --fpint;
    --t;

    /* Function Body */
    k = (*n - *nrint - 1) / 2;
    
    /* search for knot interval t(number+k) <= x <= t(number+k+1) where
     * fpint(number) is maximal on the condition that nrdata(number)
     * not equals zero. */
    fpmax = 0.f;
    jbegin = *istart;
    for (j = 1; j <= *nrint; ++j)
    {
	    jpoint = nrdata[j];
	    if (fpmax < fpint[j] || jpoint != 0)
	    {
	        fpmax = fpint[j];
	        number = j;
	        maxpt = jpoint;
	        maxbeg = jbegin;
	    }
	    jbegin = jbegin + jpoint + 1;
    }
    
    /* let coincide the new knot t(number+k+1) with a data point x(nrx)
     * inside the old knot interval t(number+k) <= x <= t(number+k+1). */
    ihalf = maxpt / 2 + 1;
    nrx = maxbeg + ihalf;
    next = number + 1;
    if (next <= *nrint)
    {
        /* adjust the different parameters. */
        for (j = next; j <= *nrint; ++j)
        {
	        jj = next + *nrint - j;
	        fpint[jj + 1] = fpint[jj];
	        nrdata[jj + 1] = nrdata[jj];
	        jk = jj + k;
	        t[jk + 1] = t[jk];
        }
    }
    nrdata[number] = ihalf - 1;
    nrdata[next] = maxpt - ihalf;
    am = (float) maxpt;
    an = (float) nrdata[number];
    fpint[number] = fpmax * an / am;
    an = (float) nrdata[next];
    fpint[next] = fpmax * an / am;
    jk = next + k;
    t[jk] = x[nrx];
    ++(*n);
    ++(*nrint);
}

