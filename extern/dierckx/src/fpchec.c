/**
 * @details
 * Subroutine fpchec verifies the number and the position of the knots
 * t(j),j=1,2,...,n of a spline of degree k, in relation to the number
 * and the position of the data points x(i),i=1,2,...,m. if all of the
 * following conditions are fulfilled, the error parameter ier is set
 * to zero. if one of the conditions is violated ier is set to ten.
 *     1) k+1 <= n-k-1 <= m
 *     2) t(1) <= t(2) <= ... <= t(k+1)
 *        t(n-k) <= t(n-k+1) <= ... <= t(n)
 *     3) t(k+1) < t(k+2) < ... < t(n-k)
 *     4) t(k+1) <= x(i) <= t(n-k)
 *     5) the conditions specified by schoenberg and whitney must hold
 *        for at least one subset of data points, i.e. there must be a
 *        subset of data points y(j) such that
 *            t(j) < y(j) < t(j+k+1), j=1,2,...,n-k-1
 */

void fpchec(float *x, int m, float *t, int n, int k, int *ier)
{
    /* Local variables */
    int i, j, l, k1, k2;
    float tj, tl;
    int nk1, nk2, nk3;

    /* Parameter adjustments */
    --x;
    --t;

    /* Function Body */
    k1 = k + 1;
    k2 = k1 + 1;
    nk1 = n - k1;
    nk2 = nk1 + 1;
    *ier = 10;
    
    /* check condition no 1 */
    if (nk1 < k1 || nk1 > m) return;
    
    /* check condition no 2 */
    j = n;
    for (i = 1; i <= k; ++i)
    {
	    if (t[i] > t[i + 1]) return;
	    if (t[j] < t[j - 1]) return;
	    --j;
    }
    
    /* check condition no 3 */
    for (i = k2; i <= nk2; ++i)
	    if (t[i] <= t[i - 1]) return;
    
    /* check condition no 4 */
    if (x[1] < t[k1] || x[m] > t[nk2]) return;
    
    /* check condition no 5 */
    if (x[1] >= t[k2] || x[m] <= t[nk1]) return;
    i = 1;
    l = k2;
    nk3 = nk1 - 1;
    if (nk3 < 2)
    {
        *ier = 0;
        return;
    }
    for (j = 2; j <= nk3; ++j)
    {
	    tj = t[j];
	    ++l;
	    tl = t[l];
	    
	    do
	    {
	        ++i;
	        if (i >= m) return;
	    } while (x[i] <= tj);
	    if (x[i] >= tl) return;
    }
    *ier = 0;
}

