/**
 * @details
 * Subroutine fpbspl evaluates the (k+1) non-zero b-splines of
 * degree k at t(l) <= x < t(l+1) using the stable recurrence
 * relation of de boor and cox.
 */

void fpbspl(const float *t, int k, float x, int l, float *h)
{
    /* Local variables */
    float f;
    int i, j;
    float hh[5];
    int li, lj;

    /* Parameter adjustments */
    --t;
    --h;

    /* Function Body */
    h[1] = 1.f;
    for (j = 1; j <= k; ++j)
    {
	    for (i = 1; i <= j; ++i)
	        hh[i - 1] = h[i];
	    h[1] = 0.f;
	    for (i = 1; i <= j; ++i)
	    {
	        li = l + i;
	        lj = li - j;
	        f = hh[i - 1] / (t[li] - t[lj]);
	        h[i] += f * (t[li] - x);
	        h[i + 1] = f * (x - t[lj]);
	    }
    }
}

