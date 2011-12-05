
extern void fpbspl(const float *t, int k, float x, int l, float *h);

void fpbisp(const float *tx, int nx, const float *ty, int ny, 
	const float *c, int kx, int ky, const float *x, int mx, const float *y, 
	int my, float *z, float *wx, float *wy, int *lx, int *ly)
{
    /* Local variables */
    float h[6];
    int i, j, l, m, i1, j1, l1, l2;
    float tb, te, sp;
    int kx1, ky1;
    float arg;
    int nkx1, nky1;

    /* Parameter adjustments */
    --tx;
    --ty;
    --c;
    --lx;
    wx -= (1 + mx);
    --x;
    --ly;
    wy -= (1 + my);
    --z;
    --y;

    /* Function Body */
    kx1 = kx + 1;
    nkx1 = nx - kx1;
    tb = tx[kx1];
    te = tx[nkx1 + 1];
    l = kx1;
    l1 = l + 1;
    for (i = 1; i <= mx; ++i)
    {
	    arg = x[i];
	    if (arg < tb)
	        arg = tb;
	    if (arg > te)
	        arg = te;

        while (!(arg < tx[l1] || l == nkx1))
        {
            l = l1;
            l1 = l + 1;
        }

	    fpbspl(&tx[1], kx, arg, l, h);
	    lx[i] = l - kx1;
	    for (j = 1; j <= kx1; ++j)
	        wx[i + j * mx] = h[j - 1];
    }
    ky1 = ky + 1;
    nky1 = ny - ky1;
    tb = ty[ky1];
    te = ty[nky1 + 1];
    l = ky1;
    l1 = l + 1;
    for (i = 1; i <= my; ++i)
    {
	    arg = y[i];
	    if (arg < tb)
	        arg = tb;
	    if (arg > te)
	        arg = te;

	    while (!(arg < ty[l1] || l == nky1))
	    {
	        l = l1;
	        l1 = l + 1;
        }
        
	    fpbspl(&ty[1], ky, arg, l, h);
	    ly[i] = l - ky1;
	    for (j = 1; j <= ky1; ++j)
	        wy[i + j * my] = h[j - 1];
    }
    
    m = 0;
    for (i = 1; i <= mx; ++i)
    {
	    l = lx[i] * nky1;
	    for (i1 = 1; i1 <= kx1; ++i1)
	        h[i1 - 1] = wx[i + i1 * mx];
	    for (j = 1; j <= my; ++j)
	    {
	        l1 = l + ly[j];
	        sp = 0.f;
	        for (i1 = 1; i1 <= kx1; ++i1)
	        {
		        l2 = l1;
		        for (j1 = 1; j1 <= ky1; ++j1)
		        {
		            ++l2;
		            sp += c[l2] * h[i1 - 1] * wy[j + j1 * my];
		        }
		        l1 += nky1;
	        }
	        ++m;
	        z[m] = sp;
	    }
    }
}

