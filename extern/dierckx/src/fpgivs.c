#include <math.h>

/**
 * @details
 * Subroutine fpgivs calculates the parameters of a givens
 * transformation.
 */

void fpgivs(float piv, float *ww, float *cos_, float *sin_)
{
    /* System generated locals */
    float t;

    /* Local variables */
    float dd, store;

    store = fabsf(piv);
    if (store >= *ww)
    {
        /* Computing 2nd power */
	    t = *ww / piv;
	    dd = store * sqrt(1.f + t * t);
    }
    if (store < *ww)
    {
        /* Computing 2nd power */
	    t = piv / *ww;
	    dd = *ww * sqrt(1.f + t * t);
    }
    *cos_ = *ww / dd;
    *sin_ = piv / dd;
    *ww = dd;
}

