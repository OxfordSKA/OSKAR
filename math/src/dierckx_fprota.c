#include "math/dierckx_fprota.h"

#ifdef __cplusplus
extern "C" {
#endif

void dierckx_fprota_f(float cos_, float sin_, float *a, float *b)
{
    float stor1, stor2;

    stor1 = *a;
    stor2 = *b;
    *b = cos_ * stor2 + sin_ * stor1;
    *a = cos_ * stor1 - sin_ * stor2;
}

void dierckx_fprota_d(double cos_, double sin_, double *a, double *b)
{
    double stor1, stor2;

    stor1 = *a;
    stor2 = *b;
    *b = cos_ * stor2 + sin_ * stor1;
    *a = cos_ * stor1 - sin_ * stor2;
}

#ifdef __cplusplus
}
#endif
