/**
 * @details
 * Subroutine fprota applies a givens rotation to a and b.
 */

void fprota(float cos_, float sin_, float *a, float *b)
{
    float stor1, stor2;

    stor1 = *a;
    stor2 = *b;
    *b = cos_ * stor2 + sin_ * stor1;
    *a = cos_ * stor1 - sin_ * stor2;
}

