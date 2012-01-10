#ifndef DIERCKX_FPORDE_H_
#define DIERCKX_FPORDE_H_

/**
 * @file fporde.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Subroutine fporde sorts the data points (x(i),y(i)),i=1,2,...,m
 * according to the panel tx(l)<=x<tx(l+1),ty(k)<=y<ty(k+1), they belong
 * to. for each panel a stack is constructed  containing the numbers
 * of data points lying inside; index(j),j=1,2,...,nreg points to the
 * first data point in the jth panel while nummer(i),i=1,2,...,m gives
 * the number of the next data point in the panel.
 */
void fporde_f(const float *x, const float *y, int m, int kx,
        int ky, const float *tx, int nx, const float *ty, int ny,
        int *nummer, int *index, int nreg);

/**
 * @details
 * Subroutine fporde sorts the data points (x(i),y(i)),i=1,2,...,m
 * according to the panel tx(l)<=x<tx(l+1),ty(k)<=y<ty(k+1), they belong
 * to. for each panel a stack is constructed  containing the numbers
 * of data points lying inside; index(j),j=1,2,...,nreg points to the
 * first data point in the jth panel while nummer(i),i=1,2,...,m gives
 * the number of the next data point in the panel.
 */
void fporde_d(const double *x, const double *y, int m, int kx,
        int ky, const double *tx, int nx, const double *ty, int ny,
        int *nummer, int *index, int nreg);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPORDE_H_ */
