#ifndef OSKAR_DIERCKX_FPSPHE_H_
#define OSKAR_DIERCKX_FPSPHE_H_

/**
 * @file oskar_dierckx_fpsphe.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Internal routine used by sphere.
 */
void oskar_dierckx_fpsphe(int iopt, int m, const double* theta,
        const double* phi, const double* r, const double* w, double s,
        int ntest, int npest, double eta, double tol, int maxit, int ncc,
        int* nt, double* tt, int* np, double* tp, double* c, double* fp,
        double* sup, double* fpint, double* coord, double* f, double* ff,
        double* row, double* coco, double* cosi, double* a, double* q,
        double* bt, double* bp, double* spt, double* spp, double* h,
        int* index, int* nummer, double* wrk, int lwrk, int* ier);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_DIERCKX_FPSPHE_H_ */

