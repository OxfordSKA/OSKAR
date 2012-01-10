#ifndef DIERCKX_FPSPHE_H_
#define DIERCKX_FPSPHE_H_

/**
 * @file fpsphe.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Internal routine used by sphere.
 */
void fpsphe_f(int iopt, int m, const float* theta, const float* phi,
        const float* r, const float* w, float s, int ntest, int npest,
        float eta, float tol, int maxit, int ncc, int* nt, float* tt,
        int* np, float* tp, float* c, float* fp, float* sup,
        float* fpint, float* coord, float* f, float* ff, float* row,
        float* coco, float* cosi, float* a, float* q, float* bt,
        float* bp, float* spt, float* spp, float* h, int* index,
        int* nummer, float* wrk, int lwrk, int* ier);

/**
 * @details
 * Internal routine used by sphere.
 */
void fpsphe_d(int iopt, int m, const double* theta, const double* phi,
        const double* r, const double* w, double s, int ntest, int npest,
        double eta, double tol, int maxit, int ncc, int* nt, double* tt,
        int* np, double* tp, double* c, double* fp, double* sup,
        double* fpint, double* coord, double* f, double* ff, double* row,
        double* coco, double* cosi, double* a, double* q, double* bt,
        double* bp, double* spt, double* spp, double* h, int* index,
        int* nummer, double* wrk, int lwrk, int* ier);

#ifdef __cplusplus
}
#endif

#endif /* DIERCKX_FPSPHE_H_ */
