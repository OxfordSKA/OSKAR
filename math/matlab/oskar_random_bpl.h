#ifndef OSKAR_RANDOM_BPL_H_
#define OSKAR_RANDOM_BPL_H_

extern "C"
void oskar_random_bpl(int n, double min, double max, double threshold,
        double power1, double power2, int seed, double * values);

#endif // OSKAR_RANDOM_BPL_H_
