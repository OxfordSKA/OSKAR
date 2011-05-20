#ifndef ANGLES_FROM_LM_H_
#define ANGLES_FROM_LM_H_

void angles_from_lm(const unsigned num_positions, const double ra0,
        const double dec0, const double * l, const double * m, double * ra,
        double * dec);

#endif // ANGLES_FROM_LM_H_
