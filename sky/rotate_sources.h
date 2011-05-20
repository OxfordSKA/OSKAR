#ifndef ROTATE_SOURCES_H_
#define ROTATE_SOURCES_H_


void rotate_sources_to_phase_centre(const unsigned num_sources,
        double * ra, double * dec, const double ra0, const double dec0);

void mult_matrix_vector(const double * M, double * v);


#endif // ROTATE_SOURCES_H_
