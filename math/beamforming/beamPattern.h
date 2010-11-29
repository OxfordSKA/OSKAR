#ifndef MATH_BEAMPATTERN_H_
#define MATH_BEAMPATTERN_H_

/**
 * @file beamPattern.h
 */

/// Computes a beam pattern using multi-threading.
void beamPattern(const int na, const float* ax, const float* ay,
        const int ns, const float* slon, const float* slat,
        const float ba, const float be, const float k,
        float* image);

#endif /* MATH_BEAMPATTERN_H_ */
