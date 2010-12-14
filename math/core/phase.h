#ifndef PHASE_H_
#define PHASE_H_

/**
 * @file phase.h
 */

/**
 * @brief
 * Inline function macro used to compute the 2D geometric phase
 * for the horizontal (azimuth/elevation) coordinate system.
 */
#define GEOMETRIC_PHASE_2D_HORIZONTAL(x, y, cosEl, sinAz, cosAz, k) \
        (-k * cosEl * (x * sinAz + y * cosAz))

/**
 * @brief
 * Inline function macro used to compute the 2D geometric phase
 * for the spherical (theta/phi) coordinate system.
 */
#define GEOMETRIC_PHASE_2D_SPHERICAL(x, y, sinTheta, cosPhi, sinPhi, k) \
        (-k * sinTheta * (x * cosPhi + y * sinPhi))

#endif /* PHASE_H_ */
