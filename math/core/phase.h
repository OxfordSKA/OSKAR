#ifndef PHASE_H_
#define PHASE_H_

/**
 * @file phase.h
 */

/// Inline function macro used to compute the geometric phase.
#define GEOMETRIC_PHASE(x, y, cosEl, sinAz, cosAz, k) \
    (-k * cosEl * (x * sinAz + y * cosAz))

#endif /* PHASE_H_ */
