#ifndef FLOATING_POINT_COMPARE_H_
#define FLOATING_POINT_COMPARE_H_

#include <cmath>
#include <limits>

namespace oskar {

/// Floating point comparison function.
template <typename T> bool approxEqual(T b, T a)
{
    return std::fabs(a - b) < std::numeric_limits<T>::epsilon();
}

// Relative error of 1.0e-5 is 99.999% accuracy
template <typename T> bool isEqual(T a, T b)
{
    if (std::fabs(a - b) < std::numeric_limits<T>::epsilon())
        return true;

    T relativeError;
    if (std::fabs(b) > std::fabs(a))
        relativeError = std::fabs((a - b) / b);
    else
        relativeError = std::fabs((a - b) / a);

    if (relativeError <= (T)1.0e-5)
        return true;

    return false;
}


} // namespace oskar
#endif // FLOATING_POINT_COMPARE_H_
