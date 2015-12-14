#ifndef ROUNDING_H_
#define ROUNDING_H_

#include <cmath>

namespace oskar {

/*
From:
http://www.eetimes.com/design/programmable-logic/4014804/An-introduction-to-different-rounding-algorithms
and
http://www.cplusplus.com/forum/articles/3638/

Two categories:
  - Symmetric about zero.
  - Biased in some way.

Biased Rounding
---------------

The <cmath> floor funciton, for example is biased towards negative infinity
as it always chooses the lower integer number - the number closer to negative
infinity.

    floor(+7.5) --> 7
    floor(-7.5) --> -8

Symmetrical rounding
--------------------

Special case of bias centred about zero. To 'fix' the floor function to
tend towards zero.

double floor0
{
    if (value < 0.0) return ceil(value)
    else return floor(value)
}

Now, the absolute value of the result will always be the same.
floor0(-7.7) --> -7
floor0(+7.7) --> +7
floor0(-7.5) --> -7
floor0(+7.5) --> +7
floor0(-7.3) --> -7
floor0(+7.3) --> +7

Unbiased Rounding
-----------------

So how to handle biases?

Using simple rounding. if next digit is 5 or more, round up, 5 or less round
down.

double round(double value)
{
    return floor(value + 0.5);
}

This is however still biased, now towards positive infinity as we always
choose to round up the exactly halfway values.

round(10.3) --> 10
round(10.5) --> 11
round(10.7) --> 11

Which way do we round when exactly halfway between two values?

One very popular method is variously called "bankers rounding", "round to even",
"convergent rounding" and even "unbiased rounding".


given a number exactly half way between two values, round to the even value
(zero considered even).

round(1.7) --> 2
round(1.5) --> 2
round(1.3) --> 1
round(2.7) --> 3
round(2.5) --> 2
round(2.3) --> 2

For random data this is very convenient as it is unbiased however its still
biased if the data is biased (not random).

One solution is "alternative rounding". It works by simply choosing to bias
up or down every other time.

round(1.5) --> 2
round(1.5) --> 1
round(1.5) --> 2
round(1.5) --> 1
etc...

This is not always useful though.

The only way to eliminate all bias is to use random bias. This is of course
impossible to generate on a typical PC but goes towards solving the problem.

If a sample is exactly halfway between two integers, it chooses one or other
randomly. Problem being the random number generator used with the pesudoranom
number generator for C and C++ being not that great.

*/

// Round down.
// Bias: -Infinity
using std::floor;


// Round up.
// Bias: +Infinity
using std::ceil;


// Symmetric round down
// Bias: towards zero
template <typename T> T floor0(const T& x)
{
    T result = floor(std::fabs(x));
    return (x < 0.0) ? -result : result;
}

// Symmetric round up
// Bias: away from zero
template <typename T> T ceil0(const T& x)
{
    T result = ceil(std::fabs(x));
    return (x < 0.0) ? -result : result;
}

// Common rounding: round half up.
// i.e. next digit 5 and above goes up.
// Bias: +Infinity
template <typename T> T roundHalfUp( const T& x )
{
    return floor(x + 0.5);
}


// Round half down
// Bias: -InfinityroundHalfUp0
template <typename T> T roundHalfDown(const T & x)
{
    return ceil(x - 0.5);
}

// Symmetric round half down
// Bias: towards zero
//
// this would also work... faster?
// -----
// (x > 0.0) ? floor(x) + centre : floor(x) + centre + 1;
// else centre;
// ------
template <typename T> T roundHalfDown0(const T & x)
{
    T result = roundHalfDown(std::fabs(x));
    return (x < 0.0) ? -result : result;
}


// symmetric round half up
// Bias: away from zero
template <typename T> T roundHalfUp0(const T & x)
{
    T result = roundHalfUp(std::fabs(x));
    return (x < 0.0) ? -result : result;
}


template <typename T> T roundHalfDownFixed(const T& x)
{
    T result = floor(x - 0.5);
    return result;
}


//template <typename T> inline T round(const T& x)
//{
//    return roundHalfUp0(x);
//}


} // namespace oskar
#endif // ROUNDING_H_
