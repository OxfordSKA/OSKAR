#ifndef GRID_CORRECTION_H_
#define GRID_CORRECTION_H_

#include <vector>

namespace oskar {

// forward class declarations.
class ConvFunc;

/**
 * @class GridCorrection
 *
 * @brief
 * Computes the grid correction.
 *
 * @details
 */

class GridCorrection
{
    public:
        void computeCorrection(ConvFunc & c, const unsigned grid_size);

        void make2D();

    public:
        const float * values() const { return &_correction[0]; }
        unsigned size() const { return _size; }


    private:
        float findMax();

    private:
        std::vector<float> _correction;
        unsigned _size;
};


} // namespace oskar
#endif // GRID_CORRECTION_H_
