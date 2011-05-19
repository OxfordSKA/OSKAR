#ifndef LINKED_SORT_H_
#define LINKED_SORT_H_

/**
 * @file LinkedSort.h
 */

#include <algorithm>
#include <utility>
#include <vector>
#include <cstring>


/**
 * @class LinkedSort
 * @brief
 * @details
 */
class LinkedSort
{
    public:
        LinkedSort() {}

        virtual ~LinkedSort() {}

        template <typename T0, typename T1>
        static void lSort(const unsigned n, T0 * values, T1 * linked1);

        template <typename T0, typename T1, typename T2, typename T3>
        static void lSort(const unsigned n, T0 * values, T1 * linked1,
                T2 * linked2, T3 * linked3);


        template <typename T> struct Pair
        {
                unsigned index;
                T value;
        };

    public:
        /// Sort values and return sorted values and corresponding indices.
        template <typename T>
        static void sortIndices(const unsigned n, T * values, unsigned * indices);

        /// Reorder values based on vector of indices.
        template <typename T>
        static void reorder(const unsigned n, T * values, const unsigned * indices);

    private:
        /// Comparison function for sorting indices.
        template <typename T>
        static bool _compare(const Pair<T> & l, const Pair<T>& r);
};


//-----------------------------------------------------------------------------
template <typename T0, typename T1>
void LinkedSort::lSort(const unsigned n, T0 * values, T1 * linked1)
{
    std::vector<unsigned> indices(n);
    sortIndices<T0>(n, values, &indices[0]);
    reorder<T1>(n, linked1, &indices[0]);
}


template <typename T0, typename T1, typename T2, typename T3>
void LinkedSort::lSort(const unsigned n, T0 * values, T1 * linked1,
        T2 * linked2, T3 * linked3)
{
    std::vector<unsigned> indices(n);
    sortIndices<T0>(n, values, &indices[0]);
    reorder<T1>(n, linked1, &indices[0]);
    reorder<T2>(n, linked2, &indices[0]);
    reorder<T3>(n, linked3, &indices[0]);
}



/// Sort values and return sorted values and corresponding indices.
template <typename T>
void LinkedSort::sortIndices(const unsigned n, T * values, unsigned * indices)
{
    std::vector<Pair<T> > temp(n);
    Pair<T> * tempPtr = &temp[0];
    for (unsigned i = 0; i < n; ++i)
    {
        tempPtr[i].index = i;
        tempPtr[i].value = values[i];
    }

    // Most of the time is taken with this sort.
    std::sort(temp.begin(), temp.end(), _compare<T>);

    for (unsigned i = 0; i < n; ++i)
    {
        indices[i] = tempPtr[i].index;
        values[i] = tempPtr[i].value;
    }
}

/// Reorder values based on vector of indices.
template <typename T>
void LinkedSort::reorder(const unsigned n, T * values, const unsigned * indices)
{
    std::vector<T> temp(n);
    for (unsigned i = 0; i < n; ++i)
    {
        temp[i] = values[indices[i]];
    }
    memcpy((void*)values, (const void*)&temp[0], n * sizeof(T));
}




/// Comparison function for sorting indices.
template <typename T>
bool LinkedSort::_compare(const Pair<T>& l, const Pair<T>& r)
{
    return l.value < r.value;
}

#endif // LINKED_SORT_H_
