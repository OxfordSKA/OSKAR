#ifndef SKY_H_
#define SKY_H_

//
// === C++ class interface is a data container and functions.
//
// === will need MATLAB mex functions.
// === will need python wrapper to the class.
//
//(=== will need c wrappers for a C library interface)



class Sky
{
    public:
        // Generate a number of sources in ra, dec (about the phase centre?)
        void generate_sources();

        // Append Ra, dec sources (in phase centre units).
        void append_sources_equatorial();

        // Append sources at l, m positions. (i.e. pixel positions)
        void append_sources_lm();

        // Filter sources by inner and outer radius
        void filter_by_radius();

        // Filter sources by brightness.
        void filter_by_brightness();

    private:



};


#endif // SKY_H_
