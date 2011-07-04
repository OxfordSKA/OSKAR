#ifndef WRITE_UVFITS_H_
#define WRITE_UVFITS_H_

#include <QtCore/QString>
#include <QtCore/QVector>
#include <fitsio.h>

namespace oskar {

void write_uvfits();


/**
 * @class UVFitsWriter
 *
 * @brief Class to write UVFITS data files.
 *
 * @details
 * see:
 *  http://archive.stsci.edu/fits/fits_standard/fits_standard.html
 *  ftp://ftp.aoc.nrao.edu/pub/software/aips/TEXT/PUBL/FITS-IDI.pdf
 */

class UVFitsWriter
{
    public:
        UVFitsWriter();

        ~UVFitsWriter();

    public:
        void open_file(const QString & filename, const bool replace = true);

        void close_file();

        void write_header(const long long num_vis);

        void write_groups_header(const long num_stokes = 1,
                const long num_freqs = 1, const long num_ra = 1,
                const long num_dec = 1);

        void write_axis_header(const int id, const QString & ctype,
                const QString & comment, const double crval, const double cdelt,
                const double crpix, const double crota);

        void write_param_header(const int id, const QString & type,
                const QString & comment, const double scale, const double zero);

        int num_amps_per_group();

        void write_data(const float * u, const float * v, const float * w,
                const float * date, const float * baseline, const float * re,
                const float * im, const float * weight);

    private:
        void check_status(const QString & message = QString());

    private:
        fitsfile * _fptr;       /// CFITSIO structure holding file info.
        QString _filename;      /// Filename of the open fits file.
        int _status;            /// CFITSIO error status.
        int _decimals;          /// Number of decimal places for double keywords.
        int _num_axis;          /// Number of data axes.
        QVector<long> _axis_dim;/// Data axis dimensions.
        int _num_param;         /// Number of parameters per visibility (group)
        int _num_vis;           /// Number of visibilities (= number of groups)
};



} // namespace oskar;
#endif // WRITE_UV_FITS
