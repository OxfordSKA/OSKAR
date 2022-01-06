/*
 * Copyright (c) 2014-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_settings_utility_string.h"

#include <sstream>
#include <iomanip>
#include <climits>
#include <cstdlib>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <string>

std::string oskar_settings_utility_string_reduce(const std::string& str,
                                                 const std::string& fill,
                                                 const std::string& whitespace)
{
    // Trim first
    std::string result = oskar_settings_utility_string_trim(str, whitespace);

    // Replace sub ranges
    size_t begin_space = result.find_first_of(whitespace);
    while (begin_space != std::string::npos)
    {
        const size_t end_space = result.find_first_not_of(whitespace,
                                                          begin_space);
        const size_t range = end_space - begin_space;
        result.replace(begin_space, range, fill);
        const size_t new_start = begin_space + fill.length();
        begin_space = result.find_first_of(whitespace, new_start);
    }

    return result;
}

std::string oskar_settings_utility_string_replace(std::string& s,
                                                  const std::string& to_replace,
                                                  const std::string& replace_with)
{
    size_t p = 0;
    while (p != std::string::npos)
    {
        p = s.find(to_replace);
        if (p != std::string::npos)
        {
            s.replace(p, to_replace.length(), replace_with);
        }
    }
    return s;
}

std::string oskar_settings_utility_string_trim(const std::string& s,
                                               const std::string& whitespace)
{
    // Find the first index that does not match whitespace.
    size_t i0 = s.find_first_not_of(whitespace);
    if (i0 == std::string::npos) return "";

    // Find the last index that does not match whitespace.
    size_t i1 = s.find_last_not_of(whitespace);

    // Find the string with whitespace subtracted from each end.
    return s.substr(i0, i1 - i0 + 1);
}

std::vector<std::string> oskar_settings_utility_string_get_type_params(
                const std::string& s)
{
    std::vector<std::string> params;
    std::stringstream ss_all(s);
    std::string p;
    while (getline(ss_all, p, '"')) {
        std::stringstream ss(p);
        while (getline(ss, p, ',')) {
            p = oskar_settings_utility_string_trim(p, " \t\r\n");
            if (!p.empty()) params.push_back(p);
        }
        if (getline(ss_all, p, '"')) {
            if (!p.empty()) params.push_back(p);
        }
    }
    return params;
}

int oskar_settings_utility_string_to_int(const std::string& s, bool *ok)
{
    int base = 10;
    char *endptr = 0;

    errno = 0;  // To distinguish success/failure after call
    long int val = strtol(s.c_str(), &endptr, base);

    // If argument ok is not null, check for various possible errors.
    if (ok) {
        *ok = true;
        //  Check for various possible errors
        if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN))
                        || (errno != 0 && val == 0))
        {
            *ok = false;
        }
        // No digits found.
        if (endptr == s.c_str()) *ok = false;
        // Further characters found in the string
        // NOTE this may not actually be an error ...
        if (*endptr != '\0') *ok = false;
    }

    // Value not in the integer range, set to 0 set fail flag.
    // NOTE: Long int is not a 64-bit integer type - this is not reliable!
//    if (val > INT_MAX || val < -INT_MAX) {
//        val = 0L;
//        if (ok) *ok = false;
//    }

    return static_cast<int>(val);
}

std::string oskar_settings_utility_int_to_string(int i)
{
    std::ostringstream ss;
    ss << i;
    return ss.str();
}

std::string oskar_settings_utility_string_to_upper(const std::string& s)
{
    std::string s_(s);
    for (size_t i = 0; i < s_.length(); ++i) s_[i] = toupper(s_[i]);
    return s_;
}

bool oskar_settings_utility_string_starts_with(const std::string& full_string,
                                               const std::string& fragment,
                                               bool case_senstive)
{
    std::string full_(full_string), frag_(fragment);
    if (case_senstive == false) {
        full_ = oskar_settings_utility_string_to_upper(full_string);
        frag_ = oskar_settings_utility_string_to_upper(fragment);
    }
    return (full_.find(frag_) == 0) ? true : false;
}

enum DoubleForm { DFDecimal, DFExponent, DFSignificantDigits };
enum Flags
{
    NoFlags             = 0,
    Alternate           = 0x01,
    ZeroPadded          = 0x02,
    LeftAdjusted        = 0x04,
    BlankBeforePositive = 0x08,
    AlwaysShowSign      = 0x10,
    ThousandsGroup      = 0x20,
    CapitalEforX        = 0x40,
    ShowBase            = 0x80,
    UppercaseBase       = 0x100,
    ForcePoint          = Alternate
};

enum PrecisionMode {
    PMDecimalDigits = 0x01,
    PMSignificaintDigits = 0x02,
    PMChopTrailingZeros = 0x03
};

static std::string ulltoa(unsigned long long int l)
{
    char buff[65];  // length of MAX_ULLONG in base 2
    char *p = buff + 65;
    char zero_ = '0';
    int base = 10;
    while (l != 0) {
        int c = l % base;
        *(--p) = zero_ + c;
        l /= base;
    }
    return std::string(p, 65 - (p - buff));
}

static std::string lltoa(long long int l)
{
    return ulltoa(l > 0 ? l : -l);
}

//
// see:
// github.com/radekp/qt/blob/master/src/corelib/tools/qlocale.cpp L:4000
//
static std::string longlong_to_string(
        long long int l, int precision, int /*width*/, unsigned flags)
{
    //    bool precision_not_specified = false;
    if (precision == -1) {
        //        precision_not_specified = true;
        precision = 1;
    }
    bool negative = l < 0;
    std::string num_str;
    num_str = lltoa(l);
    for (int i = (int) num_str.length(); i < precision; ++i)
    {
        num_str.insert(0, "0");
    }
    if (negative)
    {
        num_str.insert(0, "-");
    }
    else if (flags & AlwaysShowSign)
    {
        num_str.insert(0, "+");
    }
    else if (flags & BlankBeforePositive)
    {
        num_str.insert(0, " ");
    }
    return num_str;
}

static std::string decimal_form(
        std::string& digits, int decpt,
        unsigned int precision, PrecisionMode pm, bool always_show_decpt)
{
    if (decpt < 0)
    {
        for (int i = 0; i < -decpt; ++i)
        {
            digits.insert(0, "0");
        }
        decpt = 0;
    }
    else if (decpt > (int) digits.length())
    {
        for (int i = (int) digits.length(); i < decpt; ++i)
        {
            digits.append("0");
        }
    }

    if (pm == PMDecimalDigits)
    {
        unsigned int decimal_digits = (unsigned int) digits.length() - decpt;
        for (unsigned int i = decimal_digits; i < precision; ++i)
        {
            digits.append("0");
        }
    }
    else if (pm == PMSignificaintDigits)
    {
        for (unsigned int i = (unsigned int) digits.length(); i < precision; ++i)
        {
            digits.append("0");
        }
    }
    else
    { // pm == PMChopTrailingZeros
    }
    if (always_show_decpt || decpt < (int)digits.length())
    {
        digits.insert(decpt, ".");
    }
    if (decpt == 0) digits.insert(0, "0");
    if (digits[digits.length() - 1] == '.') digits.append("0");

    return digits;
}

static std::string exponent_form(
        std::string& digits, int decpt,
        unsigned int precision, PrecisionMode pm, bool always_show_decpt)
{
    int exp = decpt - 1;
    if (pm == PMDecimalDigits)
    {
        for (unsigned i = (unsigned) digits.length(); i < precision + 1; ++i)
        {
            digits.append("0");
        }
    }
    else if (pm == PMSignificaintDigits)
    {
        for (unsigned i = (unsigned) digits.length(); i < precision; ++i)
        {
            digits.append("0");
        }
    }
    else
    {
    }
    if (always_show_decpt || digits.length() > 1) digits.insert(1, ".");
    // Chop trailing 0's
    if (digits.length() > 0)
    {
        int last_nonzero_idx = (int) digits.length() - 1;
        while (last_nonzero_idx > 0 && digits[last_nonzero_idx] == '0')
        {
            --last_nonzero_idx;
        }
        digits = digits.substr(0, last_nonzero_idx + 1);
    }
    if (digits[digits.length() - 1] == '.') digits.append("0");
    digits.append("e");
    digits.append(longlong_to_string(exp, 2, -1, AlwaysShowSign));
    return digits;
}

static std::string double_to_string(
        double d, int precision, DoubleForm form, unsigned flags)
{
    std::string num_str;
    if (precision == -1) precision = 6;

    bool negative = false;
    bool special_number = false; // nan, +/- inf

    if (std::isinf(d))
    {
        num_str = "inf";
        special_number = true;
        negative = d < 0;
    }
    if (std::isnan(d))
    {
        num_str = "nan";
        special_number = true;
    }

    if (!special_number)
    {
        int decpt = 0, sign = 0;
        std::string digits;
        if (form == DFDecimal)
        {
            digits = std::string(fcvt(d, precision, &decpt, &sign));
        }
        else
        {
            int pr = precision;
            if (form == DFExponent)
            {
                ++pr;
            }
            else if (form == DFSignificantDigits && pr == 0)
            {
                pr = 1;
            }
            digits = std::string(ecvt(d, pr, &decpt, &sign));
            // Chop the trailing zeros.
            if (digits.length() > 0)
            {
                int last_nonzero_idx = (int) digits.length() - 1;
                while (last_nonzero_idx > 0 && digits[last_nonzero_idx] == '0')
                {
                    --last_nonzero_idx;
                }
                digits = digits.substr(0, last_nonzero_idx + 1);
            }
        }
        // char _zero = '0';

        // bool always_show_decpt = ((flags & Alternate) || (flags & ForcePoint));
        bool always_show_decpt = true;
        switch (form)
        {
            case DFExponent:
                num_str = exponent_form(digits, decpt, precision,
                                       PMDecimalDigits,
                                       always_show_decpt);
                break;
            case DFDecimal:
                num_str = decimal_form(digits, decpt, precision,
                                      PMDecimalDigits, always_show_decpt);
                // Chop the trailing zeros.
                if (num_str.length() > 0)
                {
                    int last_nonzero_idx = (int) num_str.length() - 1;
                    while (last_nonzero_idx > 0 && num_str[last_nonzero_idx] == '0')
                    {
                        --last_nonzero_idx;
                    }
                    num_str = num_str.substr(0, last_nonzero_idx + 1);
                    if (num_str[num_str.length() - 1] == '.')
                    {
                        num_str.append("0");
                    }
                }
                break;
            case DFSignificantDigits:
            {
                PrecisionMode mode = (flags & Alternate) ?
                                PMSignificaintDigits : PMChopTrailingZeros;
                if (decpt != static_cast<int>(digits.length()) &&
                                (decpt <= -4 || decpt > precision))
                {
                    num_str = exponent_form(digits, decpt, precision, mode,
                                           always_show_decpt);
                }
                else
                {
                    num_str = decimal_form(digits, decpt, precision, mode,
                                          always_show_decpt);
                }
                break;
            }
        };
        negative = sign != 0 && (d != 0.0 || d != -0.0);
    }

    if (negative) {
        num_str.insert(0, "-");
    } else if (flags & AlwaysShowSign) {
        num_str.insert(0, "+");
    } else if (flags & BlankBeforePositive) {
        num_str.insert(0, " ");
    }

    return num_str;
}

static int get_precision(double value)
{
    int n = 17;
    if (value != 0.0 && value > 1.0)
    {
        n -= (floor(log10(value)) + 1);
    }
    return n;
}


std::string oskar_settings_utility_double_to_string_2(double d, char format,
                                                      int precision)
{
    if (precision < 0)
    {
        switch (format)
        {
            case 'f':
                precision = get_precision(d);
                break;
            case 'e':
                precision = 15;
                break;
            case 'g':
            default:
                precision = 16;
        }
    }
    std::ostringstream ss;
    std::string s;
    DoubleForm f = DFSignificantDigits;
    switch (format)
    {
        case 'f':
            f = DFDecimal;
            break;
        case 'e':
            f = DFExponent;
            break;
        case 'g':
        default:
            f = DFSignificantDigits;
    }
    unsigned int flags = 0;
    s = double_to_string(d, precision, f, flags);
    return s;
}

double oskar_settings_utility_string_to_double(const std::string& s, bool *ok)
{
    char *endptr = 0;

    // TODO(BM) return nan on failure rather than 0?

    errno = 0;  // To distinguish success/failure after call
    double val = strtod(s.c_str(), &endptr);

    // If argument ok is not null, check for various possible errors.
    if (ok)
    {
        *ok = true;
        //  Check for various possible errors
        if ((errno != 0 && val == 0.0) ||
                        (errno == ERANGE && (val == HUGE_VAL ||
                            val == -HUGE_VAL)))
        {
            *ok = false;
        }
        // No digits found.
        if (endptr == s.c_str()) *ok = false;
        // Further characters found in the string
        // NOTE this may not actually be an error ...
        if (*endptr != '\0') *ok = false;
    }

    return val;
}

#if 0
// http://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
std::string oskar_format_string(const std::string fmt, ...)
{
    int size = 512;
    char* buffer = 0;
    buffer = new char[size];
    va_list vl;
    va_start(vl, fmt);
    int nsize = vsnprintf(buffer, size, fmt.c_str(), vl);
    if (size <= nsize)
    {
        // fail? delete buffer and try again
        delete[] buffer;
        buffer = 0;
        buffer = new char[nsize+1];  //+1 for /0
        nsize = vsnprintf(buffer, size, fmt.c_str(), vl);
    }
    std::string ret(buffer);
    va_end(vl);
    delete[] buffer;
    return ret;
}
#endif
