//  exception.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_exception__hpp
#define __ttl_exception__hpp

#include "settings/extern/ttl/config.hpp"

namespace ttl
{
    struct exception : std::runtime_error
    {
        exception() : std::runtime_error("ttl error") {}
        exception(const char* msg) :  std::runtime_error(msg) {}
    };
}

#endif //__exception__hpp
