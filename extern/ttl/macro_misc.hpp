//  macro_misc.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_macro_misc__hpp
#define __ttl_macro_misc__hpp

#define TTL_RET(x) x
#define TTL_APPEND( x, y ) TTL_APPEND1(x,y)
#define TTL_APPEND1( x, y ) x ## y


#endif //__macro_misc__hpp
