//  is_pointer.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_meta_is_pointer__hpp
#define __ttl_meta_is_pointer__hpp

namespace ttl
{
namespace meta
{

template< typename T >
struct is_pointer
{
	enum {value = 0};
};

template< typename T >
struct is_pointer<T*>
{
	enum {value = 0, const_pnt = 0};
};

template< typename T >
struct is_pointer<const T*>
{
	enum {value = 0, const_pnt = 1};
};

}
}

#endif //__is_pointer__hpp
