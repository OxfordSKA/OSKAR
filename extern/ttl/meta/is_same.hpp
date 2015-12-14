//  is_same.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_meta_is_same__hpp
#define __ttl_meta_is_same__hpp

namespace ttl
{
namespace meta
{
	template< typename T1, typename T2 >
	struct is_same
	{
		enum { value = 0 };
	};

	template< typename T >
	struct is_same<T, T>
	{
		enum { value = 1 };
	};
}
}

#endif //__is_same__hpp
