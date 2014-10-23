//  is_reference.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_is_reference__hpp
#define __ttl_is_reference__hpp

namespace ttl
{
namespace meta
{
	template< typename T >
	struct is_reference
	{
		enum { value = 0 };
	};
	
	template< typename T >
	struct is_reference<T&>
	{
		enum 
		{ 
			value = 1,
			const_ref = 0
		};
	};
	
	template< typename T >
	struct is_reference<const T&>
	{
		enum 
		{ 
			value = 1,
			const_ref = 1
		};
	};
}
}

#endif //__is_reference__hpp
