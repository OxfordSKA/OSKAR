//  equivalent_type.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_equivalent_type__hpp
#define __ttl_equivalent_type__hpp

namespace ttl
{
	template< typename T1, typename T2 >
	struct equivalent_types
	{
		enum { value = 0 };
	};		
	
	template< typename T >
	struct equivalent_types<T,T>
	{
		enum { value = 1 };
	};		
	
	template< typename T > 
	struct equivalent_types<T&,T>
	{
		enum { value = 1 };
	};
	template< typename T > 
	struct equivalent_types<T,T&>
	{
		enum { value = 1 };
	};
	template< typename T > 
	struct equivalent_types<T,const T&>
	{
		enum { value = 1 };
	};
	template< typename T > 
	struct equivalent_types<const T&, T&>
	{
		enum { value = 1 };
	};
}

#endif //__ttl_equivalent_type__hpp
