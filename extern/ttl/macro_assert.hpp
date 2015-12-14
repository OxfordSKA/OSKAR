//  macro_assert.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_macro_assert__hpp
#define __ttl_macro_assert__hpp

namespace ttl
{
	template< bool C > struct assert {};
	
	template<>
	struct assert<true> 
	{
		typedef int type;
	};
};


#define TTL_ASSERT( c ) sizeof( ttl::assert<c>::type );

#endif //__macro_assert__hpp
