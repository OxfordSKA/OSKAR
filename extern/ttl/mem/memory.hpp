//  memory.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_memory__hpp
#define __ttl_memory__hpp

#include "ttl/config.hpp"
#include "ttl/exception.hpp"

namespace ttl
{
namespace mem
{
	template< typename A >
	typename A::pointer create()
	{
		A a;
		typename A::pointer p = a.allocate(1,0);
		try
		{
			new( p ) typename A::value_type;
		}
		catch( ttl::exception& )
		{
			a.deallocate(p,1);
			throw;
		}
		catch( std::exception& )
		{
			a.deallocate(p,1);
			throw;
		}
		return p;
	}

	template< typename A >
	typename A::pointer create( const typename A::value_type& v )
	{
		A a;
		typename A::pointer p = a.allocate(1,0);
		try
		{
			new( p ) typename A::value_type(v);
		}
		catch( ttl::exception& )
		{
			a.deallocate(p,1);
			throw;
		}
		catch( std::exception& )
		{
			a.deallocate(p,1);
			throw;
		}
		return p;
	}
	
	template< typename A, typename P1 >
	typename A::pointer create( P1 p1 )
	{
		A a;
		typename A::pointer p = a.allocate(1,0);
		try
		{
			new( p ) typename A::value_type(p1);
		}
		catch( ttl::exception& )
		{
			a.deallocate(p,1);
			throw;
		}
		catch( std::exception& )
		{
			a.deallocate(p,1);
			throw;
		}
		return p;
	}
	
	template< typename A, typename P1, typename P2 >
	typename A::pointer create( P1 p1, P2 p2 )
	{
		A a;
		typename A::pointer p = a.allocate(1,0);
		try
		{
			new( p ) typename A::value_type(p1, p2);
		}
		catch( ttl::exception& )
		{
			a.deallocate(p,1);
			throw;
		}
		catch( std::exception& )
		{
			a.deallocate(p,1);
			throw;
		}
		return p;
	}
	
	template< typename A >
	void destroy( typename A::pointer p )
	{
		A a;
		typedef typename A::value_type type;
		try
		{
			p->~type();
		}
		catch( ttl::exception& )
		{
			a.deallocate(p,1);
			throw;
		}
		catch( std::exception& )
		{
			a.deallocate(p,1);
			throw;
		}
		a.deallocate(p,1);
	}
	
	struct memtraits
	{
		template< typename T >
		struct allocator : std::allocator<T>
		{
		};
	};
};
};

#endif //__memory__hpp
