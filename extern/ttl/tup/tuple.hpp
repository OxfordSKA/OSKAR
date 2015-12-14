//  tuple.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_tup_tuple__hpp
#define __ttl_tup_tuple__hpp

#include "ttl/config.hpp"
#include "ttl/data_holder.hpp"
#include "ttl/meta/typelist.hpp"
#include "ttl/meta/is_reference.hpp"

namespace ttl
{
namespace tup
{
namespace impl
{

#define TTL_TUPLE_CALL_PARAM(n,p) typename tuple_param<types,N+n>::type p##n,
#define TTL_TUPLE_CALL_PARAM_END(n,p) typename tuple_param<types,N+n>::type p##n
#define TTL_TUPLE_CALL_PARAMS(n,p) TTL_REPEAT(n, TTL_TUPLE_CALL_PARAM, TTL_TUPLE_CALL_PARAM_END, p)
	
#define TTL_TUPLE_CTOR(n,p) tuple_data( TTL_TUPLE_CALL_PARAMS(n,p) ) : \
					head_(p1), \
					tail_( TTL_ARGS_SX(TTL_DEC(n),p) ) {}
	
		template< typename Types, int N, bool Stop = (N > meta::length<Types>::value) >
		struct tuple_param
		{
			typedef typename ttl::data_holder< typename meta::get<Types,N-1>::type >::param_type type;
		};
		
		template< typename Types, int N >
		struct tuple_param< Types, N, true >
		{
			typedef empty_type type;
		};
		
		template< typename Types, int N=0, bool Stop = (N >= meta::length<Types>::value) >
		struct tuple_data
		{
			typedef tuple_data this_t;
			typedef Types types;
			typename ttl::meta::get<types, N>::type head_;
			tuple_data<types, N+1> tail_;
			
			tuple_data() {}
			TTL_REPEAT_NEST( TTL_MAX_TUPLE_PARAMS, TTL_TUPLE_CTOR, TTL_TUPLE_CTOR, p )
			tuple_data( const this_t& r ) : head_(r.head_), tail_(r.tail_) {}
			
			this_t& operator=( const this_t& r )
			{
				if( &r == this ) return *this;
				head_ = r.head_;
				tail_ = r.tail_;
				return *this;
			}
		};	
		
		template< typename Types, int N >
		struct tuple_data<Types, N, true>
		{
			typedef tuple_data this_t;
			tuple_data() {}
			tuple_data( const this_t& r ) {}
			
			this_t& operator=( const this_t& r )
			{
				return *this;
			}
		};
		
#undef TTL_TUPLE_CALL_PARAM
#undef TTL_TUPLE_CALL_PARAM_END
#undef TTL_TUPLE_CALL_PARAMS
#undef TTL_TUPLE_CTOR
	
	template< typename R, int N >
	struct get_field
	{
		template< typename T >
		R operator()( T& t )
		{
			get_field<R,N-1> g;
			return g( t.tail_ );
		}
	};

	template< typename R >
	struct get_field<R, 0>
	{
		template< typename T >
		R operator()( T& t )
		{
			return t.head_ ;
		}
	};
	
};  //impl namespace


#define TTL_TUPLE_CALL_PARAM(n,p) typename tup::impl::tuple_param<types,n>::type p##n,
#define TTL_TUPLE_CALL_PARAM_END(n,p) typename tup::impl::tuple_param<types,n>::type p##n
#define TTL_TUPLE_CALL_PARAMS(n,p) TTL_REPEAT(n, TTL_TUPLE_CALL_PARAM, TTL_TUPLE_CALL_PARAM_END, p)

#define TTL_TUPLE_CTOR(n,p) tuple( TTL_TUPLE_CALL_PARAMS(n,p) ) : base(TTL_ARGSX(n,p)) {}


	template< TTL_TPARAMS_DEF(TTL_MAX_TUPLE_PARAMS, empty_type) >
	struct tuple : tup::impl::tuple_data< meta::typelist< TTL_ARGS(TTL_MAX_TUPLE_PARAMS) > >
	{
		typedef meta::typelist< TTL_ARGS(TTL_MAX_TUPLE_PARAMS) > types;
		typedef tup::impl::tuple_data< types > base;
		
		template<int N>
		struct return_type
		{
			typedef typename ttl::data_holder<typename meta::get<types, N>::type>::return_type type;
		};
		
		template<int N>
		struct const_return_type
		{
			typedef typename ttl::data_holder<typename meta::get<types, N>::type>::const_return_type type;
		};
		
		tuple() {}
		TTL_REPEAT_NEST( TTL_MAX_TUPLE_PARAMS, TTL_TUPLE_CTOR, TTL_TUPLE_CTOR, p )
	
		tuple( const base& r ) : base(r) {}

		tuple& operator=( const base& r ) 
		{ 
			base::operator=( r ); 
			return *this;
		}
	};
	
#undef TTL_TUPLE_CALL_PARAM
#undef TTL_TUPLE_CALL_PARAM_END
#undef TTL_TUPLE_CALL_PARAMS
#undef TTL_TUPLE_CTOR


template< typename T >
struct length
{
	enum
	{
		value = meta::length<typename T::types>::value
	};
};

template< int N, typename T >
struct element
{
	typedef typename meta::get<typename T::types, N>::type type;
};


template< int N, typename T >
typename selector
< 
	meta::is_reference<T&>::const_ref, 
	typename T::template const_return_type<N>::type, 
	typename T::template return_type<N>::type
>::type
get( T& t )
{
	typedef typename selector
	< 
		meta::is_reference<T&>::const_ref, 
		typename T::template const_return_type<N>::type, 
		typename T::template return_type<N>::type
	>::type return_type;
	
	impl::get_field< return_type, N > g;
	return g(t);
}


};
};

#endif //__tuple__hpp
