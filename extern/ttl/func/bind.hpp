//  bind.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_func_bind__hpp
#define __ttl_func_bind__hpp

#include "ttl/tup/tuple.hpp"

namespace ttl
{
namespace func
{

template< typename R, typename F, typename T = empty_type > struct functor;

//member function
template< typename R, typename F, typename T >
struct functor
{
	typedef R result_type;

	F f_;
	T* obj_;

	functor( F f, T* obj ) : f_(f), obj_(obj) {}

	R operator()() { return (obj_->*f_)(); }

	template< typename T1 >
	R operator()( T1 p1 ) { return (obj_->*f_)(p1); }

	template< typename T1, typename T2 >
	R operator()( T1 p1, T2 p2 ) { return (obj_->*f_)(p1, p2); }
};

//function pointer or another functor
template< typename R, typename F, typename T >
struct functor<R, F, empty_type>
{
	typedef R result_type;

	F f_;

	functor( F f ) : f_(f) {}

	R operator()() { return f_(); }

	template< typename T1 >
	R operator()( T1 p1 ) { return f_(p1); }

	template< typename T1, typename T2 >
	R operator()( T1 p1, T2 p2 ) { return f_(p1, p2); }
};

enum index
{
	_1 = 1,
	_2 = 2,
	_3 = 3
};

template< int N >
struct index_t { enum { value = N }; };

template< typename Index >
struct select_param;

template<>
struct select_param< index_t<1> >
{
	template< typename T1, typename T2 >	
	T1 operator()( T1 p1, T2 p2 )
	{
		return p1;
	}
};

template<>
struct select_param< index_t<2> >
{
	template< typename T1, typename T2 >	
	T2 operator()( T1 p1, T2 p2 )
	{
		return p2;
	}
};



template<
	typename Types,
	int N,
	bool IsBound = meta::is_same<position, typename tup::element<N,Bindings>::type>::value
>
struct select_param;

template<
	typename Types,
	int N
>
struct select_param<Types, N, true>
{
	typedef meta::get<Types, N>::type return_type;
	
	template< typename Bindings, typename T1, typename T2 >
	return_type operator()( Bindings& b, data_holder<T1>::param_type p1, data_holder<T2>::param_type p2 )
	{	
		return tup::get<N>(b);
	}
	
};



template< 
		typename Bindings, 
		typename Types, 
		int N, 
		bool IsBinding = is_base_and_derived<position, element<Bindings,N>::type >::value >
struct selecto;

template< typename Bindings, typename Types, int N, bool IsBinding >
struct selector<Bindings, Types, N, true>
{
	template< typename T1, typename T2 > 
	meta::get<Types, N>::type get( T

};

template< typename R, typename F, typename Bindings >
struct bind_t
{
	typedef Bindings bindings;

	functor<R, F, T> f_;
	bindings b_;

	bind_t( F f ) : f_(f) {}
	bind_t( F f, T* obj ) : f_(f, obj) {}

	R operator()() { return f_(); }

	template< typename T1 >
	R operator()( T1 p1 ) { return f_(p1); }

	template< typename T1, typename T2 >
	R operator()( T1 p1, T2 p2 ) { return f_(p1, p2); }
};

template< typename R, typename F >
bind_t<R, F> bind( F f )
{
	bind_t< R, F, tuple<> > b(f)
	return b;
}

template< typename R, typename F, typename T1 >
bind_t<R, F> bind( F f, T1 p1 )
{
	typedef tuple<T1> bindings;
	bind_t< R, F, bindings > b(f);
	get<0>(b.b_) = p1;
	return b;
}

template< typename R, typename F, typename T1, typename T2 >
bind_t<R, F> bind( F f, T1 p1, T2 p2 )
{
	typedef tuple<T1, T2> bindings;
	bind_t< R, F, bindings > b(f);
	get<0>(b.b_) = p1;
	get<1>(b.b_) = p2;
	return b;
}

};
};

#endif //__bind__hpp
