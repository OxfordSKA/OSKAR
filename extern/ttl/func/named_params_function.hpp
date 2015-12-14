//  named_params_function.hpp
//
//  Copyright (c) 2004 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#if defined(_MSC_VER)
#pragma once
#endif

#ifndef __ttl_func_named_params_function__hpp
#define __ttl_func_named_params_function__hpp

#include "ttl/tup/tuple.hpp"
#include "ttl/meta/is_same.hpp"
#include "ttl/data_holder.hpp"
#include "ttl/func/function.hpp"

namespace ttl
{
namespace func
{
namespace detail
{
	//find index of the field of the specified type
	//in the tuple
	
	template< 
		  typename Tup //tuple
		, typename T //field to find
		, int Ind = 0
		, bool Same =
			ttl::meta::is_same
			<
				  typename tup::element<Ind,Tup>::type::name
				, T
			>::value
	>
	struct find
	{
		enum
		{
			value = find<Tup,T,Ind+1>::value
		};
	};

	template< typename Tup, typename T, int Ind > 
	struct find< Tup, T, Ind, true>
	{
		enum 
		{
			value = Ind
		};
	};

	//
	template< typename T, typename Name >
	struct get_type
	{
		typedef typename tup::element<find<T, Name>::value, T>::type tmp_type;
		typedef typename tmp_type::type type;
	};

	//
	template< 
		typename Name, //name of the function argument
		typename T //argument type
	>
	struct named_pair
	{
		typedef Name name;
		typedef T type;

		data_holder<type> d_;

		named_pair( typename ttl::data_holder<type>::param_type d ) : d_(d) {}
	};

	//just a pointer to named_pair<>
	template< 
		typename P //named_pair<...> type
	>
	struct named_pair_pnt
	{
		typedef P pair;
		typedef typename pair::name name;
		typedef typename pair::type type;

		const pair* p_;

		named_pair_pnt() : p_(0) {}
		named_pair_pnt( const pair* p ) : p_(p) {}
	};

	//	
	template <typename Pair, typename T=typename Pair::type>
	struct argument
	{
		typedef T type;

		type get( const Pair* p )
		{
			return p?p->d_.d:type();
		}

	};

	template <typename Pair, typename T>
	struct argument<Pair, const T&>
	{
		typedef T type;

		type *d_;

		argument() : d_(0) {}

		virtual ~argument()
		{
			if( !d_ ) return;
			delete d_;
		}
		const type& get( const Pair* p )
		{
			if( !p ) 
			{
				d_ = new type();
				return *d_;
			}

			return p->d_.d;
		}
	};

	template <typename Pair, typename T>
	struct argument<Pair, T&>
	{
		typedef T type;

		type *d_;

		argument() : d_(0) {}

		virtual ~argument()
		{
			if( !d_ ) return;
			delete d_;
		}
		type& get( const Pair* p )
		{
			if( !p ) 
			{
				d_ = new type();
				return *d_;
			}

			return p->d_.d;
		}
	};
}; //detail

/*
The following macros are expanded into a set of following types

template < 
	typename R, //result type
	typename N1, typename T1, //name and type pairs
	typename N2, typename T2, ... 
>
struct named_params_function< R (T1, T2...) >
{ 
	typedef named_params_function this_t;
	typedef func::function< R (T1, T2,...) > functor;
	typedef R result_type; 
	
	typedef T1 arg1;
	typedef T2 arg2;
	...
	
	
	typedef tup::tuple<
		  detail::named_pair_pnt< detail::named_pair<N1,arg1> >	
		, detail::named_pair_pnt< detail::named_pair<N2,arg2> >	
		  ...
	> data_tuple;
	
	
	functor f_;
	
	named_params_function()  {}
	named_params_function( functor f ) : f_(f) {}
	
	this_t& operator=( functor f )
	{
		f_ = f;
		return *this;
	}
	
	result_type operator()()
	{
		data_tuple d;
		return f_( TTL_FUNC_CALL_ARGS(n) );
	}

	template< typename Name >
	static detail::named_pair<Name, typename detail::get_type<data_tuple,Name>::type>
	arg( typename data_holder<typename detail::get_type<data_tuple,Name>::type>::param_type d )
	{
		return detail::named_pair<Name, typename detail::get_type<data_tuple,Name>::type>(d);
	}
	
	template< typename P1 > 
	result_type operator()( const P1& a1 )  
	{
		data_tuple d;
		//set tuple values
		tup::get< detail::find< data_tuple, typename P1::name >::value >(d) = detail::named_pair_pnt<P1>(&a1);		
		
		//use the tuple values for calling the function		
		return f_( 
			  detail::argument< detail::named_pair<N1, T1> >().get(tup::get<0>(d).p_)
			, detail::argument< detail::named_pair<N2, T2> >().get(tup::get<1>(d).p_)
			  ...
		); 
	}
	
	template< typename P1, typename P2 > 
	result_type operator()( const P1& a1, const P2& a2 )  
	{
		data_tuple d;
		//set tuple values
		tup::get< detail::find< data_tuple, typename P1::name >::value >(d) = detail::named_pair_pnt<P1>(&a1);
		tup::get< detail::find< data_tuple, typename P2::name >::value >(d) = detail::named_pair_pnt<P1>(&a2);

		//use the tuple values for calling the function		
		return f_( 
			  detail::argument< detail::named_pair<N1, T1> >().get(tup::get<0>(d).p_)
			, detail::argument< detail::named_pair<N2, T2> >().get(tup::get<1>(d).p_)
			  ...
		); 
	}
	
	...

};



*/

	#define TTL_FUNC_PAIR_DER(n,p) typename N##n = empty_type, typename T##n = empty_type,
	#define TTL_FUNC_PAIR_DEF_END(n,p) typename N##n = empty_type, typename T##n = empty_type
	#define TTL_FUNC_PAIR_DEF_PARAMS(n) TTL_REPEAT(n, TTL_FUNC_PAIR_DER, TTL_FUNC_PAIR_DEF_END, p)

	template < 
		  typename R
		, TTL_FUNC_PAIR_DEF_PARAMS( TTL_MAX_NAMED_PARAMS )
	> struct named_params_function {};



	#define TTL_FUNC_PAIR(n,p) typename N##n, typename T##n,
	#define TTL_FUNC_PAIR_END(n,p) typename N##n, typename T##n
	#define TTL_FUNC_PAIR_PARAMS(n) TTL_REPEAT(n, TTL_FUNC_PAIR, TTL_FUNC_PAIR_END, p)

	#define TTL_FUNC_PAIR_ARG(n,p) N##n, T##n,
	#define TTL_FUNC_PAIR_ARG_END(n,p) N##n, T##n
	#define TTL_FUNC_PAIR_ARGS(n) TTL_REPEAT(n, TTL_FUNC_PAIR_ARG, TTL_FUNC_PAIR_ARG_END, p)

	#define TTL_FUNC_ARG_TYPE(n,p) typedef T##n arg##n;
	#define TTL_FUNC_ARG_TYPE_END(n,p) typedef T##n arg##n;
	#define TTL_FUNC_ARG_TYPES(n) TTL_REPEAT(n, TTL_FUNC_ARG_TYPE, TTL_FUNC_ARG_TYPE_END, p)

	#define TTL_FUNC_TUPLE_TYPE(n,p) detail::named_pair_pnt< detail::named_pair<N##n,arg##n> >,
	#define TTL_FUNC_TUPLE_TYPE_END(n,p) detail::named_pair_pnt< detail::named_pair<N##n,arg##n> >
	#define TTL_FUNC_TUPLE_TYPES(n) TTL_REPEAT(n, TTL_FUNC_TUPLE_TYPE, TTL_FUNC_TUPLE_TYPE_END, p)

	#define TTL_FUNC_CALL_ARG(n,p) detail::argument< detail::named_pair<N##n, T##n> >().get(tup::get<TTL_DEC(n)>(d).p_),
	#define TTL_FUNC_CALL_ARG_END(n,p) detail::argument< detail::named_pair<N##n, T##n> >().get(tup::get<TTL_DEC(n)>(d).p_)
	#define TTL_FUNC_CALL_ARGS(n) TTL_REPEAT(n, TTL_FUNC_CALL_ARG, TTL_FUNC_CALL_ARG_END, p)

	#define TTL_FUNC_OPERATOR_ARG(n,p) const P##n& a##n,
	#define TTL_FUNC_OPERATOR_ARG_END(n,p) const P##n& a##n
	#define TTL_FUNC_OPERATOR_ARGS(n) TTL_REPEAT(n, TTL_FUNC_OPERATOR_ARG, TTL_FUNC_OPERATOR_ARG_END, p)

	#define TTL_FUNC_TUPLE_ARG(n,p) tup::get< detail::find< data_tuple, typename P##n::name >::value >(d) = detail::named_pair_pnt<P##n>(&a##n);
	#define TTL_FUNC_TUPLE_ARG_END(n,p) tup::get< detail::find< data_tuple, typename P##n::name >::value >(d) = detail::named_pair_pnt<P##n>(&a##n);
	#define TTL_FUNC_TUPLE_ARGS(n) TTL_REPEAT(n, TTL_FUNC_TUPLE_ARG, TTL_FUNC_TUPLE_ARG_END, p)


	#define TTL_FUNC_OPERATOR(n, TotalArg)  \
		template< TTL_TPARAMSX(n,P) >  \
		result_type operator()( TTL_FUNC_OPERATOR_ARGS(n) )  {  \
			data_tuple d;  \
			TTL_FUNC_TUPLE_ARGS(n)  \
			return f_( TTL_FUNC_CALL_ARGS(TotalArg) ); }
	

	#define TTL_FUNC_TEMPLATE_DECLARATION(n)  \
		template < typename R, TTL_FUNC_PAIR_PARAMS(n) >  \
		struct named_params_function< R ( TTL_FUNC_PAIR_ARGS(n) ) >  \
		{  \
			typedef named_params_function this_t;  \
			typedef func::function< R (TTL_ARGS(n)) > functor;  \
			typedef R result_type;  \
		\
			TTL_FUNC_ARG_TYPES(n)  \
		\
			typedef tup::tuple<  \
				TTL_FUNC_TUPLE_TYPES(n)  \
			> data_tuple; \
		\
			functor f_; \
		\
			named_params_function()  {} \
			named_params_function( functor f ) : f_(f) {} \
		\
			this_t& operator=( functor f ) \
			{ \
				f_ = f; \
				return *this; \
			} \
		\
			result_type operator()()  \
			{ \
				data_tuple d; \
				return f_( TTL_FUNC_CALL_ARGS(n) ); \
			} \
		\
			template< typename Name > \
			static detail::named_pair<Name, typename detail::get_type<data_tuple,Name>::type> \
			arg( typename data_holder<typename detail::get_type<data_tuple,Name>::type>::param_type d ) \
			{ \
				return detail::named_pair<Name, typename detail::get_type<data_tuple,Name>::type>(d); \
			}

	//repeat following blocks TTL_MAX_NAMED_PARAMS times

	TTL_FUNC_TEMPLATE_DECLARATION(1)
		TTL_FUNC_OPERATOR(1,1)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(2)
		TTL_FUNC_OPERATOR(1,2)
		TTL_FUNC_OPERATOR(2,2)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(3)
		TTL_FUNC_OPERATOR(1,3)
		TTL_FUNC_OPERATOR(2,3)
		TTL_FUNC_OPERATOR(3,3)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(4)
		TTL_FUNC_OPERATOR(1,4)
		TTL_FUNC_OPERATOR(2,4)
		TTL_FUNC_OPERATOR(3,4)
		TTL_FUNC_OPERATOR(4,4)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(5)
		TTL_FUNC_OPERATOR(1,5)
		TTL_FUNC_OPERATOR(2,5)
		TTL_FUNC_OPERATOR(3,5)
		TTL_FUNC_OPERATOR(4,5)
		TTL_FUNC_OPERATOR(5,5)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(6)
		TTL_FUNC_OPERATOR(1,6)
		TTL_FUNC_OPERATOR(2,6)
		TTL_FUNC_OPERATOR(3,6)
		TTL_FUNC_OPERATOR(4,6)
		TTL_FUNC_OPERATOR(5,6)
		TTL_FUNC_OPERATOR(6,6)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(7)
		TTL_FUNC_OPERATOR(1,7)
		TTL_FUNC_OPERATOR(2,7)
		TTL_FUNC_OPERATOR(3,7)
		TTL_FUNC_OPERATOR(4,7)
		TTL_FUNC_OPERATOR(5,7)
		TTL_FUNC_OPERATOR(6,7)
		TTL_FUNC_OPERATOR(7,7)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(8)
		TTL_FUNC_OPERATOR(1,8)
		TTL_FUNC_OPERATOR(2,8)
		TTL_FUNC_OPERATOR(3,8)
		TTL_FUNC_OPERATOR(4,8)
		TTL_FUNC_OPERATOR(5,8)
		TTL_FUNC_OPERATOR(6,8)
		TTL_FUNC_OPERATOR(7,8)
		TTL_FUNC_OPERATOR(8,8)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(9)
		TTL_FUNC_OPERATOR(1,9)
		TTL_FUNC_OPERATOR(2,9)
		TTL_FUNC_OPERATOR(3,9)
		TTL_FUNC_OPERATOR(4,9)
		TTL_FUNC_OPERATOR(5,9)
		TTL_FUNC_OPERATOR(6,9)
		TTL_FUNC_OPERATOR(7,9)
		TTL_FUNC_OPERATOR(8,9)
		TTL_FUNC_OPERATOR(9,9)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(10)
		TTL_FUNC_OPERATOR(1,10)
		TTL_FUNC_OPERATOR(2,10)
		TTL_FUNC_OPERATOR(3,10)
		TTL_FUNC_OPERATOR(4,10)
		TTL_FUNC_OPERATOR(5,10)
		TTL_FUNC_OPERATOR(6,10)
		TTL_FUNC_OPERATOR(7,10)
		TTL_FUNC_OPERATOR(8,10)
		TTL_FUNC_OPERATOR(9,10)
		TTL_FUNC_OPERATOR(10,10)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(11)
		TTL_FUNC_OPERATOR(1,11)
		TTL_FUNC_OPERATOR(2,11)
		TTL_FUNC_OPERATOR(3,11)
		TTL_FUNC_OPERATOR(4,11)
		TTL_FUNC_OPERATOR(5,11)
		TTL_FUNC_OPERATOR(6,11)
		TTL_FUNC_OPERATOR(7,11)
		TTL_FUNC_OPERATOR(8,11)
		TTL_FUNC_OPERATOR(9,11)
		TTL_FUNC_OPERATOR(10,11)
		TTL_FUNC_OPERATOR(11,11)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(12)
		TTL_FUNC_OPERATOR(1,12)
		TTL_FUNC_OPERATOR(2,12)
		TTL_FUNC_OPERATOR(3,12)
		TTL_FUNC_OPERATOR(4,12)
		TTL_FUNC_OPERATOR(5,12)
		TTL_FUNC_OPERATOR(6,12)
		TTL_FUNC_OPERATOR(7,12)
		TTL_FUNC_OPERATOR(8,12)
		TTL_FUNC_OPERATOR(9,12)
		TTL_FUNC_OPERATOR(10,12)
		TTL_FUNC_OPERATOR(11,12)
		TTL_FUNC_OPERATOR(12,12)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(13)
		TTL_FUNC_OPERATOR(1,13)
		TTL_FUNC_OPERATOR(2,13)
		TTL_FUNC_OPERATOR(3,13)
		TTL_FUNC_OPERATOR(4,13)
		TTL_FUNC_OPERATOR(5,13)
		TTL_FUNC_OPERATOR(6,13)
		TTL_FUNC_OPERATOR(7,13)
		TTL_FUNC_OPERATOR(8,13)
		TTL_FUNC_OPERATOR(9,13)
		TTL_FUNC_OPERATOR(10,13)
		TTL_FUNC_OPERATOR(11,13)
		TTL_FUNC_OPERATOR(12,13)
		TTL_FUNC_OPERATOR(13,13)
	};


	TTL_FUNC_TEMPLATE_DECLARATION(14)
		TTL_FUNC_OPERATOR(1,14)
		TTL_FUNC_OPERATOR(2,14)
		TTL_FUNC_OPERATOR(3,14)
		TTL_FUNC_OPERATOR(4,14)
		TTL_FUNC_OPERATOR(5,14)
		TTL_FUNC_OPERATOR(6,14)
		TTL_FUNC_OPERATOR(7,14)
		TTL_FUNC_OPERATOR(8,14)
		TTL_FUNC_OPERATOR(9,14)
		TTL_FUNC_OPERATOR(10,14)
		TTL_FUNC_OPERATOR(11,14)
		TTL_FUNC_OPERATOR(12,14)
		TTL_FUNC_OPERATOR(13,14)
		TTL_FUNC_OPERATOR(14,14)
	};

	TTL_FUNC_TEMPLATE_DECLARATION(15)
		TTL_FUNC_OPERATOR(1,15)
		TTL_FUNC_OPERATOR(2,15)
		TTL_FUNC_OPERATOR(3,15)
		TTL_FUNC_OPERATOR(4,15)
		TTL_FUNC_OPERATOR(5,15)
		TTL_FUNC_OPERATOR(6,15)
		TTL_FUNC_OPERATOR(7,15)
		TTL_FUNC_OPERATOR(8,15)
		TTL_FUNC_OPERATOR(9,15)
		TTL_FUNC_OPERATOR(10,15)
		TTL_FUNC_OPERATOR(11,15)
		TTL_FUNC_OPERATOR(12,15)
		TTL_FUNC_OPERATOR(13,15)
		TTL_FUNC_OPERATOR(14,15)
		TTL_FUNC_OPERATOR(15,15)
	};

	//clean up
	#undef TTL_FUNC_PAIR_DER
	#undef TTL_FUNC_PAIR_DEF_END
	#undef TTL_FUNC_PAIR_DEF_PARAMS

	#undef TTL_FUNC_PAIR
	#undef TTL_FUNC_PAIR_END
	#undef TTL_FUNC_PAIR_PARAMS

	#undef TTL_FUNC_PAIR_ARG
	#undef TTL_FUNC_PAIR_ARG_END
	#undef TTL_FUNC_PAIR_ARGS

	#undef TTL_FUNC_ARG_TYPE
	#undef TTL_FUNC_ARG_TYPE_END
	#undef TTL_FUNC_ARG_TYPES

	#undef TTL_FUNC_TUPLE_TYPE
	#undef TTL_FUNC_TUPLE_TYPE_END
	#undef TTL_FUNC_TUPLE_TYPES

	#undef TTL_FUNC_CALL_ARG
	#undef TTL_FUNC_CALL_ARG_END
	#undef TTL_FUNC_CALL_ARGS

	#undef TTL_FUNC_OPERATOR_ARG
	#undef TTL_FUNC_OPERATOR_ARG_END
	#undef TTL_FUNC_OPERATOR_ARGS

	#undef TTL_FUNC_TUPLE_ARG
	#undef TTL_FUNC_TUPLE_ARG_END
	#undef TTL_FUNC_TUPLE_ARGS

	#undef TTL_FUNC_OPERATOR

	#undef TTL_FUNC_TEMPLATE_DECLARATION
	
	
	template< typename T, T Def >
	struct numeric_argument
	{
		T d_;
		numeric_argument( const T& d = Def ) : d_(d)
		{
		}
		operator T() { return d_; }
	};

	template<typename T>
	struct argument
	{
		T d_;
		argument( T d = T() ) : d_(d) {}
		operator T() { return d_; }
	};
	

}; //func
}; //ttl

#endif //__ttl_func_named_params_function__hpp
