//  add_repeat.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_sccs_add_repeat_hpp
#define __ttl_sccs_add_repeat_hpp

namespace ttl
{
namespace sccs
{
	namespace detail
	{
		///
		template< typename A >
		struct add_repeat_visitor
		{
			typedef A agent;

			agent &a_;
			agent &d_; //repeat entry

			add_repeat_visitor( agent& a, agent& d ) 
				:a_(a)
				,d_(d)
			{
			}

			void operator()( const detail::sum_operator& )
			{
				a_.l_->repeat(d_);
				a_.r_->repeat(d_);
			}
			void operator()( const detail::prefix_operator& )
			{
				if(a_.r_)
				{
					a_.r_->repeat(d_);
				}
				else
				{
					a_.r_.reset( new agent(prefix_operator()) );
					typename agent::repeat_pnt c( new repeat<agent>(d_) );
					a_.r_->l_.reset( new agent(c) );
				}
			}
			void operator()( const detail::root& )
			{
				a_.c_ = detail::prefix_operator();
				typename agent::repeat_pnt c( new detail::repeat<agent>(d_) );
				a_.l_.reset( new agent(c) );
			}

			template< typename P >
			void operator()( const P&  )
			{
				throw std::runtime_error("unable to add repeat");
			}
		};


	}; //detail

}; //sccs
}; //ttl

#endif //__ttl_sccs_add_repeat_hpp

