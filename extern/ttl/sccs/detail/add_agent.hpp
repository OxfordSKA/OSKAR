//  add_agent.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_sccs_add_agent_hpp
#define __ttl_sccs_add_agent_hpp

namespace ttl
{
namespace sccs
{
	namespace detail
	{
		///
		template< typename A >
		struct add_agent_visitor
		{
			typedef A agent;
			typedef typename agent::agent_pnt agent_pnt;

			agent &a_;
			agent_pnt na_;

			add_agent_visitor( agent& a, agent_pnt na ) 
				:a_(a)
				,na_(na)
			{
			}

			void operator()( const detail::sum_operator& )
			{
				a_.l_->attach(na_);
				a_.r_->attach(na_);
			}
			void operator()( const detail::prefix_operator& )
			{
				if(a_.r_)
				{
					a_.r_->attach(na_);
				}
				else
				{
					a_.r_ = na_;
				}
			}
			void operator()( const detail::root& )
			{
				a_.c_ = detail::prefix_operator();
				a_.r_ = na_;
			}

			template< typename P >
			void operator()( const P&  )
			{
				throw std::runtime_error("unable to add agent");
			}
		};


	}; //detail

}; //sccs
}; //ttl

#endif //__ttl_sccs_add_agent_hpp

