//  add_stop.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_sccs_add_stop_hpp
#define __ttl_sccs_add_stop_hpp

namespace ttl
{
namespace sccs
{
	namespace detail
	{
		///
		template< typename A >
		struct add_stop_visitor
		{
			typedef A agent;

			agent &a_;

			add_stop_visitor( agent& a ) 
				:a_(a)
			{
			}

			void operator()( const detail::sum_operator& )
			{
				a_.l_->stop();
				a_.r_->stop();
			}
			void operator()( const detail::prefix_operator& )
			{
				if(a_.r_)
				{
					a_.r_->stop();
				}
				else
				{
					a_.r_.reset( new agent(prefix_operator()) );
					a_.r_->l_.reset( new agent(detail::stop()) );
				}
			}
			void operator()( const detail::root& )
			{
				a_.c_ = detail::prefix_operator();
				a_.r_->l_.reset( new agent(detail::stop()) );
			}

			template< typename P >
			void operator()( const P&  )
			{
				throw std::runtime_error("unable to add stop");
			}
		};


	}; //detail

}; //sccs
}; //ttl

#endif //__ttl_sccs_add_stop_hpp

