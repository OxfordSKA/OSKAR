//  add_action.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_sccs_add_action_item_hpp
#define __ttl_sccs_add_action_item_hpp

namespace ttl
{
namespace sccs
{
	namespace detail
	{
		///
		template< typename A, typename Act >
		struct add_action_item_visitor
		{
			typedef A agent;
			typedef typename agent::message message;
			typedef Act action;
			typedef typename message::list message_types;

			agent &a_;
			action act_;

			add_action_item_visitor( agent& a, action act ) 
				: a_(a)
				, act_(act)
			{
			}

			void operator()( const detail::sum_operator& )
			{
				a_.l_->add_act(act_);
				a_.r_->add_act(act_);
			}
			void operator()( const detail::prefix_operator& )
			{
				if(a_.r_)
				{
					a_.r_->add_act(act_);
				}
				else
				{
					a_.r_.reset( new agent(prefix_operator()) );
					a_.r_->l_.reset( new agent(act_) );
				}
			}
			void operator()( const detail::root& )
			{
				a_.c_ = detail::prefix_operator();
				a_.l_.reset( new agent(act_) );
			}

			template< typename P >
			void operator()( const P&  )
			{
				throw std::runtime_error("unable to add the action");
			}
		};


	}; //detail

}; //sccs
}; //ttl

#endif //__ttl_sccs_add_action_item_hpp

