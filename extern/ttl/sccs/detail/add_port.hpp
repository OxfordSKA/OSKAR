//  add_port.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_sccs_add_port_hpp
#define __ttl_sccs_add_port_hpp

namespace ttl
{
namespace sccs
{
	namespace detail
	{

		///
		template< typename A, typename T >
		struct add_in_port_visitor
		{
			typedef A agent;
			typedef typename agent::message message;
			typedef T type; //port type
			typedef typename message::list message_types;

			agent &a_;
			const type& d_;
			add_in_port_visitor( agent& a, const type& d ) : a_(a), d_(d)
			{
			}

			void operator()( const detail::sum_operator& )
			{
				a_.l_->template in<type>(d_);
				a_.r_->template in<type>(d_);
			}
			void operator()( const detail::prefix_operator& )
			{
				if(a_.r_)
				{
					a_.r_->template in<type>(d_);
				}
				else
				{
					a_.r_.reset( new agent(prefix_operator()) );
					typename agent::in_port_pnt c( new in_port_complete<message,type>(d_) );
					a_.r_->l_.reset( new agent(c) );
				}
			}
			void operator()( const detail::root& )
			{
				a_.c_ = detail::prefix_operator();
				typename agent::in_port_pnt c( new in_port_complete<message,type>(d_) );
				a_.l_.reset( new agent(c) );
			}

			template< typename P >
			void operator()( const P&  )
			{
				throw std::runtime_error("unable to add the input port");
			}
		};

		///
		template< typename A, typename T >
		struct add_out_port_visitor
		{
			typedef A agent;
			typedef typename agent::message message;
			typedef T type; //port type
			typedef typename message::list message_types;

			agent &a_;
			const type& d_;

			add_out_port_visitor( agent& a, const type& d ) : a_(a), d_(d)
			{
			}

			void operator()( const detail::sum_operator& )
			{
				a_.l_->template out<type>(d_);
				a_.r_->template out<type>(d_);
			}
			void operator()( const detail::prefix_operator& )
			{
				if(a_.r_)
				{
					a_.r_->template out<type>(d_);
				}
				else
				{
					a_.r_.reset( new agent(prefix_operator()) );
					typename agent::out_port_pnt c( new out_port_complete<message,type>(d_) );
					a_.r_->l_.reset( new agent(c) );
				}
			}
			void operator()( const detail::root& )
			{
				a_.c_ = detail::prefix_operator();
				typename agent::out_port_pnt c( new out_port_complete<message,type>(d_) );
				a_.l_.reset( new agent(c) );
			}

			template< typename P >
			void operator()( const P&  )
			{
				throw std::runtime_error("unable to add the input port");
			}
		};


	}; //detail

}; //sccs
}; //ttl

#endif //__ttl_sccs_add_port_hpp

