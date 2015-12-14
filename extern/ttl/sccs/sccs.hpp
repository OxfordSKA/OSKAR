//  sccs.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_sccs_sccs_hpp
#define __ttl_sccs_sccs_hpp

#include <iostream>
#include <map>
#include "ttl/var/variant.hpp"
#include "ttl/func/function.hpp"

#define BOOST_THREAD_BUILD_LIB
#include "boost/thread/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/thread/condition.hpp"
#include "boost/shared_ptr.hpp"

#include "ttl/sccs/detail/node_types.hpp"
#include "ttl/sccs/detail/add_port.hpp"
#include "ttl/sccs/detail/add_action.hpp"
#include "ttl/sccs/detail/add_repeat.hpp"
#include "ttl/sccs/detail/add_stop.hpp"
#include "ttl/sccs/detail/add_agent.hpp"
#include "ttl/sccs/detail/runtime_types.hpp"


namespace ttl
{
namespace sccs
{
namespace detail
{
	template <typename A> struct add_repeat_visitor;
	template <typename T> struct runtime_node;
	template < typename T > struct product_visitor;
	template< typename A, typename Act > struct add_action_item_visitor;
};

	namespace var = ::ttl::var;

	template< typename M >
	struct agent
	{
		typedef agent this_t;
		typedef M message;
		typedef boost::shared_ptr<message> message_pnt;

		typedef ttl::func::function<message_pnt (message_pnt)> action;

		typedef boost::shared_ptr<this_t> agent_pnt;
		typedef boost::shared_ptr< detail::in_port<message> > in_port_pnt;
		typedef boost::shared_ptr< detail::out_port<message> > out_port_pnt;
		typedef boost::shared_ptr< detail::action_item<action> > action_item_pnt;
		typedef boost::shared_ptr< detail::repeat<this_t> > repeat_pnt;


		/*
		typedef void (*simple_function)(void);
		struct simple_action
		{
			simple_action( simple_function f ) : f_(f) {}

			message_pnt operator()( message_pnt )
			{
				f_();
				return message_pnt();
			}

		private:
			simple_function f_;
		};

		typedef void (*simple_function1)( message_pnt );
		struct simple_action1
		{
			simple_action1( simple_function1 f ) : f_(f) {}

			message_pnt operator()( message_pnt m )
			{
				f_(m);
				return message_pnt();
			}

		private:
			simple_function1 f_;
		};

		typedef message_pnt (*simple_function_r)(void);
		struct simple_action_r
		{
			simple_action_r( simple_function_r f ) : f_(f) {}

			message_pnt operator()( message_pnt m )
			{
				return f_();
			}

		private:
			simple_function_r f_;
		};
		*/

		typedef var::variant<
			in_port_pnt
			,out_port_pnt
			,action_item_pnt
			,repeat_pnt

			,detail::sum_operator
			,detail::product_operator
			,detail::prefix_operator
			,detail::root
			,detail::stop
		> content;

		content c_;

		agent_pnt l_; //left
		agent_pnt r_; //right

		agent() : c_(detail::root()) 
		{}
		agent( const content& c ) : c_(c) 
		{}


			//input message of type P
		template< typename P >
		this_t* in( const P& d = P() )
		{
			detail::add_in_port_visitor<this_t, P>  v( *this, d );
			var::apply_visitor(v, c_);
			return this;
		}

			//output message or type P
		template< typename P >
		this_t* out( const P& d = P() )
		{
			detail::add_out_port_visitor<this_t, P>  v( *this, d );
			var::apply_visitor(v, c_);
			return this;
		}

			//action input parameters are
			// - the last message
			// - the list of agents that the action must be monitoring
			//   if one of the monitored agent becomes active (due to 
			//   an input message), the action must pass control to
			//   to that agent
		this_t* act( action d, const std::string& name = "" )
		{
			detail::add_action_item_visitor<this_t, action_item_pnt>
				v( *this, action_item_pnt(new detail::action_item<action>(d,name)) );
			var::apply_visitor(v, c_);
			return this;
		}
		/*
		this_t* act( simple_function f, const std::string& name = "" )
		{
			return act( simple_action(f), name );
		}
		this_t* act( simple_function1 f, const std::string& name = "" )
		{
			return act( simple_action1(f), name );
		}
		this_t* act( simple_function_r f, const std::string& name = "" )
		{
			return act( simple_action_r(f), name );
		}
		*/

			//repeat the agent
		this_t* repeat()
		{
			detail::add_repeat_visitor<this_t>  v( *this, *this );
			var::apply_visitor(v, c_);
			return this;
		}

		this_t* stop()
		{
			detail::add_stop_visitor<this_t>  v( *this );
			var::apply_visitor(v, c_);
			return this;
		}

		this_t* attach( agent_pnt a )
		{
			detail::add_agent_visitor<this_t>  v( *this, a );
			var::apply_visitor(v, c_);
			return this;
		}

		////////////////////////////////////////////////////////
		// PRINTING

		#if !defined(__USE_BOOST__)
		struct print_visitor
		#else
		struct print_visitor : boost::static_visitor<>
		#endif
		{
			std::ostream& out_;
			const agent &a_;
			print_visitor( std::ostream& out, const agent& a ) : out_(out), a_(a) {}

			void operator()( const in_port_pnt& d )
			{
				d->print(out_);
			}

			void operator()( const out_port_pnt d )
			{
				d->print(out_);
			}

			void operator()( const action_item_pnt d )
			{
				d->print(out_);
			}

			void operator()( const detail::sum_operator& d )
			{
				out_<<"+";
			}
			void operator()( const detail::product_operator& d )
			{
				out_<<"*";
			}
			void operator()( const detail::prefix_operator& d )
			{
				if( a_.r_ )
					out_<<":";
			}
			void operator()( const detail::root& d )
			{
				out_<<"[...]\n";
			}
			void operator()( const detail::stop& d )
			{
				out_<<"[STOP]\n";
			}
			void operator()( const repeat_pnt d )
			{
				out_<<"[REPEAT]\n";
			}
		};

		#if !defined(__USE_BOOST__)
		struct print_paren_visitor
		#else
		struct print_paren_visitor : boost::static_visitor<>
		#endif
		{
			bool o_;
			std::ostream &out_;

			print_paren_visitor( std::ostream& out, bool open ) : out_(out), o_(open) {}

			void operator()( const detail::sum_operator& d )
			{
				if( o_ )
					out_<<"(";
				else
					out_<<")";
			}
			void operator()( const detail::product_operator& d )
			{
				if( o_ )
					out_<<"(";
				else
					out_<<")";
			}

			template< typename T >
			void operator()( const T& d )
			{
			}
		};

		void print( std::ostream& out )
		{
			print_paren_visitor op(out, true);
			var::apply_visitor( op, c_ );

			if( l_ )
				l_->print( out );

			print_visitor visitor(out, *this);
			var::apply_visitor(visitor, c_);

			if( r_ )
				r_->print( out );

			print_paren_visitor cp(out, false);
			var::apply_visitor( cp, c_ );
		}
	private:
		this_t* add_act( action_item_pnt a )
		{
			detail::add_action_item_visitor<this_t, action_item_pnt>
				v( *this, a );
			var::apply_visitor(v, c_);
			return this;
		}

		void repeat( this_t& d )
		{
			detail::add_repeat_visitor<this_t>  v( *this, d );
			var::apply_visitor(v, c_);
		}

		//noncopyable
		agent( const this_t& r ) {}
		this_t& operator=( const this_t& r ) { return *this; }

		friend struct detail::add_repeat_visitor<agent>;
		friend struct detail::add_action_item_visitor<agent, action_item_pnt>;
	};

	template<typename A>
	A* start() { return new A(); }

	template< typename M >
	boost::shared_ptr< agent<M> > 
	operator + ( boost::shared_ptr< agent<M> > l, boost::shared_ptr< agent<M> > r )
	{
        boost::shared_ptr< agent<M> > tmp( new agent<M>( detail::sum_operator() ) );
		tmp->l_ = l;
		tmp->r_ = r;
		return tmp;
	}


	template< typename M >
	boost::shared_ptr< agent<M> > 
	operator * ( boost::shared_ptr< agent<M> > l, boost::shared_ptr< agent<M> > r )
	{
        boost::shared_ptr< agent<M> > tmp( new agent<M>( detail::product_operator() ) );
		tmp->l_ = l;
		tmp->r_ = r;
		return tmp;
	}

	struct boost_threads
	{
		typedef boost::thread thread;
		typedef boost::mutex mutex;
		typedef boost::condition condition;
		typedef boost::try_mutex try_mutex;
	};


	template< typename A, typename TT = boost_threads >
	struct runtime
	{
		typedef runtime this_t;

		typedef A agent;

		typedef boost::shared_ptr<agent> agent_pnt;

		typedef TT thread_traits;
		typedef typename thread_traits::thread thread;
		typedef typename thread_traits::mutex mutex;
		typedef typename thread_traits::try_mutex try_mutex;
		typedef typename thread_traits::condition condition;

		typedef typename agent::message message;
		typedef boost::shared_ptr<message> message_pnt;

		typedef detail::runtime_node<this_t> runtime_node;


		typedef boost::shared_ptr< runtime_node > node_pnt;
		typedef std::map<runtime_node*, node_pnt> nodes;

		runtime( agent_pnt a ) : a_(a)
		{
			add_runtime_node( a_.get() );
		}

		~runtime()
		{
			join();
		}

		void kill()
		{
			{
				typename mutex::scoped_lock g(m_);
				typename nodes::iterator it;
				for( it = n_.begin(); it != n_.end(); ++it )
				{
					it->second->kill();
				}
			}
			join();
		}

		void join()
		{
			typename mutex::scoped_lock g(m_);
			while( n_.size() )
			{
				c_.wait(g);
			}
		}

			// synch up a sent message with this message
		message_pnt synch_up( const message& m )
		{
			typename mutex::scoped_lock g(m_);
			del_.clear();

			typename nodes::iterator it;
			message_pnt out_msg;
			for( it = send_.begin(); it != send_.end(); ++it )
			{
				out_msg = it->second->synch_up(m);
				if( out_msg )
				{
					send_.erase( it );
					break;
				}
			}
			return out_msg;
		}

	private:
		agent_pnt a_;
		nodes n_;
		nodes del_;
		nodes send_; //sending nodes

		//sync stuff
		mutex m_;
		condition c_;

		void add_runtime_node( agent* a )
		{
			typename mutex::scoped_lock g(m_);
			del_.clear();
			node_pnt p(new runtime_node(*this,a));
			n_[p.get()] = p;
		}

		void send( runtime_node* n )
		{
			typename mutex::scoped_lock g(m_);
			del_.clear();

			//find the node
			typename nodes::iterator it;
			for( it = n_.begin(); it != n_.end(); ++it )
			{
				if( it->second.get() == n )
					break;
			}
			if( it == n_.end() )
				assert( 0 ); //node not found

			send_[n]=it->second;
			wakeup_all();
		}

		//must be the last call from the runtime_node
		void remove_node( runtime_node* n )
		{
			typename mutex::scoped_lock g(m_);
			typename nodes::iterator it;
			for( it = n_.begin(); it != n_.end(); ++it )
			{
				if( it->second.get() == n )
				{
					del_[n] = it->second;
					n_.erase(it);
					c_.notify_all();
					return;
				}
			}
			assert( 0 ); //node not found
		}

		void wakeup_all()
		{
			typename nodes::iterator it;
			for( it = n_.begin(); it != n_.end(); ++it )
			{
				it->second->wakeup();
			}
		}

		friend struct detail::runtime_node<this_t>;
		friend struct detail::product_visitor<this_t>;

		//noncopyable
		runtime( const runtime& r ) {}
		this_t& operator=( const this_t& r ) { return *this; }
	};

}; //sccs
}; //ttl


#endif //__ttl_sccs_sccs_hpp

