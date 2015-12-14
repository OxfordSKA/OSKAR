//  runtime_types.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_sccs_runtime_types_hpp
#define __ttl_sccs_runtime_types_hpp

#include "boost/bind.hpp"

namespace ttl
{
namespace sccs
{
	namespace detail
	{
		template < typename T >
		struct synch_up_visitor;

		template < typename T >
		struct send_visitor;

		template < typename T >
		struct product_visitor;

		template < typename T >
		struct action_visitor;

		template< typename T >
		struct runtime_node
		{
			typedef runtime_node this_t;
			typedef T runtime;

			typedef typename runtime::agent agent;
			typedef agent* agent_pnt;
			typedef typename agent::message message;
			typedef boost::shared_ptr<message> message_pnt;

			typedef typename runtime::thread thread;
			typedef typename runtime::condition condition;
//			typedef typename runtime::try_mutex try_mutex;
			typedef typename runtime::mutex mutex;

			typedef boost::shared_ptr<thread> thread_pnt;

			runtime_node( runtime& rt, agent_pnt a ) 
				:rt_(rt)
				,a_(a)
				,s_(transition)
				,kill_(false)
			{
				t_.reset( new thread( boost::bind(&this_t::proc, get_this()) ) );
			}
			~runtime_node()
			{
				t_->join();
			}
			this_t* get_this() { return this; }

			bool kill()
			{
				kill_ = true;
				c_.notify_all();
				return true;
			}

			void wakeup()
			{
				c_.notify_all();
			}

			message_pnt synch_up( const message& in )
			{
				typename mutex::scoped_lock g(m_);

				message_pnt out;

				if( !out_ )
					return out;

				assert( get_state() == wait );

				// same message types
				if( out_->which() != in.which() )
					return out;

				out = out_;
				out_.reset();

				set_state( transition );
				c_.notify_all();
				return out;
			}


		private:
			enum state
			{
				wait
				,transition
				,exit
			};

			runtime& rt_;
			agent_pnt a_;
			state s_;
//			mutex sm_; //state mutex
			bool kill_;
			thread_pnt t_;

			message_pnt out_;
			message_pnt in_;

			mutex m_;
			condition c_; //wake up and process the message

			void set_state( state s )
			{
//				mutex::scoped_lock g( sm_ );
				s_ = s;
			}

			state get_state()
			{
//				mutex::scoped_lock g( sm_ );
				return s_;
			}

			state do_transition()
			{
				if( out_ ) //waiting to synch up
					return wait;

				if( !a_ )  //empty agent
					return exit;  //exit the thread

				if( do_synchup() )
				{
					return transition;
				}

				if( do_product() || do_action() )
				{
					in_.reset();
					return transition;
				}

				return wait;
			}

			bool do_product()
			{
				product_visitor<runtime> v( rt_, a_ );
				var::apply_visitor( v, a_->c_ );
				if( v.found_ )
				{
					a_ = a_->r_.get();  //move to next
				}
				return v.found_;
			}

			bool do_action()
			{
				action_visitor<this_t> v( a_, in_ );
				var::apply_visitor( v, a_->c_ );

				if( !v.found_ )
					return false;

				if( v.out_ )
				{
					out_ = v.out_;
					rt_.send( this ); //request to synch up
				}

				a_ = v.next_;
				return true;
			}

			bool do_synchup()
			{
				synch_up_visitor<this_t> v( rt_, a_ );
				var::apply_visitor( v, a_->c_ );

				if( !v.found_ )
					return false;

				in_ = v.in_;

				a_ = v.next_;
				return true;
			}

			void proc()
			{
				try
				{
					do_proc();
				}
				catch( const std::exception& )
				{
					assert(0);
				}
			}

			void do_proc()
			{
				typename mutex::scoped_lock g(m_);
				bool loop = true;
				while( loop )
				{
					if( kill_ )
					{
						rt_.remove_node( this );
						loop = false;
						continue;
					}
					switch( get_state() )
					{
					case wait:  //wait for the next message
						c_.wait(g);
						set_state( transition );
						break;

					case transition:
						set_state( do_transition() ); // transition to the next agent
						break;

					case exit:
						// remove the node from runtime
						rt_.remove_node( this );
						loop = false;
						break;
					}
				}
			}
		};


		template < typename T >
		struct is_outport_visitor
		{
			typedef T runtime_node;
			typedef typename runtime_node::agent agent;
			typedef typename agent::message message;
			typedef typename agent::out_port_pnt out_port_pnt;

			const message* m_;

			is_outport_visitor()
				: m_(0)
			{}

			void operator()( const out_port_pnt& p )
			{
				m_ = &(p->m_);
			}

			template< typename P > void operator()( const P&  )
			{}
		};

		template < typename T >
		struct find_outport_visitor
		{
			typedef T runtime_node;
			typedef typename runtime_node::runtime runtime;
			typedef typename runtime_node::agent agent;
			typedef agent* agent_pnt;
			typedef typename agent::message message;
			typedef boost::shared_ptr<message> message_pnt;

			agent_pnt next_; //the next agent in the agent tree
			bool found_;

			find_outport_visitor( agent_pnt a, message_pnt m ) 
				:next_(0)
				,found_(false)
				,a_(a)
				,m_(m)
			{
			}

			void operator()( const detail::prefix_operator& )
			{
				if( !a_->l_ )
				{
					next_ = a_->r_.get();
					found_ = true;
					return;
				}
				is_outport_visitor<runtime_node> v;
				var::apply_visitor(v, a_->l_->c_);
				if( !v.m_ )
					return;

				//found out_port

					//compare message types
				if( m_->which() == v.m_->which() )
				{
					found_ = true;
					next_ = a_->r_.get();
				}
			}

			void operator()( const detail::sum_operator& )
			{
				assert( a_->l_ && a_->r_ );
				find_outport_visitor<runtime_node> v( a_->l_.get(), m_ );
				var::apply_visitor(v, a_->l_->c_);
				if( !v.found_ ) //not found in the left branch
				{
					//try the right branch
					find_outport_visitor<runtime_node> v( a_->r_.get(), m_ );
					var::apply_visitor(v, a_->r_->c_);
					next_ = v.next_;
					found_ = v.found_;
				}
				else
				{
					next_ = v.next_;
					found_ = v.found_;
				}
			}

			template< typename P > void operator()( const P&  )
			{}

		private:
			agent_pnt a_; //start the search from this agent
			message_pnt m_;

		};


		template < typename T >
		struct is_inport_visitor
		{
			typedef T runtime_node;
			typedef typename runtime_node::agent agent;
			typedef typename agent::message message;
			typedef typename agent::in_port_pnt in_port_pnt;

			const message* m_;

			is_inport_visitor()
				: m_(0)
			{}

			void operator()( const in_port_pnt& p )
			{
				m_ = &(p->m_);
			}
			template< typename P > void operator()( const P&  )
			{}
		};

		template < typename T >
		struct synch_up_visitor
		{
			typedef T runtime_node;
			typedef typename runtime_node::runtime runtime;
			typedef typename runtime_node::agent agent;
			typedef agent* agent_pnt;
			typedef typename agent::message message;
			typedef boost::shared_ptr<message> message_pnt;

			agent_pnt next_; //the next agent in the agent tree
			bool found_;
			message_pnt in_;

			synch_up_visitor( runtime& rt, agent_pnt a ) 
				:next_(0)
				,found_(false)
				,rt_(rt)
				,a_(a)
			{
			}

			void operator()( const detail::prefix_operator& )
			{
				if( !a_->l_ )
				{
					next_ = a_->r_.get();
					found_ = true;
					return;
				}

				is_inport_visitor<runtime_node> v;
				var::apply_visitor(v, a_->l_->c_);
				if( !v.m_ )
					return;

				//found in_port

					//try synch up
				in_ = rt_.synch_up( *v.m_ );
				if( in_ )
				{
					found_ = true;
					next_ = a_->r_.get();
				}
			}

			void operator()( const detail::sum_operator& )
			{
				assert( a_->l_ && a_->r_ );
				synch_up_visitor<runtime_node> v( rt_, a_->l_.get() );
				var::apply_visitor(v, a_->l_->c_);
				if( !v.found_ ) //not found in the left branch
				{
					//try the right branch
					synch_up_visitor<runtime_node> v( rt_, a_->r_.get() );
					var::apply_visitor(v, a_->r_->c_);
					next_ = v.next_;
					found_ = v.found_;
					in_ = v.in_;
				}
				else
				{
					next_ = v.next_;
					found_ = v.found_;
					in_ = v.in_;
				}
			}

			template< typename P > void operator()( const P&  )
			{}

		private:
			runtime& rt_;
			agent_pnt a_; //start the search from this agent

		};



		template < typename T >
		struct product_visitor
		{
			typedef T runtime;
			typedef typename runtime::agent agent;
			typedef agent* agent_pnt;

			bool found_;

			product_visitor( runtime &rt, agent_pnt a )
				:found_(false)
				,rt_(rt)
				,a_(a)
			{}

			void operator()( const detail::product_operator& )
			{
				rt_.add_runtime_node( a_->l_.get() );
				found_ = true;
			}

			template< typename P >
			void operator() ( const P& p )
			{}

		private:
			runtime &rt_;
			agent_pnt a_;
		};

		template < typename T >
		struct prefix_action_visitor
		{
			typedef T runtime_node;
			typedef typename runtime_node::agent agent;
			typedef agent* agent_pnt;
			typedef typename agent::message message;
			typedef boost::shared_ptr<message> message_pnt;

			typedef typename agent::in_port_pnt in_port_pnt;
			typedef typename agent::out_port_pnt out_port_pnt;
			typedef typename agent::action_item_pnt action_item_pnt;
			typedef typename agent::repeat_pnt repeat_pnt;

			agent_pnt next_; //repeat node
			bool found_;
			message_pnt out_;

			prefix_action_visitor( agent_pnt a, message_pnt in )
				:next_(0)
				,found_(false)
				,a_(a)
				,in_(in)
			{}

			void operator()( const out_port_pnt& p )
			{
				out_.reset(new message(p->m_));
				next_ = a_->r_.get();
				found_ = true;
			}

			void operator()( const action_item_pnt& p )
			{
				out_ = p->a_(in_);
				next_ = a_->r_.get();
				if( out_ )
				{
					//find the output port
					if( next_ )
					{
						find_outport_visitor<runtime_node> v( a_->r_.get(), out_ );
						var::apply_visitor(v, a_->r_->c_);
						//if an action is generating a message
						//the message type should be listed as possible transition
						assert( v.found_ );
						next_ = v.next_;
					}
				}
				found_ = true;
			}

			void operator()( const repeat_pnt& p )
			{
				next_ = &(p->a_);
				found_ = true;
			}

			void operator()( const stop& )
			{
				next_ = 0;
				found_ = true;
			}

			template< typename P >
			void operator()( const P&  )
			{
			}

		private:
			agent_pnt a_;
			message_pnt in_;
		};


		template < typename T >
		struct action_visitor
		{
			typedef T runtime_node;
			typedef typename runtime_node::agent agent;
			typedef agent* agent_pnt;
			typedef typename agent::message message;
			typedef boost::shared_ptr<message> message_pnt;

			agent_pnt next_; //the next agent in the agent tree
			bool found_;
			message_pnt out_;

			action_visitor( agent_pnt a, message_pnt in ) 
				:next_(0)
				,found_(false)
				,a_(a)
				,in_(in)
			{
			}

			void operator()( const detail::prefix_operator& )
			{
				if( !a_->l_ )
				{
					next_ = a_->r_.get();
					found_ = true;
					return;
				}

				prefix_action_visitor<runtime_node> v( a_, in_ );
				var::apply_visitor(v, a_->l_->c_);
				next_ = v.next_;
				out_ = v.out_;
				found_ = v.found_;
			}

			void operator()( const detail::sum_operator& )
			{
				assert( a_->l_ && a_->r_ );
				action_visitor<runtime_node> v( a_->l_.get(), in_ );
				var::apply_visitor(v, a_->l_->c_);
				if( !v.found_ ) //not found in the left branch
				{
					//try the right branch
					action_visitor<runtime_node> v( a_->r_.get(), in_ );
					var::apply_visitor(v, a_->r_->c_);
					next_ = v.next_;
					out_ = v.out_;
					found_ = v.found_;
				}
				else
				{
					next_ = v.next_;
					out_ = v.out_;
					found_ = v.found_;
				}
			}

			template< typename P >
			void operator()( const P&  )
			{
			}


		private:
			agent_pnt a_; //start the search from this agent
			message_pnt in_;
		};


	}; //detail

}; //sccs
}; //ttl

#endif //runtime_types
