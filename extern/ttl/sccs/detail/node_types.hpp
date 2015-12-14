//  node_types.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_sccs_node_types_hpp
#define __ttl_sccs_node_types_hpp

namespace ttl
{
namespace sccs
{
	namespace detail
	{

		template< typename T >
		struct in_port
		{
			typedef T message; //message variant
			message m_;
			in_port( const message& m ) : m_(m) {}

			virtual void print(std::ostream& out) = 0;
		};

		template< typename V, typename T >
		struct in_port_complete : in_port<V>
		{
			typedef T type;
			in_port_complete( const type& m ) : in_port<V>(m) {}

			virtual void print( std::ostream& out )
			{
				out<<typeid(type).name()<<"?";
			}
		};


		template< typename T >
		struct out_port
		{
			typedef T message; //message variant

			message m_;
			out_port( const message& m ) : m_(m) {}

			virtual void print( std::ostream& out ) = 0;
		};

		template< typename V, typename T >
		struct out_port_complete : out_port<V>
		{
			typedef T type;

			out_port_complete( const type& m ) : out_port<V>(m) {}

			virtual void print( std::ostream& out )
			{
				out<<typeid(type).name()<<"!";
			}
		};


		template< typename T >
		struct action_item
		{
			typedef T action;
			action a_;
			std::string n_;

			action_item( action a, const std::string& name ) 
				:a_(a)
				,n_(name)
			{}

			void print(std::ostream& out)
			{
				out<<n_.c_str();
			}
		};

		struct sum_operator
		{};

		struct product_operator
		{};

		struct prefix_operator
		{};

		struct root
		{};

		struct stop
		{};

		template< typename A >
		struct repeat
		{
			typedef A agent;
			agent &a_;
			repeat( A& a ) : a_(a) {}
		};

	}; //detail

}; //sccs
}; //ttl

#endif //__ttl_sccs_node_types_hpp


