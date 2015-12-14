//  collection.hpp
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

#ifndef __ttl_collection__hpp
#define __ttl_collection__hpp

namespace ttl
{

#define TTL_COLLECTION_ARG(n,t) const T& t##n,
#define TTL_COLLECTION_ARG_END(n,t) const T& t##n

#define TTL_COLLECTION_ITEM(n,t) push_back(t##n);
#define TTL_COLLECTION_ITEM_END(n,t) push_back(t##n);

#define TTL_COLLECTION_ARGS(n) TTL_REPEAT(n, TTL_COLLECTION_ARG, TTL_COLLECTION_ARG_END, p)
#define TTL_COLLECTION_ITEMS(n) TTL_REPEAT(n, TTL_COLLECTION_ITEM, TTL_COLLECTION_ITEM_END, p)

	template< typename T, typename C = std::vector<T> >
	struct collection : C
	{
		typedef C container;
				
		collection( ) {}
		
		collection( TTL_COLLECTION_ARGS(1) ) 
		{
			reserve(1);
			TTL_COLLECTION_ITEMS(1)
		}
		collection( TTL_COLLECTION_ARGS(2) ) 
		{
			reserve(2);
			TTL_COLLECTION_ITEMS(2)
		}
		collection( TTL_COLLECTION_ARGS(3) ) 
		{
			reserve(3);
			TTL_COLLECTION_ITEMS(3)
		}
		collection( TTL_COLLECTION_ARGS(4) ) 
		{
			reserve(4);
			TTL_COLLECTION_ITEMS(4)
		}
		collection( TTL_COLLECTION_ARGS(5) ) 
		{
			reserve(5);
			TTL_COLLECTION_ITEMS(5)
		}
		collection( TTL_COLLECTION_ARGS(6) ) 
		{
			reserve(6);
			TTL_COLLECTION_ITEMS(6)
		}
		collection( TTL_COLLECTION_ARGS(7) ) 
		{
			reserve(7);
			TTL_COLLECTION_ITEMS(7)
		}
		collection( TTL_COLLECTION_ARGS(8) ) 
		{
			reserve(8);
			TTL_COLLECTION_ITEMS(8)
		}
		collection( TTL_COLLECTION_ARGS(9) ) 
		{
			reserve(9);
			TTL_COLLECTION_ITEMS(9)
		}
		collection( TTL_COLLECTION_ARGS(10) ) 
		{
			reserve(10);
			TTL_COLLECTION_ITEMS(10)
		}
		collection( TTL_COLLECTION_ARGS(11) ) 
		{
			reserve(11);
			TTL_COLLECTION_ITEMS(11)
		}
		collection( TTL_COLLECTION_ARGS(12) ) 
		{
			reserve(12);
			TTL_COLLECTION_ITEMS(12)
		}
		collection( TTL_COLLECTION_ARGS(13) ) 
		{
			reserve(13);
			TTL_COLLECTION_ITEMS(13)
		}
		collection( TTL_COLLECTION_ARGS(14) ) 
		{
			reserve(14);
			TTL_COLLECTION_ITEMS(14)
		}
		collection( TTL_COLLECTION_ARGS(15) ) 
		{
			reserve(15);
			TTL_COLLECTION_ITEMS(15)
		}
	};
};

#endif //__ttl_collection__hpp
