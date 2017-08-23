//  config.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_config__hpp
#define __ttl_config__hpp

#include <new>
#include <stdexcept>
#include <functional>
#include <memory>
#include <string>

#if defined(_MSC_VER)
#	define TTL_MAX_TEMPLATE_PARAMS 25
#elif defined(__GNUC__)
#	define TTL_MAX_TEMPLATE_PARAMS 25
#else
#	define TTL_MAX_TEMPLATE_PARAMS 25
#endif

#define TTL_MAX_TYPELIST_PARAMS TTL_MAX_TEMPLATE_PARAMS
#define TTL_MAX_TUPLE_PARAMS 15
#define TTL_MAX_NAMED_PARAMS TTL_MAX_TUPLE_PARAMS

namespace ttl
{
#if defined(UNICODE)||defined(_UNICODE)
	typedef std::wstring string;
	#define SC( txt ) L txt
#else
	typedef std::string string;
	#define SC( txt ) txt
#endif

	typedef string::value_type tchar;


	struct empty_type {};
}


#endif //__ttl_config__hpp
