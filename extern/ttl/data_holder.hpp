//  data_holder.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

#ifndef __ttl_impl_data_holder__hpp
#define __ttl_impl_data_holder__hpp

namespace ttl
{
	struct data_holder_base {};
	
	template< typename T >
	struct data_holder : data_holder_base
	{
		enum { const_value = 0 };
		
		typedef T type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef T* pointer;
		typedef const T* const_pointer;

		typedef T& return_type;
		typedef const T& const_return_type;
		typedef const T& param_type;

		T d;

		data_holder() : d() {}
		data_holder( param_type d_ ) : d(d_) {}

		void set( param_type d_ ) { d = d_; }
		const_return_type get() const { return d; }
		return_type get() { return d; }
		const_pointer operator&() const { return &d; }
		pointer operator&() { return &d; }
		const_reference operator*() const { return d; }
		reference operator*() { return d; }
	};

	template< typename T >
	struct data_holder< const T > : data_holder_base
	{
		enum { const_value = 1 };
		
		typedef T type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef T* pointer;
		typedef const T* const_pointer;

		typedef const T& return_type;
		typedef const T& const_return_type;
		typedef const T& param_type;
		
		const T d;

		data_holder() : d() {}
		data_holder( param_type d_ ) : d(d_) {}

		const_return_type get() const { return d; }
		const_pointer operator&() const { return &d; }
		const_reference operator*() const { return d; }
	};

	template< typename T >
	struct data_holder<T&> : data_holder_base
	{
		enum { const_value = 0 };
		
		typedef T type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef T* pointer;
		typedef const T* const_pointer;

		typedef T& return_type;
		typedef const T& const_return_type;
		typedef T& param_type;

		T& d;

		data_holder( param_type d_ ) : d(d_) {}

		void set( param_type d_ ) { d = d_; }
		const_return_type get() const { return d; }
		return_type get() { return d; }
		const_pointer operator&() const { return &d; }
		pointer operator&() { return &d; }
		const_reference operator*() const { return d; }
		reference operator*() { return d; }
	};

	template< typename T >
	struct data_holder<const T&> : data_holder_base
	{
		enum { const_value = 0 };
		
		typedef T type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef T* pointer;
		typedef const T* const_pointer;

		typedef const T& return_type;
		typedef const T& const_return_type;
		typedef const T& param_type;

		const T& d;

		data_holder( param_type d_ ) : d(d_) {}

		void set( param_type d_ ) { d = d_; }
		const_return_type get() const { return d; }
		const_pointer operator&() const { return &d; }
		const_reference operator*() const { return d; }
	};

	template< typename T >
	struct data_holder<T*> : data_holder_base
	{
		enum { const_value = 0 };
		
		typedef T type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef T* pointer;
		typedef const T* const_pointer;

		typedef T* return_type;
		typedef const T* const_return_type;
		typedef T* param_type;

		T* d;

		data_holder() : d(0) {}
		data_holder( param_type d_ ) : d(d_) {}

		void set( param_type d_ ) { d = d_; }
		const_return_type get() const { return d; }
		return_type get() { return d; }
		const_pointer operator&() const { return d; }
		pointer operator&() { return d; }
		const_reference operator*() const { return *d; }
		reference operator*() { return *d; }
	};

	template<>
	struct data_holder<void*> : data_holder_base
	{
		enum { const_value = 0 };
		
		typedef void type;
		typedef void* pointer;

		typedef void* return_type;
		typedef void* param_type;

		void* d;

		data_holder() : d(0) {}
		data_holder( param_type d_ ) : d(d_) {}

		void set( param_type d_ ) { d = d_; }
		return_type get() { return d; }
		pointer operator&() { return d; }
	};
	
	template< typename T >
	struct data_holder<const T*> : data_holder_base
	{
		enum { const_value = 0 };
		
		typedef T type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef T* pointer;
		typedef const T* const_pointer;

		typedef const T* return_type;
		typedef const T* const_return_type;
		typedef const T* param_type;

		const T* d;

		data_holder() : d(0) {}
		data_holder( param_type d_ ) : d(d_) {}

		void set( param_type d_ ) { d = d_; }
		const_return_type get() const { return d; }
		const_pointer operator&() const { return d; }
		const_reference operator*() const { return *d; }
	};
	
}

#endif //__data_holder__hpp
