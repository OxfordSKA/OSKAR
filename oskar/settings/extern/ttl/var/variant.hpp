//  variant.hpp
//
//  Copyright (c) 2003 Eugene Gladyshev
//
//  Permission to copy, use, modify, sell and distribute this software
//  is granted provided this copyright notice appears in all copies.
//  This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//

//*************
// variant cannot have 'const' types
// since we are using data_holder, the alignment should not be an issue...?

#ifndef __ttl_variant__hpp
#define __ttl_variant__hpp

#include "settings/extern/ttl/meta/typelist.hpp"
#include "settings/extern/ttl/meta/is_reference.hpp"
#include "settings/extern/ttl/meta/is_pointer.hpp"
#include "settings/extern/ttl/meta/is_same.hpp"

namespace ttl
{
namespace var
{
struct exception : ttl::exception
{
    exception() : ttl::exception("variant error") {}
};

namespace impl
{
    template< typename F, typename V >
    struct variant_binary_visitor
    {
        template<typename T1>
        struct visitor_functor
        {
            F& f_;
            T1 p1_;
            visitor_functor( F& f, T1 p1 ) : f_(f), p1_(p1) {}
            template < typename T2 >
            void operator()( T2 p2 ) { f_(p1_, p2); }
        };

        F& f_;
        V v_;

        variant_binary_visitor( F& f, V v ): f_(f), v_(v) {}

        template< typename T >
        void operator()( T& p )
        {
            visitor_functor<T> f(f_,p);
            apply_visitor(f, v_);
        }
    };

    template< typename F, typename V >
    struct visitor_adapter_ref
    {
        F& f_;
        V v_;
        visitor_adapter_ref( F& f, V v ) : f_(f), v_(v) {}

        template< typename T >
        void operator()()
        {
            f_( v_.template get_holder<T>()->get() );
        }

    };

    template< typename V >
    struct copier
    {
        V& l_;
        const V& r_;

        copier( V& l, const V& r ) : l_(l), r_(r) {}

        template<typename T>
        void operator()()
        {
            if( r_.is_singular() )
            {
                l_.destroy();
                return;
            }
            const ttl::data_holder<T>* rd = static_cast<const ttl::data_holder<T>*>(r_.pnt_);
            l_.destroy();
            l_.pnt_ = new(l_.stor_.buf_) ttl::data_holder<T>(rd->d);
            l_.which_ = r_.which_;
        }
    };

    template<typename T, typename V>
    struct non_const_data
    {
        enum
        {
            value = !meta::is_reference<T&>::const_ref && !meta::is_same<T,V>::value
        };
    };

}  //impl namespace


///////////////////////////////////////////////////////////////////////
    template< TTL_TPARAMS_DEF(TTL_MAX_TYPELIST_PARAMS, empty_type) >
    struct variant
    {
        typedef variant this_t;
        typedef meta::typelist< TTL_ARGS(TTL_MAX_TYPELIST_PARAMS) > list;

        enum
        {
            size = sizeof(ttl::data_holder<typename list::largest_type>)
        };

        variant() : which_(0), pnt_(0)
        {
        }

        variant( const this_t& r ) : which_(0), pnt_(0)
        {
            impl::copier<this_t> c(*this, r);
            meta::type_switch<list> ts;
            ts( r.which(), c );
        }

        template< typename T >
        variant( const T& r ) : which_(0), pnt_(0)
        {
            typedef meta::find_equivalent_type<const T&, list> found;
            pnt_ = new(stor_.buf_) ttl::data_holder<typename found::type>(r);
            which_ = found::index;
        }

#if 0 // Note: Removed to fix a compiler warning - 25Apr15

        //GCC v3.2.3 can't handle T& and const T& overloads
        //to workaround, try SFINAE
        // *as far as I know, a similar method is used in boost::variant as well
        template<typename T>
        variant(T& r,
            typename selector<
                    impl::non_const_data<T, this_t>::value,
                    int,
                    int&  //substitution will fail here (pointer to reference is illegal)
                >::type* p = 0
         ) : which_(0), pnt_(0)
        {
            typedef meta::find_equivalent_type< T&, list> found;
            pnt_ = new(stor_.buf_) ttl::data_holder<typename found::type>(r);
            which_ = found::index;
        }

#endif
        ~variant() { destroy(); }

        this_t& operator=( const this_t& r )
        {
            if( this == &r ) return *this;
            impl::copier<this_t> c(*this, r);
            meta::type_switch<list> ts;
            ts( r.which(), c );
            return *this;
        }

        inline size_t get_types() const { return list::length; }
        inline int which() const { return which_; }
        inline bool is_singular() const { return pnt_ == 0; }

        template< typename T >
        const ttl::data_holder<T>* get_holder() const
        {
            return static_cast<const ttl::data_holder<T>*>(pnt_);
        }

        template< typename T >
        ttl::data_holder<T>* get_holder()
        {
            return static_cast<ttl::data_holder<T>*>(pnt_);
        }

    private:
        template<int N>
        struct storage
        {
            union
            {
                ttl::data_holder_base dummy;  //hope to satisfy alignment settings
                char buf_[N];
            };
        };

        int which_;
        ttl::data_holder_base* pnt_;
        storage< sizeof(ttl::data_holder<typename list::largest_type>) > stor_;

        struct deleter
        {
            ttl::data_holder_base *p_;
            deleter( ttl::data_holder_base *p ) : p_(p) {}

            template<typename T>
            void operator()()
            {
                typedef ttl::data_holder<T> var_type;
                static_cast<var_type*>(p_)->~var_type();
            }
        };


        void destroy()
        {
            if( !pnt_ ) return;
            ttl::data_holder_base *p = pnt_;
            pnt_ = 0;
            deleter d(p);
            meta::type_switch<list> ts;
            ts(which_, d);
        }

        friend struct impl::copier<this_t>;
    };

    template< typename T, typename V >
    typename selector
    <
        meta::is_pointer<V*>::const_pnt,
        typename ttl::data_holder<T>::const_pointer,
        typename ttl::data_holder<T>::pointer
    >::type
    get( V* v )
    {
        typedef meta::find_equivalent_type<T, typename V::list> found;
        if( v->which() != found::index ) return 0;
        return &(v->template get_holder<typename found::type>()->get());
    }

    template< typename T, typename V >
    typename selector
        <
            meta::is_reference<V&>::const_ref,
            typename ttl::data_holder<T>::const_return_type,
            typename ttl::data_holder<T>::return_type
        >::type
    get( V& v )
    {
        typedef meta::find_equivalent_type<T, typename V::list> found;
        if( v.which() != found::index ) throw var::exception();
        return v.template get_holder<typename found::type>()->get();
    }

    template< typename F, typename V >
    void apply_visitor( F& f, V& v )
    {
        if( v.is_singular() ) return;
        meta::type_switch<typename V::list> ts;
        impl::visitor_adapter_ref<F, V&> va(f, v);
        ts(v.which(), va);
    }

    template< typename F, typename V1, typename V2 >
    void apply_visitor( F& f, V1& v1, V2& v2 )
    {
        impl::variant_binary_visitor<F,V2&> bv(f,v2);
        apply_visitor(bv, v1);
    }

}
}

#endif //__variant__hpp
