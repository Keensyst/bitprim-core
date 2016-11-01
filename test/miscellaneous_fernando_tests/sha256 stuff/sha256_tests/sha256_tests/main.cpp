//// clang++ -O3 -std=c++1y merkle_reduction.cpp
//// g++ -O3 -std=c++11 merkle_reduction.cpp
//
//
//#include <iterator>
//
//// Concepts
//#define Integer typename
//#define Container typename
//#define Iterator typename
//#define ForwardIterator typename
//#define BinaryOperation typename
//#define Semiregular typename
//#define UnaryFunction typename
//
//#define requires(...)
//
////TODO:
//// Mutable<I>
//// Domain<Op>
//
//// Type Attributes / Type Functions
//template <Iterator I>
//using ValueType = typename std::iterator_traits<I>::value_type;
//
//
//// -----------------------------------------------------------------
//
//template<Integer I>
//bool even(I const& a) {
//    return (a bitand I(1)) == I(0);
//}
//
//// -----------------------------------------------------------------
//
////template <BinaryOperation Op>
////struct transpose_operation {
////	const Op op;
////
////	explicit
////		transpose_operation(Op op)
////		: op{ op }
////	{}
////
////	template <typename T>
////	requires(T == Domain<Op>)
////		T operator()(T const& a, T const& b) const {
////		return op(b, a);
////	}
////};
//
//
//// -----------------------------------------------------------------
//// Algorithms (Generic)
//// -----------------------------------------------------------------
//
//template <Iterator I, BinaryOperation Op, UnaryFunction F>
//requires(I == Domain<F> && Codomain<F> == Domain<Op> && ValueType<I> == Domain<Op>)
//ValueType<I> reduce_nonempty(I f, I l, Op op, F fun) {
//    // precondition: bounded_range(f, l) and f != l and
//    //               partially_associative(op) and
//    //               ForAll x in [f, l), fun(x) is defined
//    
//    ValueType<I> r = fun(f);
//    ++f;
//    while (f != l) {
//        r = op(r, fun(f));
//        ++f;
//    }
//    return r;
//}
//
//template <Iterator I, BinaryOperation Op>
//requires(ValueType<I> == Domain<Op>)
//ValueType<I> reduce_nonempty(I f, I l, Op op) {
//    // precondition: bounded_range(f, l) and f != l and
//    //               partially_associative(op) and
//    
//    ValueType<I> r = *f;
//    ++f;
//    while (f != l) {
//        r = op(r, *f);
//        ++f;
//    }
//    return r;
//}
//
//
//template <Iterator I, BinaryOperation Op, UnaryFunction F>
//requires(I == Domain<F> && Codomain<F> == Domain<Op> && ValueType<I> == Domain<Op>) //TODO...
//ValueType<I> reduce_nonzeroes(I f, I l, Op op, F fun, ValueType<I> const& e) {
//    // precondition: bounded_range(f, l) and
//    //               partially_associative(op) and
//    //               ForAll x in [f, l), fun(x) is defined
//    
//    ValueType<I> x;
//    do {
//        if (f == l) return e;
//        x = fun(f);
//        ++f;
//    } while (x == e);
//    
//    while (f != l) {
//        ValueType<I> y = fun(f);
//        if (y != e) x = op(x, y);
//        ++f;
//    }
//    return x;
//}
//
//template <Iterator I, BinaryOperation Op>
//requires(ValueType<I> == Domain<Op>) //TODO...
//ValueType<I> reduce_nonzeroes(I f, I l, Op op, ValueType<I> const& e) {
//    // precondition: bounded_range(f, l) and
//    //               partially_associative(op) and
//    
//    ValueType<I> x;
//    do {
//        if (f == l) return e;
//        x = *f;
//        ++f;
//    } while (x == e);
//    
//    while (f != l) {
//        // ValueType<I> y = *f;
//        // ValueType<I> const& y = *f;
//        // if (y != e) x = op(x, y);
//        if (*f != e) x = op(x, *f);
//        ++f;
//    }
//    return x;
//}
//
//
//// Counter Machine
//// template <ForwardIterator I, BinaryOperation Op>
//// add_to_counter(I f, I l, Op op, Domain<Op> x, )
//
//
//template <ForwardIterator I, BinaryOperation Op>
//// 	TODO: requires Op is associative
//requires(Mutable<I> && ValueType<I> == Domain<Op>)
//ValueType<I> add_to_counter(I f, I l, Op op, ValueType<I> x, ValueType<I> const& e) {
//    // precondition: x != e
//    while (f != l) {
//        if (*f == e) {
//            *f = x;
//            return e;
//        }
//        x = op(*f, x);
//        *f = e;
//        ++f;
//    }
//    return x;
//}
//
//// ------------------------------------------------------------------------------------------------------
//// ------------------------------------------------------------------------------------------------------
//
//
//
//template <Semiregular T, BinaryOperation Op, std::size_t Size = 64>
//// 	requires Op is associative
//requires(Domain<Op> == T) //TODO...
//struct counter_machine {
//    // counter_machine(Op const& op, T const& e)
//    counter_machine(Op op, T const& e)
//    : op{ op }, e{ e }, l{ f }
//    {}
//    
//    counter_machine(counter_machine const&) = delete;
//    counter_machine& operator=(counter_machine const&) = delete;
//    
//    //Move Constructor and Move Assignment Operator are deleted too
//    //See http://stackoverflow.com/questions/37092864/should-i-delete-the-move-constructor-and-the-move-assignment-of-a-smart-pointer/38820178#38820178
//    // and http://talesofcpp.fusionfenix.com/post-24/episode-eleven-to-kill-a-move-constructor
//    
//    
//    void add_to(T x, T* to) {
//        // precondition: TODO
//        x = add_to_counter(to, l, op, x, e);
//        if (x != e) {
//            *l = x;
//            ++l;
//        }
//    }
//    
//    // void add(T const& x) {
//    void add(T x) {
//        // precondition: must not be called more than 2^Size - 1 times
//        x = add_to_counter(f, l, op, x, e);
//        if (x != e) {
//            *l = x;
//            ++l;
//        }
//    }
//    
//    const Op op;
//    const T e;
//    T f[Size];
//    T* l;
//};
//
//// template <Semiregular T, BinaryOperation Op>
//// // 	requires Op is associative
//// 	requires(Domain<Op> == T) //TODO...
//// counter_machine<T, Op> make_counter_machine(Op op, T const& e) {
//// 	return counter_machine<T, Op>{op, e};
//// };
//
//
//
//template <Semiregular T, BinaryOperation Op, BinaryOperation OpNoCheck, std::size_t Size = 64>
////  requires Op is associative //TODO
////  requires OpNoCheck is associative //TODO
//requires(Domain<Op> == T && Domain<Op> == Domain<OpNoCheck>) //TODO...
//struct counter_machine_check {
//    counter_machine_check(Op op, OpNoCheck op_nocheck, T const& e)
//    : op(op), op_nocheck(op_nocheck), e(e), l(f)
//    {}
//    
//    counter_machine_check(counter_machine_check const&) = delete;
//    counter_machine_check& operator=(counter_machine_check const&) = delete;
//    
//    //Move Constructor and Move Assignment Operator are deleted too
//    //See http://stackoverflow.com/questions/37092864/should-i-delete-the-move-constructor-and-the-move-assignment-of-a-smart-pointer/38820178#38820178
//    // and http://talesofcpp.fusionfenix.com/post-24/episode-eleven-to-kill-a-move-constructor
//    
//    void add(T x) {
//        // precondition: must not be called more than 2^Size - 1 times
//        x = add_to_counter(f, l, op, x, e);
//        if (x != e) {
//            *l = x;
//            ++l;
//        }
//    }
//    
//    void add_to(T x, T* to) {
//        // precondition: TODO
//        x = add_to_counter(to, l, op_nocheck, x, e);
//        if (x != e) {
//            *l = x;
//            ++l;
//        }
//    }
//    
//    const Op op;
//    const OpNoCheck op_nocheck;
//    const T e;
//    T f[Size];
//    T* l;
//};
//
//
//// ------------------------------------------------------------------------------------------------------
//// ------------------------------------------------------------------------------------------------------
//
//
//
//
//// -----------------------------------------------------------------
//
//#include "sha256.h"
//#include <array>
//#include <string>
//#include <vector>
//#include <iostream>
//
//using namespace std;
//
//template <size_t Size>
//using byte_array = std::array<uint8_t, Size>;
//
//constexpr size_t hash_size = 32;
//constexpr size_t hash_double_size = hash_size * 2;
//
//using hash_t = byte_array<hash_size>;
//using hash_double_t = byte_array<hash_double_size>;
//
////constexpr hash_t null_hash{ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
//
////typedef std::vector<hash_t> hash_list;
//
//
//template <Container C>
//inline
//hash_t sha256(C const& c) {
//    hash_t hash;
//    SHA256_(reinterpret_cast<uint8_t const*>(c.data()), c.size(), hash.data());
//    return hash;
//}
//
////TODO: it is not working
//// template <typename T, std::size_t N>
//// inline
//// hash_t sha256(T const (&x)[N]) {
////     hash_t hash;
////     SHA256_(reinterpret_cast<uint8_t const*>(x), N, hash.data());
////     return hash;
//// }
//
//template <Container C>
//inline
//hash_t sha256_double(C const& c) {
//    return sha256(sha256(c));
//}
//
//template <Container C>
//inline
//hash_t sha256_double_optimized(C const& c) {
//    hash_t hash;
//    SHA256OptDouble_(reinterpret_cast<uint8_t const*>(c.data()), c.size(), hash.data());
//    return hash;
//}
//
//template <Container C1, Container C2>
//inline
//hash_t sha256_double_dual_buffer(C1 const& a, C2 const& b) {
//	//precondition: a.size() == 32 && b.size() == 32
//
//	hash_t hash;
//	SHA256OptDoubleDualBuffer_(
//		reinterpret_cast<uint8_t const*>(a.data()),
//		reinterpret_cast<uint8_t const*>(b.data()), 
//		hash.data());
//	return hash;
//}
//
//template <Container C1, Container C2>
//inline
//hash_double_t concat_containers(C1 const& a, C2 const& b) {
//	hash_double_t res;
//
//	std::memcpy(res.data(), a.data(), a.size() * sizeof(hash_t::value_type));
//	std::memcpy(res.data() + a.size(), b.data(), b.size() * sizeof(hash_t::value_type));
//
//	return res;
//}
//
//template <Container C1, Container C2>
//inline
//hash_t sha256_double_concat(C1 const& a, C2 const& b) {
//	return sha256_double(concat_containers(a, b));
//}
//
//
//
//template <typename HashType>
//void print_hash(HashType const& hash) {
//    for (auto x : hash) {
//        printf("%02x", x);
//    }
//}
//
//// -----------------------------------------------------------------
//
//int main() {
//    
////    auto data256 = "123" + string(253, 'a');
////    string data512(512, 'b');
////    
////    auto a = sha256(data256);
////    auto b = sha256(data512);
//
//	// ---------------------------------------------------------------
//
//    //string x = "a";
//    //auto a = sha256_double(x);
//    //auto b = sha256_double_optimized(x);
//
//
//	//print_hash(a);
//	//cout << endl;
//	//print_hash(b);
//	//cout << endl;
//	// ---------------------------------------------------------------
//
//	string a(32, 'a');
//	string b(32, 'b');
//
//	auto x = sha256_double_dual_buffer(a, b);
//	auto y = sha256_double_concat(a, b);
//	
//
//	print_hash(x);
//	cout << endl;
//
//	print_hash(y);
//	cout << endl;
//
//
//	// ---------------------------------------------------------------
//
//	//string a(64, 'a');
//
//	//auto x = sha256(a);
//
//	//print_hash(x);
//	//cout << endl;
//
//
//    
//    return 0;
//}
