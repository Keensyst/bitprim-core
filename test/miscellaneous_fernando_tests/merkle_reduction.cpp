// clang++ -O3 -std=c++14 merkle_reduction.cpp sha256.c zeroize.c
// clang++ -O3 -std=c++14 merkle_reduction.cpp sha256_tests/sha256_tests/sha256.c sha256_tests/sha256_tests/zeroize.c

// g++ -O3 -std=c++14 merkle_reduction.cpp sha256.c zeroize.c
// g++ -O3 -std=c++14 merkle_reduction.cpp hash-library/sha256.cpp
// g++ -O3 -std=c++14 merkle_reduction.cpp sha256_tests/sha256_tests/sha256.c sha256_tests/sha256_tests/zeroize.c


#include <cstring>
#include <algorithm>
#include <chrono>
#include <iterator>
#include <random>

// Concepts
#define Integer typename
#define Container typename
#define Iterator typename
#define ForwardIterator typename
#define BinaryOperation typename
#define Semiregular typename
#define UnaryFunction typename
#define RandomEngine typename
#define Sequence typename
#define AssocContainer typename

#define requires(...) 

//TODO: 
	// Mutable<I>
	// Domain<Op>

// Type Attributes / Type Functions
template <Iterator I>
using ValueType = typename std::iterator_traits<I>::value_type;


// -----------------------------------------------------------------

template<Integer I>
bool even(I const& a) {
    return (a bitand I(1)) == I(0);
}

// -----------------------------------------------------------------

template <BinaryOperation Op>
struct transpose_operation {
	const Op op;
	
	explicit
	transpose_operation(Op op) 
		: op{op}
	{}

	template <typename T>
		requires(T == Domain<Op>)
	T operator()(T const& a, T const& b) const {
		return op(b, a);
	}
};


// -----------------------------------------------------------------
// Algorithms (Generic)
// -----------------------------------------------------------------

template <Iterator I, BinaryOperation Op, UnaryFunction F>
    requires(I == Domain<F> && Codomain<F> == Domain<Op> && ValueType<I> == Domain<Op>)
ValueType<I> reduce_nonempty(I f, I l, Op op, F fun) {
	// precondition: bounded_range(f, l) and f != l and
	//               partially_associative(op) and
	//               ForAll x in [f, l), fun(x) is defined

    ValueType<I> r = fun(f);
    ++f;
    while (f != l) {
        r = op(r, fun(f)); 
        ++f;
    }
    return r;
}

template <Iterator I, BinaryOperation Op>
    requires(ValueType<I> == Domain<Op>)
ValueType<I> reduce_nonempty(I f, I l, Op op) {
	// precondition: bounded_range(f, l) and f != l and
	//               partially_associative(op) and

    ValueType<I> r = *f;
    ++f;
    while (f != l) {
        r = op(r, *f); 
        ++f;
    }
    return r;
}


template <Iterator I, BinaryOperation Op, UnaryFunction F>
	requires(I == Domain<F> && Codomain<F> == Domain<Op> && ValueType<I> == Domain<Op>) //TODO...
ValueType<I> reduce_nonzeroes(I f, I l, Op op, F fun, ValueType<I> const& e) {
	// precondition: bounded_range(f, l) and
	//               partially_associative(op) and
	//               ForAll x in [f, l), fun(x) is defined

	ValueType<I> x;
	do {
		if (f == l) return e;
		x = fun(f);
		++f;
	} while (x == e);

	while (f != l) {
		ValueType<I> y = fun(f);
		if (y != e) x = op(x, y);
		++f;
	}
	return x;
}

template <Iterator I, BinaryOperation Op>
	requires(ValueType<I> == Domain<Op>) //TODO...
ValueType<I> reduce_nonzeroes(I f, I l, Op op, ValueType<I> const& e) {
	// precondition: bounded_range(f, l) and
	//               partially_associative(op) and

	ValueType<I> x;
	do {
		if (f == l) return e;
		x = *f;
		++f;
	} while (x == e);

	while (f != l) {
		// ValueType<I> y = *f;
		// ValueType<I> const& y = *f;
		// if (y != e) x = op(x, y);
		if (*f != e) x = op(x, *f);
		++f;
	}
	return x;
}


// Counter Machine
// template <ForwardIterator I, BinaryOperation Op>
// add_to_counter(I f, I l, Op op, Domain<Op> x, )


template <ForwardIterator I, BinaryOperation Op>
// 	TODO: requires Op is associative
	requires(Mutable<I> && ValueType<I> == Domain<Op>) 
ValueType<I> add_to_counter(I f, I l, Op op, ValueType<I> x, ValueType<I> const& e) {
	// precondition: x != e
	while (f != l) {
		if (*f == e) {
			*f = x;
			return e;
		}
		x = op(*f, x);
		*f = e;
		++f;
	}
	return x;
}

// ------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------



template <Semiregular T, BinaryOperation Op, std::size_t Size = 64>
// 	requires Op is associative
	requires(Domain<Op> == T) //TODO...
struct counter_machine {
	// counter_machine(Op const& op, T const& e)
	counter_machine(Op op, T const& e)
		: op{op}, e{e}, l{f}
	{}

	counter_machine(counter_machine const&) = delete;
	counter_machine& operator=(counter_machine const&) = delete;

    //Move Constructor and Move Assignment Operator are deleted too
    //See http://stackoverflow.com/questions/37092864/should-i-delete-the-move-constructor-and-the-move-assignment-of-a-smart-pointer/38820178#38820178
    // and http://talesofcpp.fusionfenix.com/post-24/episode-eleven-to-kill-a-move-constructor


	void add_to(T x, T* to) {
		// precondition: TODO
		x = add_to_counter(to, l, op, x, e);
		if (x != e) {
			*l = x;
			++l;
		}
	}

	// void add(T const& x) {
	void add(T x) {
		// precondition: must not be called more than 2^Size - 1 times
		x = add_to_counter(f, l, op, x, e);
		if (x != e) {
			*l = x;
			++l;
		}
	}

	const Op op;
	const T e;
	T f[Size];
	T* l;
};

// template <Semiregular T, BinaryOperation Op>
// // 	requires Op is associative
// 	requires(Domain<Op> == T) //TODO...
// counter_machine<T, Op> make_counter_machine(Op op, T const& e) {
// 	return counter_machine<T, Op>{op, e};
// };



template <Semiregular T, BinaryOperation Op, BinaryOperation OpNoCheck, std::size_t Size = 64>
//  requires Op is associative //TODO
//  requires OpNoCheck is associative //TODO
    requires(Domain<Op> == T && Domain<Op> == Domain<OpNoCheck>) //TODO...
struct counter_machine_check {
    counter_machine_check(Op op, OpNoCheck op_nocheck, T const& e)
        : op(op), op_nocheck(op_nocheck), e(e), l(f)
    {}

    counter_machine_check(counter_machine_check const&) = delete;
    counter_machine_check& operator=(counter_machine_check const&) = delete;
    
    //Move Constructor and Move Assignment Operator are deleted too
    //See http://stackoverflow.com/questions/37092864/should-i-delete-the-move-constructor-and-the-move-assignment-of-a-smart-pointer/38820178#38820178
    // and http://talesofcpp.fusionfenix.com/post-24/episode-eleven-to-kill-a-move-constructor

    void add(T x) {
        // precondition: must not be called more than 2^Size - 1 times
        x = add_to_counter(f, l, op, x, e);
        if (x != e) {
            *l = x;
            ++l;
        }        
    }

    void add_to(T x, T* to) {
        // precondition: TODO
        x = add_to_counter(to, l, op_nocheck, x, e);
        if (x != e) {
            *l = x;
            ++l;
        }
    }

    const Op op;
    const OpNoCheck op_nocheck;
    const T e;
    T f[Size];
    T* l;
};


// ------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------



template <Iterator I, BinaryOperation Op, UnaryFunction F>
// requires Op is associative
	requires(Readable<I> && ValueType<I> == Domain<Op>) 
ValueType<I> reduce_balanced(I f, I l, Op op, F fun, ValueType<I> const& e) {
	// precondition: bounded_range(f, l) and l - f < 2^64 and
	//               partially_associative(op) and
	//               ForAll x in [f, l), fun(x) is defined
	
	counter_machine<ValueType<I>, Op> c{op, e};
	while (f != l) {
		c.add(fun(f));
		++f;
	}

	transpose_operation<Op> t_op{op};
	return reduce_nonzeroes(c.f, c.l, t_op, e);
}

template <Iterator I, BinaryOperation Op>
// requires Op is associative
	requires(Readable<I> && ValueType<I> == Domain<Op>) 
ValueType<I> reduce_balanced(I f, I l, Op op, ValueType<I> const& e) {
	// precondition: bounded_range(f, l) and l - f < 2^64 and
	//               partially_associative(op) and
	
	counter_machine<ValueType<I>, Op> c{op, e};
	while (f != l) {
		c.add(*f);
		++f;
	}

	transpose_operation<Op> t_op{op};
	return reduce_nonzeroes(c.f, c.l, t_op, e);
}

// template <Iterator I, BinaryOperation Op, UnaryFunction F>
// // requires Op is associative
// 	requires(Readable<I> && ValueType<I> == Domain<Op>) 
// ValueType<I> reduce_balanced_nonempty(I f, I l, Op op, F fun) {
// 	// precondition: bounded_range(f, l) and f != l and l - f < 2^64 and
// 	//               partially_associative(op) and
// 	//               ForAll x in [f, l), fun(x) is defined
	
// 	counter_machine<ValueType<I>, Op> c{op, e};
// 	while (f != l) {
// 		c.add(fun(f));
// 		++f;
// 	}

// 	transpose_operation<Op> t_op{op};
// 	return reduce_nonempty(c.f, c.l, t_op, e);
// }

// template <Iterator I, BinaryOperation Op>
// // requires Op is associative
// 	requires(Readable<I> && ValueType<I> == Domain<Op>) 
// ValueType<I> reduce_balanced_nonempty(I f, I l, Op op) {
// 	// precondition: bounded_range(f, l) and f != l and l - f < 2^64 and
// 	//               partially_associative(op) and
	
// 	counter_machine<ValueType<I>, Op> c{op, e};
// 	while (f != l) {
// 		c.add(*f);
// 		++f;
// 	}

// 	transpose_operation<Op> t_op{op};
// 	return reduce_nonempty(c.f, c.l, t_op, e);
// }

// template <Semiregular T, BinaryOperation Op>
// // 	requires Op is associative
// 	requires(Domain<Op> == T) 
// class counter_machine {
// public:
// 	counter_machine(Op const& op, T const& e)
// 		: op{op}, e{e}
// 	{}

// 	counter_machine(counter_machine const&) = delete;
// 	counter_machine& operator=(counter_machine const&) = delete;
// 	counter_machine(counter_machine&&) = delete;
// 	counter_machine& operator=(counter_machine&&) = delete;

// 	void add(T x) {
// 		x = add_to_counter(std::begin(counter), std::end(counter), op, x, e);
// 		if (x != e) {
// 			counter.push_back(x);
// 		}
// 	}

// 	T reduce() const {
// 		return reduce_counter(std::begin(counter), std::end(counter), op, e);
// 	}

// private:
// 	std::vector<T> counter;
// 	const Op op;
// 	const T e;
// };


// -----------------------------------------------------------------

// #include "sha256.h"
// #include "hash-library/sha256.h"
#include "sha256_tests/sha256_tests/sha256.h"
#include <array>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

template <size_t Size>
using byte_array = std::array<uint8_t, Size>;

constexpr size_t hash_size = 32;
constexpr size_t hash_double_size = hash_size * 2;

using hash_t = byte_array<hash_size>;
using hash_double_t = byte_array<hash_double_size>;

constexpr hash_t null_hash {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

typedef std::vector<hash_t> hash_list;




template <Container C>
inline
hash_t sha256(C const& c) {
    hash_t hash;
    SHA256_(reinterpret_cast<uint8_t const*>(c.data()), c.size(), hash.data());
    return hash;
}

// template <Container C>
// inline
// hash_t sha256_optimized1(C const& c) {
//     hash_t hash;
//     SHA256Opt1_(reinterpret_cast<uint8_t const*>(c.data()), c.size(), hash.data());
//     return hash;
// }

//TODO: it is not working
// template <typename T, std::size_t N>
// inline
// hash_t sha256(T const (&x)[N]) {
//     hash_t hash;
//     SHA256_(reinterpret_cast<uint8_t const*>(x), N, hash.data());
//     return hash;
// }

// template <Container C>
// inline
// hash_t sha256_double(C const& c) {
//     return sha256(sha256(c));
// }

template <Container C>
	requires(sizeof(ValueType<C>) == 8)
inline
hash_t sha256_double(C const& c) {
    hash_t hash;
    SHA256OptDouble_(reinterpret_cast<uint8_t const*>(c.data()), c.size(), hash.data());
    return hash;
}

inline
hash_double_t concat_hash(hash_t const& a, hash_t const& b) {
	hash_double_t res;

  	std::memcpy(res.data(),            a.data(), a.size() * sizeof(hash_t::value_type));
  	std::memcpy(res.data() + a.size(), b.data(), b.size() * sizeof(hash_t::value_type));

  	return res;
}

template <Container C1, Container C2>
inline
hash_t sha256_double_dual_buffer(C1 const& a, C2 const& b) {
    //precondition: a.size() == 32 && b.size() == 32

    hash_t hash;
    SHA256OptDoubleDualBuffer_(
        reinterpret_cast<uint8_t const*>(a.data()),
        reinterpret_cast<uint8_t const*>(b.data()), 
        hash.data());
    return hash;
}

// inline
// hash_t merkle_op(hash_t const& a, hash_t const& b) {
//   	return sha256_double(concat_hash(a, b));
// }

struct merkle_op {
    hash_t operator()(hash_t const& a, hash_t const& b) const {
        // return sha256_double(concat_hash(a, b));
        return sha256_double_dual_buffer(a, b);
    }
};

struct merkle_op_check {
    bool any_equal = false;
	hash_t operator()(hash_t const& a, hash_t const& b) {
        any_equal |= (a == b);
        return merkle_op()(a, b);
	}
};


// struct merkle_op_optimized {
//     hash_t operator()(hash_t const& a, hash_t const& b) const {
//         return sha256_double_optimized(concat_hash(a, b));
//     }
// };

// struct merkle_op_check_optimized {
//     bool any_equal = false;
// 	hash_t operator()(hash_t const& a, hash_t const& b) {
//         any_equal |= (a == b);
//         return merkle_op_optimized()(a, b);
// 	}
// };

template <typename HashType>
void print_hash(HashType const& hash) {
	for (auto x : hash) {
    	printf("%02x", x);	
	}
}

template <typename HashType>
void print_hash_ln(HashType const& hash) {
    for (auto x : hash) {
        printf("%02x", x);  
    }
    printf("\n");  
}

// template <Container C>
// hash_t generate_merkle_root(C const& txs) {
//     if (txs.empty()) return null_hash;

//     hash_list tx_hashes;
//     tx_hashes.reserve((txs.size() + 1) & ~1); //round up to the next even number
    
//     for (auto const& tx: txs) {
//         tx_hashes.push_back(sha256_double(tx));
//     }

//     if (!even(txs.size())) {
//     	tx_hashes.push_back(tx_hashes.back());
//     }

// 	//return reduce_balanced_nonempty(begin(tx_hashes), end(tx_hashes), merkle_op);
// 	return reduce_balanced(begin(tx_hashes), end(tx_hashes), merkle_op, null_hash); //TODO: null_hash never be used
// }

template <size_t depth = 20, Container C>
pair<hash_t, bool> generate_merkle_root(C const& txs) {
    // precondition: ???

	if (txs.empty()) return{ null_hash, false};

    //counter_machine<hash_t, merkle_op, 20> c(merkle_op(), null_hash);

    merkle_op_check op;
    counter_machine_check<hash_t, merkle_op_check, merkle_op, depth> c(op, merkle_op(), null_hash);

	for (auto&& tx : txs) {
		c.add(sha256_double(tx));
	}

	auto f = c.f;
	while (f != c.l - 1) {
		if (*f != null_hash) {
			c.add_to(*f, f);
		}
		++f;
	}
	return {*(c.l - 1), op.any_equal};
}

// template <size_t depth = 20, Container C>
// pair<hash_t, bool> generate_merkle_root_opt(C const& txs) {
//     // precondition: ???

// 	if (txs.empty()) return{ null_hash, false};

//     merkle_op_check_optimized op;
//     counter_machine_check<hash_t, merkle_op_check_optimized, merkle_op_optimized, depth> c(op, merkle_op_optimized(), null_hash);

// 	for (auto&& tx : txs) {
// 		c.add(sha256_double_optimized(tx));
// 	}

// 	auto f = c.f;
// 	while (f != c.l - 1) {
// 		if (*f != null_hash) {
// 			c.add_to(*f, f);
// 		}
// 		++f;
// 	}
// 	return {*(c.l - 1), op.any_equal};
// }
// ------------------------------------------------------------------
// Original Libbitcoin code
// ------------------------------------------------------------------

hash_t build_merkle_tree(hash_list& merkle) {
    // Stop if hash list is empty.
    if (merkle.empty())
        return null_hash;

    // While there is more than 1 hash in the list, keep looping...
    while (merkle.size() > 1) {
        // If number of hashes is odd, duplicate last hash in the list.
        if (merkle.size() % 2 != 0)
            merkle.push_back(merkle.back());

        // // List size is now even.
        // BITCOIN_ASSERT(merkle.size() % 2 == 0);

        // New hash list.
        hash_list new_merkle;

        // Loop through hashes 2 at a time.
        for (auto it = merkle.begin(); it != merkle.end(); it += 2) {
            // // Join both current hashes together (concatenate).
            // data_chunk concat_data;
            // data_sink concat_stream(concat_data);
            // ostream_writer concat_sink(concat_stream);
            // concat_sink.write_hash(*it);
            // concat_sink.write_hash(*(it + 1));
            // concat_stream.flush();

            // BITCOIN_ASSERT(concat_data.size() == (2 * hash_size));


			const auto concat_data = concat_hash(*it, *(it + 1));

            // Hash both of the hashes.
            const auto new_root = sha256_double(concat_data);

            // Add this to the new list.
            new_merkle.push_back(new_root);
        }

        // This is the new list.
        merkle = new_merkle;
    }

    // Finally we end up with a single item.
    return merkle[0];
}

template <Container C>
hash_t generate_merkle_root_old(C const& txs) {
    hash_list tx_hashes;
    for (const auto& tx: txs)
        tx_hashes.push_back(sha256_double(tx));
    	// tx_hashes.push_back(tx.hash());
    

    // Build merkle tree.
    return build_merkle_tree(tx_hashes);
}


// -----------------------------------------------------------------

template <Integer I = unsigned int, I From = 0, I To = std::numeric_limits<I>::max()>
struct random_int_generator {
    using dis_t = uniform_int_distribution<I>;
    static constexpr I from = From;
    static constexpr I to = To;

    random_int_generator()
        // : mt {rd()}
        : dis {from, to}  // closed range [1, 1000]     
    {}

    random_int_generator(random_int_generator const&) = default;

    auto operator()() {
        return dis(eng);
    }

    // random_device rd;
    // mt19937 eng;
    std::mt19937 eng{std::random_device{}()};
    // std::mt19937 eng{std::chrono::system_clock::now().time_since_epoch().count()};
    // std::mt19937 eng(std::chrono::system_clock::now().time_since_epoch().count());
    
    dis_t dis;
}; // Models: RandomIntGenerator

inline
uint8_t const* as_bytes(uint32_t const& x) {
    return static_cast<uint8_t const*>(static_cast<void const*>(&x));
}

inline
uint8_t const* as_bytes(uint32_t const* x) {
    return static_cast<uint8_t const*>(static_cast<void const*>(x));
}

using transaction_t = array<uint8_t, 256>;

transaction_t random_transaction_t(random_int_generator<>& gen) {
    uint32_t temp[] = { gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(),
                        gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(),
                        gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(),
                        gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen()};

    transaction_t res;
    std::copy(as_bytes(temp), as_bytes(temp) + sizeof(temp), res.data());
    return res;
}

std::vector<transaction_t> make_transactions(random_int_generator<>& gen, size_t n) {
    // random_int_generator<> gen;
    std::vector<transaction_t> res(n);
    // res.reserve(n);
    std::generate(std::begin(res), std::end(res), [&gen]{ return random_transaction_t(gen); });
    return res;
}

// -----------------------------------------------------------------

template <size_t depth = 20>
void bench_merkle(random_int_generator<>& gen, size_t n, size_t reps = 1) {
    auto data = make_transactions(gen, n);

    // cout << data.size() << endl;
    // print_hash_ln(data[0]);

    auto start = std::chrono::high_resolution_clock::now();

    auto res = generate_merkle_root<depth>(data); 
    for (int i = 0; i < reps - 1; ++i)  {
        res = generate_merkle_root<depth>(data);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
 
    cout << "bench_merkle - "
         << " n: " << n 
         << " - res.first:  ";
    
    print_hash(res.first);

    cout << " - res.second: " << res.second 
         << " - Measured time: " << dur.count() << " ns - " 
         << (dur.count() / 1000000000.0) << " secs" << endl;
}

// template <size_t depth = 20>
// void bench_merkle_optimized(random_int_generator<>& gen, size_t n, size_t reps = 1) {
//     auto data = make_transactions(gen, n);

//     // cout << data.size() << endl;
//     // print_hash_ln(data[0]);

//     auto start = std::chrono::high_resolution_clock::now();

//     auto res = generate_merkle_root_opt<depth>(data); 
//     for (int i = 0; i < reps - 1; ++i)  {
//         res = generate_merkle_root_opt<depth>(data);
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
 
//     cout << "bench_merkle_optimized - "
//          << " n: " << n 
//          << " - res.first:  ";
    
//     print_hash(res.first);

//     cout << " - res.second: " << res.second 
//          << " - Measured time: " << dur.count() << " ns - " 
//          << (dur.count() / 1000000000.0) << " secs" << endl;
// }

void bench_merkle_old(random_int_generator<>& gen, size_t n, size_t reps = 1) {
    auto data = make_transactions(gen, n);

    auto start = std::chrono::high_resolution_clock::now();
    auto res = generate_merkle_root_old(data); 

    // for (int i = 0; i < reps - 1; ++i)  {
    //     res = generate_merkle_root_old(data); 
    // }

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
 
    cout << "bench_merkle_old - "
         << " n: " << n 
         << " - res:  ";
    
    print_hash(res);

    cout << " - Measured time: " << dur.count() << " ns - " 
         << (dur.count() / 1000000000.0) << " secs" << endl;
}

// -----------------------------------------------------------------


int main(int /*argc*/, char const* /*argv*/ []) {
    
	// string luca = "luca";
	// print_hash(sha256(luca));
	// cout << endl;
	// print_hash(sha256_optimized1(luca));
	// cout << endl;


    random_int_generator<> gen;
    const size_t reps = 10;

    bench_merkle(gen, 4096, reps);
    bench_merkle(gen, 8192, reps);
    bench_merkle(gen, 16384, reps);
    bench_merkle(gen, 32768, reps);
    bench_merkle(gen, 65536, reps);
    bench_merkle(gen, 131072, reps);
    bench_merkle(gen, 262144, reps);
    bench_merkle(gen, 524288, reps);
    bench_merkle<21>(gen, 1048576, reps);

    // bench_merkle_optimized(gen, 4096, reps);
    // bench_merkle_optimized(gen, 8192, reps);
    // bench_merkle_optimized(gen, 16384, reps);
    // bench_merkle_optimized(gen, 32768, reps);
    // bench_merkle_optimized(gen, 65536, reps);
    // bench_merkle_optimized(gen, 131072, reps);
    // bench_merkle_optimized(gen, 262144, reps);
    // bench_merkle_optimized(gen, 524288, reps);
    // bench_merkle_optimized<21>(gen, 1048576, reps);


    // bench_merkle_old(gen, 4096);
    // bench_merkle_old(gen, 8192);
    // bench_merkle_old(gen, 16384);
    // bench_merkle_old(gen, 32768);
    // bench_merkle_old(gen, 65536);
    // bench_merkle_old(gen, 131072);
    // bench_merkle_old(gen, 262144);
    // bench_merkle_old(gen, 524288);
    // bench_merkle_old(gen, 1048576);


    return 0;
}


// -----------------------------------------------------------------


// int main(int /*argc*/, char const* /*argv*/ []) {
	
// 	//even
// 	// vector<string> data0 {"hola", "como", "estas", "sabia"};
// 	//odd
// 	// vector<string> data0 {"hola", "como", "estas"};
// 	// vector<string> data0 {"hola", "como", "estas", "sabia", "que", "vendrias", "por", "aca", "yo"};
// 	// vector<string> data0 {"hola", "como", "estas", "sabia", "que", "vendrias", "por"};
// 	// vector<string> data0 {"hola", "como", "estas", "sabia", "que", "vendrias", "por", "aca", "yo", "estoy", "bien"};
// 	vector<string> data0 {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "l", "l", "l", "l", "l"};


// // 4096
// // 8192
// // 16384
// // 32768
// // 65536
// // 131072
// // 262144
// // 524288
// // 1048576


// 	auto a = generate_merkle_root_old(data0);
// 	auto b = generate_merkle_root(data0);

// 	print_hash(a);
// 	cout << endl;
// 	print_hash(b);
// 	cout << endl;


// 	return 0;
// }



	// auto h0 = sha256_double(data0[0]);
	// auto h1 = sha256_double(data0[1]);
	// auto h2 = sha256_double(data0[2]);
	// auto h3 = sha256_double(data0[3]);
	// // auto h3 = sha256_double(data0[2]);

	// auto _01 = concat_hash(h0, h1);
	// auto h01 = sha256_double(_01);

	// auto _23 = concat_hash(h2, h3);
	// auto h23 = sha256_double(_23);

	// auto _0123 = concat_hash(h01, h23);
	// auto h0123 = sha256_double(_0123);

	// // print_hash(h0);
	// // cout << endl;

	// print_hash(h0123);
	// cout << endl;

