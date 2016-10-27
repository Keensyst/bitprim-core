// clang++ -O3 -std=c++1z -DMAP -I/Users/fernando/Libs/boost_1_61_0 -I/Users/fernando/Libs/sparsehash-c11-master -L/Users/fernando/Libs/boost_1_61_0/stage/lib -lboost_filesystem -lboost_system tx_cache_tests.cpp
// g++ -O3 -std=c++14 -I/opt/libs/boost_1_61_0 -L/opt/libs/boost_1_61_0/stage/lib -lboost_filesystem -lboost_system -pthread tx_cache_tests.cpp
// g++ -O3 -std=c++14 -I/opt/libs/boost_1_61_0 tx_cache_tests.cpp
// g++ -O3 -std=c++11 tx_cache_tests.cpp


#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

#include <unordered_map>
// #include <sparsehash/sparse_hash_map>
// #include <sparsehash/dense_hash_map>

#include <vector>
#include <tuple>

// #include <boost/interprocess/managed_mapped_file.hpp>
// #include <boost/interprocess/allocators/allocator.hpp>
// #include <boost/filesystem.hpp>
#include <boost/functional/hash.hpp>




using namespace std;

#define Integer typename
#define RandomEngine typename
#define Sequence typename
#define Container typename
#define AssocContainer typename

template <Integer I = unsigned int, I From = 0, I To = std::numeric_limits<I>::max()>
struct random_int_generator {
	using dis_t = uniform_int_distribution<I>;
	static constexpr I from = From;
	static constexpr I to = To;

	random_int_generator()
		: mt {rd()}
		, dis {from, to}  // closed range [1, 1000]		
	{}

	random_int_generator(random_int_generator const&) = default;

	auto operator()() {
		return dis(mt);
	}

	random_device rd;
	mt19937 mt;
	dis_t dis;
}; // Models: RandomIntGenerator

template <typename F1, typename F2>
inline
void measure_and_print(string const& str, F1 f, F2 gen, size_t reps) {

	auto data = gen();

	auto start = chrono::high_resolution_clock::now();
	auto res = f(data); 
	auto end = chrono::high_resolution_clock::now();
	auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);

	for (int i = 0; i < reps - 1; ++i)	{
		auto data = gen();
		start = chrono::high_resolution_clock::now();
		res = f(data); 
		end = chrono::high_resolution_clock::now();
		dur += chrono::duration_cast<chrono::nanoseconds>(end - start);
	}
 
	cout << str << " value: " << res << " - Measured time: " << dur.count() << " ns - " << (dur.count() / 1000000000.0) << " secs" << endl;
}

// ------------------------------------------------


using hash_t = array<uint8_t, 32>;
using script_t = array<uint8_t, 256>;
// using script_t = array<uint8_t, 512>;


namespace std {
template <>
struct hash<hash_t> {
    size_t operator()(hash_t const& x) const {
        size_t seed = 0;
        boost::hash_combine(seed, x);
        return seed;
    }
};
} // namespace std

// struct data {
// 	hash_t hash;
// 	uint32_t index;
// 	// script_t script;
// };

// bool operator==(::data const& a, ::data const& b) {
//     return a.hash == b.hash && a.index == b.index;
// }

// bool operator!=(::data const& a, ::data const& b) {
//     return !(a == b);
// }


// namespace std {
// template <>
// struct hash<::data> {
//     size_t operator()(::data const& x) const {
//         size_t seed = 0;
//         boost::hash_combine(seed, x.hash);
//         boost::hash_combine(seed, x.index);
//         return seed;
//     }
// };
// } // namespace std

// namespace boost {
// template <>
// struct hash<::data> {
//     size_t operator()(::data const& x) const {
//         size_t seed = 0;
//         boost::hash_combine(seed, x.hash);
//         boost::hash_combine(seed, x.index);
//         return seed;
//     }
// };
// } // namespace boost


// using vec_val_t = pair<size_t, ::data>;
using vec_val_t = hash_t;
using vec_t = vector<vec_val_t>;


inline
uint8_t const* as_bytes(uint32_t const& x) {
	return static_cast<uint8_t const*>(static_cast<void const*>(&x));
}

inline
uint8_t const* as_bytes(uint32_t const* x) {
	return static_cast<uint8_t const*>(static_cast<void const*>(x));
}


hash_t random_hash_t(random_int_generator<>& gen) {
	uint32_t temp[] = {gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen()};
	hash_t res;
	std::copy(as_bytes(temp), as_bytes(temp) + sizeof(temp), res.data());
	return res;
}

script_t random_script_t_256(random_int_generator<>& gen) {
	uint32_t temp[] = {gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(),
						gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(),
						gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(),
						gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen()};

	// uint32_t temp[] = {gen()};

	script_t res;
	std::copy(as_bytes(temp), as_bytes(temp) + sizeof(temp), res.data());
	return res;
}


auto make_data_memory(size_t max) {

	unordered_map<hash_t, script_t> res(max);
	// google::sparse_hash_map<hash_t, script_t> res(max);
	// google::dense_hash_map<hash_t, script_t> res(max);
	// hash_t null_data = {0};
	//   // 0, 0, 0, 0, 0, 0, 0, 0,
	// 		// 			0, 0, 0, 0, 0, 0, 0, 0,
	// 		// 			0, 0, 0, 0, 0, 0, 0, 0,
	// 		// 			0, 0, 0, 0, 0, 0, 0, 0,
	// 		// 			0, 0, 0, 0};
	// res.set_empty_key(null_data);


	random_int_generator<> gen;

	auto script = random_script_t_256(gen);

	for (size_t i = 0; i < max; ++i) {
		hash_t x = random_hash_t(gen);

		// res.insert(make_pair(x, script_t{}));
		// res.insert(make_pair(x, random_script_t_256(gen)));
		res.insert(make_pair(x, script));

		// if (i % 100000 == 0) {
		// 	cout << "generated " << i << " elements...\n";
		// }

	}
	return res;
}


template <size_t max, size_t max_queries, AssocContainer C>
auto make_queries(C const& data) {

	random_int_generator<size_t, 0, max - 1> gen;


	vector<size_t> indexes;
	indexes.reserve(max_queries);
	
	for (size_t i = 0; i < max_queries; ++i) {
		indexes.push_back(gen());
	}
	sort(begin(indexes), end(indexes));

	// for (auto index : indexes) {
	// 	cout << index << endl;
	// }

	vec_t queries;
	queries.reserve(max_queries);

	auto f = begin(data);
	size_t prev = 0;
	for (auto index : indexes) {
		std::advance(f, index - prev);

		auto& x = f->first;
		// queries.emplace_back(std::hash<::data>()(x), x);
		queries.push_back(x);
		prev = index;
	}

	return queries;
}


template <AssocContainer C>
auto bench1(C const& data, vec_t const& queries) {
	size_t found = 0;
	for (auto&& query : queries) {
		if (data.count(query) > 0) 
			++found;
	}
	return found;
}

template <AssocContainer C>
auto bench2(C const& data, vec_t const& queries) {
	size_t found = 0;
	for (auto&& query : queries) {
		if (data.find(query) != data.end()) 
			++found;
	}
	return found;
}

template <AssocContainer C>
auto bench3(C& data, vec_t const& queries) {

	// cout << "queries.size():   " << queries.size() << endl;

	size_t erased = 0;
	for (auto&& query : queries) {
		// auto found = data.count(query);
		// auto local_erased = data.erase(query);

		// cout << "found:        " << found << endl;
		// cout << "local_erased: " << local_erased << endl;

		erased += data.erase(query);
		
	}
	return erased;
}

inline
void measure_and_print_bench3(size_t reps) {
	constexpr size_t max = 10'000;
	constexpr size_t max_queries = 3'000;

	auto data = make_data_memory(max);
	auto queries = make_queries<max, max_queries>(data);

	auto start = chrono::high_resolution_clock::now();
	auto res = bench3(data, queries); 
	auto end = chrono::high_resolution_clock::now();
	auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);

	for (int i = 0; i < reps - 1; ++i)	{
		data = make_data_memory(max);
		queries = make_queries<max, max_queries>(data);
		start = chrono::high_resolution_clock::now();
		res = bench3(data, queries); 
		end = chrono::high_resolution_clock::now();
		dur += chrono::duration_cast<chrono::nanoseconds>(end - start);
	}
 
	cout << "bench3" << " value: " << res << " - Measured time: " << dur.count() << " ns - " << (dur.count() / 1000000000.0) << " secs" << endl;
}


void test() {

	constexpr size_t max = 10'000;
	constexpr size_t max_queries = 3'000;
	// constexpr size_t mod = max / max_queries;

	auto data = make_data_memory(max);


	cout << "beginning benchmarks...\n";

	// measure_and_print("bench1", 
	// 	[&data](vec_t const& queries){
	// 		return bench1(data, queries);	
	// 	},
	// 	[&data](){
	// 		return make_queries<max, max_queries>(data);
	// 	}
	// 	, 1000
	// );

	measure_and_print("bench2", 
		[&data](vec_t const& queries){
			return bench2(data, queries);	
		},
		[&data](){
			return make_queries<max, max_queries>(data);
		}
		, 10000
	);

	measure_and_print_bench3(10000);

	// measure_and_print("bench3", 
	// 	[&data](vec_t const& queries){
	// 		return bench3(data, queries);	
	// 	},
	// 	[&data](){
	// 		return make_queries<max, max_queries>(data);
	// 	}
	// 	, 1000
	// );		

	// queries = make_queries<max, max_queries>(data);
	// measure_and_print("bench2", [&data, &queries](){
	// 	return bench2(data, queries);	
	// }, 10000);

	// queries = make_queries<max, max_queries>(data);
	// measure_and_print("bench1", [&data, &queries](){
	// 	return bench1(data, queries);	
	// }, 10000);

	// queries = make_queries<max, max_queries>(data);
	// measure_and_print("bench2", [&data, &queries](){
	// 	return bench2(data, queries);	
	// }, 10000);

	// queries = make_queries<max, max_queries>(data);
	// measure_and_print("bench3", [&data, &queries](){
	// 	return bench3(data, queries);	
	// }, 1);

}

int main(int /*argc*/, char const* /*argv*/ []) {
	test();
	return 0;
}
