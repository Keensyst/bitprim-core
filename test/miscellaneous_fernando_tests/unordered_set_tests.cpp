// clang++ -O3 -std=c++1z -DMAP -I/Users/fernando/Libs/boost_1_61_0 -I/Users/fernando/Libs/sparsehash-c11-master -L/Users/fernando/Libs/boost_1_61_0/stage/lib -lboost_filesystem -lboost_system unordered_set_tests.cpp
// g++ -O3 -std=c++14 -I/opt/libs/boost_1_61_0 -L/opt/libs/boost_1_61_0/stage/lib -lboost_filesystem -lboost_system -pthread unordered_set_tests.cpp
// g++ -O3 -std=c++11 unordered_set_tests.cpp

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

#ifdef MAP
#include <unordered_map>
#include <sparsehash/sparse_hash_map>
#include <sparsehash/dense_hash_map>
#else
#include <unordered_set>
#include <sparsehash/sparse_hash_set>
#include <sparsehash/dense_hash_set>
#endif

#include <vector>
#include <tuple>

#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/filesystem.hpp>
#include <boost/functional/hash.hpp>

#ifdef MAP
#include <boost/unordered_map.hpp>
#else
#include <boost/unordered_set.hpp>
#endif



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

template <typename F>
inline
void measure_and_print(string const& str, F f, size_t reps) {
	auto start = chrono::high_resolution_clock::now();

	auto res = f(); 

	for (int i = 0; i < reps - 1; ++i)	{
		res = f(); 
		// f();
	}

	auto end = chrono::high_resolution_clock::now();
	auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
 
	cout << str << " value: " << res << " - Measured time: " << dur.count() << " ns - " << (dur.count() / 1000000000.0) << " secs" << endl;
}

template <typename F>
inline
auto execute_measure_and_print(string const& str, F f) {
	auto start = chrono::high_resolution_clock::now();
	auto res = f(); 
	auto end = chrono::high_resolution_clock::now();
	auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
	cout << str << " - Measured time: " << dur.count() << " ns - " << (dur.count() / 1000000000.0) << " secs" << endl;
	return res;
}


// ------------------------------------------------


using hash_t = array<uint8_t, 32>;
// using script_t = array<uint8_t, 512>;
using script_t = array<uint8_t, 256>;

struct data {
	hash_t hash;
	uint32_t index;
	// script_t script;
};

bool operator==(::data const& a, ::data const& b) {
    return a.hash == b.hash && a.index == b.index;
}

bool operator!=(::data const& a, ::data const& b) {
    return !(a == b);
}


namespace std {
template <>
struct hash<::data> {
    size_t operator()(::data const& x) const {
        size_t seed = 0;
        boost::hash_combine(seed, x.hash);
        boost::hash_combine(seed, x.index);
        return seed;
    }
};
} // namespace std

namespace boost {
template <>
struct hash<::data> {
    size_t operator()(::data const& x) const {
        size_t seed = 0;
        boost::hash_combine(seed, x.hash);
        boost::hash_combine(seed, x.index);
        return seed;
    }
};
} // namespace boost


using vec_val_t = pair<size_t, ::data>;
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

#ifdef MAP
	// unordered_map<::data, script_t> res(max);
	// google::sparse_hash_map<::data, script_t> res(max);
	google::dense_hash_map<::data, script_t> res(max);
	::data null_data = {0};
	  // 0, 0, 0, 0, 0, 0, 0, 0,
			// 			0, 0, 0, 0, 0, 0, 0, 0,
			// 			0, 0, 0, 0, 0, 0, 0, 0,
			// 			0, 0, 0, 0, 0, 0, 0, 0,
			// 			0, 0, 0, 0};
	res.set_empty_key(null_data);
#else
	// unordered_set<::data> res(max);
	google::sparse_hash_set<::data> res(max);
#endif


	random_int_generator<> gen;

#ifdef MAP
		auto script = random_script_t_256(gen);
#endif

	for (size_t i = 0; i < max; ++i) {
		::data x {random_hash_t(gen), gen()};

#ifdef MAP
		// res.insert(make_pair(x, script_t{}));
		// res.insert(make_pair(x, random_script_t_256(gen)));
		res.insert(make_pair(x, script));
#else
		res.insert(x);
#endif

		if (i % 100000 == 0) {
			cout << "generated " << i << " elements...\n";
		}

	}
	return res;
}

#ifndef MAP
auto make_data_disk(size_t max, boost::interprocess::managed_mapped_file& mfile) {
	

#ifdef MAP
	using cont_type = boost::unordered_map<::data, script_t, hasher, key_equal, allocator_type>;
#else
	using value_type = ::data;
	using hasher = boost::hash<::data>;
	using key_equal = std::equal_to<::data>;
	using allocator_type = boost::interprocess::allocator<value_type, boost::interprocess::managed_mapped_file::segment_manager>;
	using cont_type = boost::unordered_set<::data, hasher, key_equal, allocator_type>;
#endif


	// constexpr uint64_t initial_file_size = 5ull * 1024 * 1024 * 1024; //5 GiB

	// if (boost::filesystem::exists("fer.data")) {
	// 	boost::filesystem::remove("fer.data");
	// }

	// // std::unique_ptr<boost::interprocess::managed_mapped_file> mfile_ptr_;
	// // mfile_ptr_.reset(new boost::interprocess::managed_mapped_file(boost::interprocess::open_or_create, filename_.c_str(), initial_file_size));
	// // file_size_ = boost::filesystem::file_size(filename_);
	// // cont_ = mfile_ptr_->construct<cont_type>(mapname_.c_str())(max, hasher(), key_equal(), mfile_ptr_->get_allocator<value_type>());
	// boost::interprocess::managed_mapped_file mfile(boost::interprocess::open_or_create, "fer.data", initial_file_size);
	
	auto res = mfile.construct<cont_type>("fer_data")(max, hasher(), key_equal(), mfile.get_allocator<value_type>());

	cout << "file created\n";


	random_int_generator<> gen;

	for (size_t i = 0; i < max; ++i) {
		::data x {random_hash_t(gen), gen()};

#ifdef MAP
		res->insert(make_pair(x, script_t{}));
#else
		res->insert(x);
#endif


		if (i % 100000 == 0) {
			cout << "generated " << i << " elements...\n";
		}
	}


	return *res;
	// return std::move(*res);
}
#endif



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

#ifdef MAP
		auto& x = f->first;
#else
		auto& x = *f;
#endif

		queries.emplace_back(std::hash<::data>()(x), x);
		prev = index;
	}

	return queries;
}


template <AssocContainer C>
auto bench1(C const& data, vec_t const& queries) {
	size_t found = 0;
	for (auto&& query : queries) {
		if (data.count(query.second) > 0) 
			++found;
	}
	return found;
}

template <AssocContainer C>
auto bench2(C const& data, vec_t const& queries) {
	size_t found = 0;
	for (auto&& query : queries) {
		if (data.find(query.second) != data.end()) 
			++found;
	}
	return found;
}

template <AssocContainer C>
auto bench3(C const& data, vec_t& queries) {
	size_t found = 0;

	std::sort(begin(queries), end(queries), [](vec_val_t const& a, vec_val_t const& b) {
		return get<0>(a) < get<0>(b);
	});

	for (auto&& query : queries) {
		if (data.find(query.second) != data.end()) 
			++found;
	}
	return found;
}


void test() {

	constexpr size_t max = 30'000'000;
	constexpr size_t max_queries = 10'000;
	constexpr size_t mod = max / max_queries;


	// auto data = execute_measure_and_print("make_data_memory", make_data_memory);
	auto data = make_data_memory(max);

	// // constexpr uint64_t initial_file_size = 5ull * 1024 * 1024 * 1024; //5 GiB
	// constexpr uint64_t initial_file_size = 15ull * 1024 * 1024 * 1024; //5 GiB
	// if (boost::filesystem::exists("fer.data")) {
	// 	boost::filesystem::remove("fer.data");
	// }

	// boost::interprocess::managed_mapped_file mfile(boost::interprocess::open_or_create, "fer.data", initial_file_size);
	// auto data = make_data_disk(max, mfile);


	cout << "beginning benchmarks...\n";

	auto queries = make_queries<max, max_queries>(data);
	measure_and_print("bench1", [&data, &queries](){
		return bench1(data, queries);	
	}, 1);

	queries = make_queries<max, max_queries>(data);
	measure_and_print("bench2", [&data, &queries](){
		return bench2(data, queries);	
	}, 1);


	queries = make_queries<max, max_queries>(data);
	measure_and_print("bench1", [&data, &queries](){
		return bench1(data, queries);	
	}, 1);

	queries = make_queries<max, max_queries>(data);
	measure_and_print("bench3", [&data, &queries](){
		return bench3(data, queries);	
	}, 1);
}

int main(int /*argc*/, char const* /*argv*/ []) {
	test();
	return 0;
}
