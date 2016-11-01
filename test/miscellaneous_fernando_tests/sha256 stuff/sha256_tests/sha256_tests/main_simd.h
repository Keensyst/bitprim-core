#pragma once

//template <size_t Bits>
//struct simd_type_impl;
//
//template <>
//struct simd_type_impl<128> {
//	using type = __m128i;
//};
//
//template <>
//struct simd_type_impl<256> {
//	using type = __m256i;
//};
//
//template <size_t Bits>
//using simd_type = typename simd_type_impl<Bits>::type;


template <size_t Bits>
struct simd_type;

template <>
struct simd_type<128> {
    using type = __m128i;
};

template <>
struct simd_type<256> {
    using type = __m256i;
};

//template <size_t Bits>
//using simd_type = typename simd_type_impl<Bits>::type;


//using simd256 = simd_type<256>;
//using simd128 = simd_type<128>;


// simd_set_epi32 ************************************************************
template <size_t Bits>
auto simd_set_epi32(int x);

template <>
inline /*constexpr*/
auto simd_set_epi32<128>(int x) {
	return _mm_set_epi32(x, x, x, x);
}

template <>
inline /*constexpr*/
auto simd_set_epi32<256>(int x) {
	return _mm256_set_epi32(x, x, x, x, x, x, x, x);
}



// simd_and ******************************************************************
auto simd_and(simd_type<256> a, simd_type<256> b) {
	return _mm256_and_si256(a, b);
}

auto simd_and(simd_type<128> a, simd_type<128> b) {
	return _mm_and_si128(a, b);
}

// simd_or ******************************************************************
auto simd_or(simd_type<256> a, simd_type<256> b) {
	return _mm256_or_si256(a, b);
}

auto simd_or(simd_type<128> a, simd_type<128> b) {
	return _mm_or_si128(a, b);
}




// simd_xor ******************************************************************
auto simd_xor(simd_type<256> a, simd_type<256> b) {
	return _mm256_xor_si256(a, b);
}

auto simd_xor(simd_type<128> a, simd_type<128> b) {
	return _mm_xor_si128(a, b);
}

// simd_rshift ***************************************************************
auto simd_rshift(simd_type<256> a, int x) {
	return _mm256_srli_epi32(a, x);
}

auto simd_rshift(simd_type<128> a, int x) {
	return _mm_srli_epi32(a, x);
}


// simd_lshift ***************************************************************
auto simd_lshift(simd_type<256> a, int x) {
	return _mm256_slli_epi32(a, x);
}

auto simd_lshift(simd_type<128> a, int x) {
	return _mm_slli_epi32(a, x);
}

// simd_add ***************************************************************
auto simd_add(simd_type<256> a, simd_type<256> b) {
	return _mm256_add_epi32(a, b);
}

auto simd_add(simd_type<128> a, simd_type<128> b) {
	return _mm_add_epi32(a, b);
}




// simd_sigma_0 **************************************************************
template <size_t Bits>
inline
simd_type<Bits> simd_sigma_0(simd_type<Bits> W) {
	return
		simd_xor(
			simd_xor(
				simd_xor(
					simd_rshift(W, 7),
					simd_rshift(W, 18)
					),
				simd_xor(
					simd_rshift(W, 3),
					simd_lshift(W, 25)
					)
				),
			simd_lshift(W, 14)
			);
}

// simd_sigma_1 **************************************************************
template <size_t Bits>
inline
simd_type<Bits> simd_sigma_1(simd_type<Bits> W) {
	return
		simd_xor(
			simd_xor(
				simd_xor(
					simd_rshift(W, 17),
					simd_rshift(W, 10)
				),
				simd_xor(
					simd_rshift(W, 19),
					simd_lshift(W, 15)
				)
			),
			simd_lshift(W, 13)
		);
}




// simd_gather_be ************************************************************
template <size_t Bits>
auto simd_gather_be(uint32_t const* address);

template <>
auto simd_gather_be<256>(uint32_t const* address) {
	static const __m256i bswap_mask = _mm256_set_epi8(28, 29, 30, 31, 24, 25, 26, 27, 20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
	auto tmp = _mm256_set_epi32(address[112], address[96], address[80], address[64], address[48], address[32], address[16], address[0]);
	return _mm256_shuffle_epi8(tmp, bswap_mask);
}

template <>
auto simd_gather_be<128>(uint32_t const* address) {
	static const __m128i bswap_mask = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
	auto tmp = _mm_set_epi32(address[48], address[32], address[16], address[0]);
	return _mm_shuffle_epi8(tmp, bswap_mask);
}



// sha256_init_state *********************************************************
template <size_t Bits>
/*constexpr*/
auto sha256_init_state() {
	/*constexpr*/
	//static simd_type<Bits> state[SHA256_STATE_LENGTH] = {
	//	simd_set_epi32<Bits>(0x6A09E667),
	//	simd_set_epi32<Bits>(0xBB67AE85),
	//	simd_set_epi32<Bits>(0x3C6EF372),
	//	simd_set_epi32<Bits>(0xA54FF53A),
	//	simd_set_epi32<Bits>(0x510E527F),
	//	simd_set_epi32<Bits>(0x9B05688C),
	//	simd_set_epi32<Bits>(0x1F83D9AB),
	//	simd_set_epi32<Bits>(0x5BE0CD19)
	//};

	static std::array<simd_type<Bits>, SHA256_STATE_LENGTH> state {
		simd_set_epi32<Bits>(0x6A09E667),
		simd_set_epi32<Bits>(0xBB67AE85),
		simd_set_epi32<Bits>(0x3C6EF372),
		simd_set_epi32<Bits>(0xA54FF53A),
		simd_set_epi32<Bits>(0x510E527F),
		simd_set_epi32<Bits>(0x9B05688C),
		simd_set_epi32<Bits>(0x1F83D9AB),
		simd_set_epi32<Bits>(0x5BE0CD19)
	};

	return state;
}


// sha256_k_constants ********************************************************
template <size_t Bits>
/*constexpr*/
auto sha256_k_constants() {
	/*constexpr*/
	static const simd_type<Bits> k[64] = {
		simd_set_epi32<Bits>(0x428a2f98),
		simd_set_epi32<Bits>(0x71374491),
		simd_set_epi32<Bits>(0xb5c0fbcf),
		simd_set_epi32<Bits>(0xe9b5dba5),
		simd_set_epi32<Bits>(0x3956c25b),
		simd_set_epi32<Bits>(0x59f111f1),
		simd_set_epi32<Bits>(0x923f82a4),
		simd_set_epi32<Bits>(0xab1c5ed5),
		simd_set_epi32<Bits>(0xd807aa98),
		simd_set_epi32<Bits>(0x12835b01),
		simd_set_epi32<Bits>(0x243185be),
		simd_set_epi32<Bits>(0x550c7dc3),
		simd_set_epi32<Bits>(0x72be5d74),
		simd_set_epi32<Bits>(0x80deb1fe),
		simd_set_epi32<Bits>(0x9bdc06a7),
		simd_set_epi32<Bits>(0xc19bf174),
		simd_set_epi32<Bits>(0xe49b69c1),
		simd_set_epi32<Bits>(0xefbe4786),
		simd_set_epi32<Bits>(0x0fc19dc6),
		simd_set_epi32<Bits>(0x240ca1cc),
		simd_set_epi32<Bits>(0x2de92c6f),
		simd_set_epi32<Bits>(0x4a7484aa),
		simd_set_epi32<Bits>(0x5cb0a9dc),
		simd_set_epi32<Bits>(0x76f988da),
		simd_set_epi32<Bits>(0x983e5152),
		simd_set_epi32<Bits>(0xa831c66d),
		simd_set_epi32<Bits>(0xb00327c8),
		simd_set_epi32<Bits>(0xbf597fc7),
		simd_set_epi32<Bits>(0xc6e00bf3),
		simd_set_epi32<Bits>(0xd5a79147),
		simd_set_epi32<Bits>(0x06ca6351),
		simd_set_epi32<Bits>(0x14292967),
		simd_set_epi32<Bits>(0x27b70a85),
		simd_set_epi32<Bits>(0x2e1b2138),
		simd_set_epi32<Bits>(0x4d2c6dfc),
		simd_set_epi32<Bits>(0x53380d13),
		simd_set_epi32<Bits>(0x650a7354),
		simd_set_epi32<Bits>(0x766a0abb),
		simd_set_epi32<Bits>(0x81c2c92e),
		simd_set_epi32<Bits>(0x92722c85),
		simd_set_epi32<Bits>(0xa2bfe8a1),
		simd_set_epi32<Bits>(0xa81a664b),
		simd_set_epi32<Bits>(0xc24b8b70),
		simd_set_epi32<Bits>(0xc76c51a3),
		simd_set_epi32<Bits>(0xd192e819),
		simd_set_epi32<Bits>(0xd6990624),
		simd_set_epi32<Bits>(0xf40e3585),
		simd_set_epi32<Bits>(0x106aa070),
		simd_set_epi32<Bits>(0x19a4c116),
		simd_set_epi32<Bits>(0x1e376c08),
		simd_set_epi32<Bits>(0x2748774c),
		simd_set_epi32<Bits>(0x34b0bcb5),
		simd_set_epi32<Bits>(0x391c0cb3),
		simd_set_epi32<Bits>(0x4ed8aa4a),
		simd_set_epi32<Bits>(0x5b9cca4f),
		simd_set_epi32<Bits>(0x682e6ff3),
		simd_set_epi32<Bits>(0x748f82ee),
		simd_set_epi32<Bits>(0x78a5636f),
		simd_set_epi32<Bits>(0x84c87814),
		simd_set_epi32<Bits>(0x8cc70208),
		simd_set_epi32<Bits>(0x90befffa),
		simd_set_epi32<Bits>(0xa4506ceb),
		simd_set_epi32<Bits>(0xbef9a3f7),
		simd_set_epi32<Bits>(0xc67178f2)
	};
	return k;
}
