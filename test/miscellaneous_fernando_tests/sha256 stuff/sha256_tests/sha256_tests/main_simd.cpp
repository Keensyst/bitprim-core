// clang++ -O3 -std=c++14 sha256_simd_analysis.cpp
// g++ -O3 -std=c++14 sha256_simd_analysis.cpp

//#include <intrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "sha256.h"
#include "zeroize.h"
#include <string>

#include "main_simd.h"


void print_bytes(uint8_t const* data, size_t n) {
    while (n != 0) {
        printf("%02x", *data);
        ++data;
        --n;
    }
    printf("\n");
}



// ----------------------------------------------------------------
// ----------------------------------------------------------------
// SIMD - SSE3 (from Parallelizing message schedules to accelerate the computations of hash functions)
// ----------------------------------------------------------------
// ----------------------------------------------------------------



// this function takes a word from each chunk, and puts it in a single register
//inline
//__m128i gather_old(uint32_t const* address) {
//    __m128i temp;
//    temp = _mm_cvtsi32_si128(address[0]);
//    temp = _mm_insert_epi32(temp, address[16], 1);
//    temp = _mm_insert_epi32(temp, address[32], 2);
//    temp = _mm_insert_epi32(temp, address[48], 3);
//    return temp;
//}
//
//inline
//__m128i gather(uint32_t const* address) {
//    return _mm_set_epi32 (address[48], address[32], address[16], address[0]);
//}

inline
__m128i gather_be(uint32_t const* address) {
    static const __m128i bswap_mask = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    
    auto tmp = _mm_set_epi32 (address[48], address[32], address[16], address[0]);
    return _mm_shuffle_epi8(tmp, bswap_mask);
}

inline
__m256i gather_64_be(uint32_t const* address) {
    static const __m256i bswap_mask = _mm256_set_epi8(28, 29, 30, 31, 24, 25, 26, 27, 20, 21, 22, 23, 16, 17, 18, 19,
                                                      12, 13, 14, 15,  8,  9, 10, 11,  4,  5,  6,  7,  0,  1,  2, 3);
    
    auto tmp = _mm256_set_epi32 (address[112], address[96], address[80], address[64], address[48], address[32], address[16], address[0]);
    
    return _mm256_shuffle_epi8(tmp, bswap_mask);
}




// this function calculates the small sigma 0 transformation
inline
__m128i sigma_0(__m128i W) {
	return
		_mm_xor_si128(
			_mm_xor_si128(
				_mm_xor_si128(
					_mm_srli_epi32(W, 7),
					_mm_srli_epi32(W, 18)
				),
				_mm_xor_si128(
					_mm_srli_epi32(W, 3),
					_mm_slli_epi32(W, 25)
				)
			),
			_mm_slli_epi32(W, 14)
		);
}

// this function calculates the small sigma 1 transformation
inline
__m128i sigma_1(__m128i W) {
	return
		_mm_xor_si128(
			_mm_xor_si128(
				_mm_xor_si128(
					_mm_srli_epi32(W, 17),
					_mm_srli_epi32(W, 10)
				),
				_mm_xor_si128(
					_mm_srli_epi32(W, 19),
					_mm_slli_epi32(W, 15)
				)
			),
			_mm_slli_epi32(W, 13)
		);
}

// the message scheduling round
#define SCHEDULE_ROUND(w1, w2, w3, w4) \
    s0 = sigma_0(w1); \
    s1 = sigma_1(w2); \
    schedule[i] = _mm_add_epi32(w3, Ki[i]); \
    w3 = _mm_add_epi32( \
        _mm_add_epi32(w3, w4), \
        _mm_add_epi32(s0, s1) \
    ); \
    i++;


// this function calculates the small sigma 0 transformation
inline
__m256i sigma_8_0(__m256i W) {
    return
    _mm256_xor_si256(
                  _mm256_xor_si256(
                                _mm256_xor_si256(
                                              _mm256_srli_epi32(W, 7),
                                              _mm256_srli_epi32(W, 18)
                                              ),
                                _mm256_xor_si256(
                                              _mm256_srli_epi32(W, 3),
                                              _mm256_slli_epi32(W, 25)
                                              )
                                ),
                  _mm256_slli_epi32(W, 14)
                  );
}

// this function calculates the small sigma 1 transformation
inline
__m256i sigma_8_1(__m256i W) {
    return
    _mm256_xor_si256(
                  _mm256_xor_si256(
                                _mm256_xor_si256(
                                              _mm256_srli_epi32(W, 17),
                                              _mm256_srli_epi32(W, 10)
                                              ),
                                _mm256_xor_si256(
                                              _mm256_srli_epi32(W, 19),
                                              _mm256_slli_epi32(W, 15)
                                              )
                                ),
                  _mm256_slli_epi32(W, 13)
                  );
}

#define SCHEDULE_ROUND_8(w1, w2, w3, w4) \
    s0 = sigma_8_0(w1); \
    s1 = sigma_8_1(w2); \
    schedule[i] = _mm256_add_epi32(w3, Ki[i]); \
    w3 = _mm256_add_epi32( \
        _mm256_add_epi32(w3, w4), \
    _mm256_add_epi32(s0, s1) \
    ); \
    i++;


inline
__m128i _mm_set_epi32_4(int x) {
	return _mm_set_epi32(x, x, x, x);
}

inline
__m256i _mm_set_epi32_8(int x) {
    return _mm256_set_epi32(x, x, x, x, x, x, x, x);
}

//inline
//__m512i _mm_set_epi32_16(int x) {
//    return _mm512_set_epi32(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x);
//}


#define Ch(x, y, z)  ((x & (y ^ z)) ^ z)
#define Maj(x, y, z) ((x & (y | z)) | (y & z))
#define SHR(x, n)    (x >> n)
#define ROTR(x, n)   ((x >> n) | (x << (32 - n)))
#define S0(x)        (ROTR(x,  2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x)        (ROTR(x,  6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define s0(x)        (ROTR(x,  7) ^ ROTR(x, 18) ^  SHR(x,  3))
#define s1(x)        (ROTR(x, 17) ^ ROTR(x, 19) ^  SHR(x, 10))

//#define Ch_Simd(x, y, z)  (_mm_xor_si128(_mm_and_si128(x, _mm_xor_si128(y, z)), z))								//((x & (y ^ z)) ^ z)
//#define Maj_Simd(x, y, z) (_mm_or_si128(_mm_and_si128(x, _mm_or_si128(y, z)), _mm_and_si128(y, z)))				//((x & (y | z)) | (y & z))
//#define SHR_Simd(x, n)    (_mm_srli_epi32(x, n))																//(x >> n)
//#define ROTR_Simd(x, n)   (_mm_or_si128(_mm_srli_epi32(x, n), _mm_slli_epi32(x, (32 - n))))						//((x >> n) | (x << (32 - n)))
//#define S0_Simd(x)        (_mm_xor_si128(ROTR_Simd(x, 2), _mm_xor_si128(ROTR_Simd(x, 13), ROTR_Simd(x, 22))))	//(ROTR_Simd(x,  2) ^ ROTR_Simd(x, 13) ^ ROTR_Simd(x, 22))
//#define S1_Simd(x)        (_mm_xor_si128(ROTR_Simd(x, 6), _mm_xor_si128(ROTR_Simd(x, 11), ROTR_Simd(x, 25))))	//(ROTR_Simd(x,  6) ^ ROTR_Simd(x, 11) ^ ROTR_Simd(x, 25))	
//#define s0_Simd(x)        (_mm_xor_si128(ROTR_Simd(x, 7), _mm_xor_si128(ROTR_Simd(x, 18), SHR_Simd(x, 3))))		//(ROTR(x,  7) ^ ROTR(x, 18) ^  SHR(x,  3))
//#define s1_Simd(x)        (_mm_xor_si128(ROTR_Simd(x, 17), _mm_xor_si128(ROTR_Simd(x, 19), SHR_Simd(x, 10))))	//(ROTR(x, 17) ^ ROTR(x, 19) ^  SHR(x, 10))


inline
__m128i Ch_Simd(__m128i x, __m128i y, __m128i z) {
    return _mm_xor_si128(_mm_and_si128(x, _mm_xor_si128(y, z)), z);
    //((x & (y ^ z)) ^ z)
}

inline
__m128i Maj_Simd(__m128i x, __m128i y, __m128i z) {
    return _mm_or_si128(_mm_and_si128(x, _mm_or_si128(y, z)), _mm_and_si128(y, z));
    //((x & (y | z)) | (y & z))
}

inline
__m128i SHR_Simd(__m128i x, int n) {
    return _mm_srli_epi32(x, n);
    //(x >> n)
}

inline
__m128i ROTR_Simd(__m128i x, int n) {
    return _mm_or_si128(_mm_srli_epi32(x, n), _mm_slli_epi32(x, (32 - n)));
    //((x >> n) | (x << (32 - n)))
}

inline
__m128i S0_Simd(__m128i x) {
    return _mm_xor_si128(ROTR_Simd(x, 2), _mm_xor_si128(ROTR_Simd(x, 13), ROTR_Simd(x, 22)));
    //(ROTR_Simd(x,  2) ^ ROTR_Simd(x, 13) ^ ROTR_Simd(x, 22))
}

inline
__m128i S1_Simd(__m128i x) {
    return _mm_xor_si128(ROTR_Simd(x, 6), _mm_xor_si128(ROTR_Simd(x, 11), ROTR_Simd(x, 25)));
    //(ROTR_Simd(x,  6) ^ ROTR_Simd(x, 11) ^ ROTR_Simd(x, 25))
}

inline
__m128i s0_Simd(__m128i x) {
    return _mm_xor_si128(ROTR_Simd(x, 7), _mm_xor_si128(ROTR_Simd(x, 18), SHR_Simd(x, 3)));
    //(ROTR(x,  7) ^ ROTR(x, 18) ^  SHR(x,  3))
}

inline
__m128i s1_Simd(__m128i x) {
    return _mm_xor_si128(ROTR_Simd(x, 17), _mm_xor_si128(ROTR_Simd(x, 19), SHR_Simd(x, 10)));
    //(ROTR(x, 17) ^ ROTR(x, 19) ^  SHR(x, 10))
}






//#define RND(a, b, c, d, e, f, g, h, k) \
//    t0 = h + S1(e) + Ch(e, f, g) + k;  \
//    t1 = S0(a) + Maj(a, b, c); \
//    d += t0; \
//    h = t0 + t1;

//#define RNDr(S, i, sch, idx) \
//    RND(S[(64 - i) % 8], S[(65 - i) % 8], \
//    S[(66 - i) % 8], S[(67 - i) % 8], \
//    S[(68 - i) % 8], S[(69 - i) % 8], \
//    S[(70 - i) % 8], S[(71 - i) % 8], \
//    _mm_extract_epi32(sch[i], idx))
//	

inline
void RND(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t& e, uint32_t& f, uint32_t& g, uint32_t& h, uint32_t k) {
	auto t0 = h + S1(e) + Ch(e, f, g) + k;
	auto t1 = S0(a) + Maj(a, b, c);
	d += t0;
	h = t0 + t1;
}

template <size_t idx>
inline
void RNDr(uint32_t S[SHA256_STATE_LENGTH], size_t i, __m128i sch[64]) {
	RND(S[(64 - i) % 8],
		S[(65 - i) % 8],
		S[(66 - i) % 8],
		S[(67 - i) % 8],
		S[(68 - i) % 8],
		S[(69 - i) % 8],
		S[(70 - i) % 8],
		S[(71 - i) % 8],
		_mm_extract_epi32(sch[i], idx));
}

inline
void RND_Simd(__m128i& a, __m128i& b, __m128i& c, __m128i& d, __m128i& e, __m128i& f, __m128i& g, __m128i& h, __m128i k) {
	//auto t0 = h + S1_Simd(e) + Ch_Simd(e, f, g) + k;
	//auto t1 = S0_Simd(a) + Maj_Simd(a, b, c);
	//d += t0;
	//h = t0 + t1;

	auto t0 = _mm_add_epi32(_mm_add_epi32(_mm_add_epi32(h, S1_Simd(e)), Ch_Simd(e, f, g)), k);
	auto t1 = _mm_add_epi32(S0_Simd(a), Maj_Simd(a, b, c));
	d = _mm_add_epi32(d, t0);
	h = _mm_add_epi32(t0, t1);
}

inline
void RNDr_Simd(__m128i S[SHA256_STATE_LENGTH], size_t i, __m128i sch[64]) {
	RND_Simd(S[(64 - i) % 8],
		S[(65 - i) % 8],
		S[(66 - i) % 8],
		S[(67 - i) % 8],
		S[(68 - i) % 8],
		S[(69 - i) % 8],
		S[(70 - i) % 8],
		S[(71 - i) % 8],
		sch[i]);
}

void SHA256_QMS(__m128i schedule[64], uint32_t message[64]) {
//	__m128i bswap_mask = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
//	__m128i W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15;

	static const __m128i k[64] = {
		_mm_set_epi32_4(0x428a2f98),
		_mm_set_epi32_4(0x71374491),
		_mm_set_epi32_4(0xb5c0fbcf),
		_mm_set_epi32_4(0xe9b5dba5),
		_mm_set_epi32_4(0x3956c25b),
		_mm_set_epi32_4(0x59f111f1),
		_mm_set_epi32_4(0x923f82a4),
		_mm_set_epi32_4(0xab1c5ed5),
		_mm_set_epi32_4(0xd807aa98),
		_mm_set_epi32_4(0x12835b01),
		_mm_set_epi32_4(0x243185be),
		_mm_set_epi32_4(0x550c7dc3),
		_mm_set_epi32_4(0x72be5d74),
		_mm_set_epi32_4(0x80deb1fe),
		_mm_set_epi32_4(0x9bdc06a7),
		_mm_set_epi32_4(0xc19bf174),
		_mm_set_epi32_4(0xe49b69c1),
		_mm_set_epi32_4(0xefbe4786),
		_mm_set_epi32_4(0x0fc19dc6),
		_mm_set_epi32_4(0x240ca1cc),
		_mm_set_epi32_4(0x2de92c6f),
		_mm_set_epi32_4(0x4a7484aa),
		_mm_set_epi32_4(0x5cb0a9dc),
		_mm_set_epi32_4(0x76f988da),
		_mm_set_epi32_4(0x983e5152),
		_mm_set_epi32_4(0xa831c66d),
		_mm_set_epi32_4(0xb00327c8),
		_mm_set_epi32_4(0xbf597fc7),
		_mm_set_epi32_4(0xc6e00bf3),
		_mm_set_epi32_4(0xd5a79147),
		_mm_set_epi32_4(0x06ca6351),
		_mm_set_epi32_4(0x14292967),
		_mm_set_epi32_4(0x27b70a85),
		_mm_set_epi32_4(0x2e1b2138),
		_mm_set_epi32_4(0x4d2c6dfc),
		_mm_set_epi32_4(0x53380d13),
		_mm_set_epi32_4(0x650a7354),
		_mm_set_epi32_4(0x766a0abb),
		_mm_set_epi32_4(0x81c2c92e),
		_mm_set_epi32_4(0x92722c85),
		_mm_set_epi32_4(0xa2bfe8a1),
		_mm_set_epi32_4(0xa81a664b),
		_mm_set_epi32_4(0xc24b8b70),
		_mm_set_epi32_4(0xc76c51a3),
		_mm_set_epi32_4(0xd192e819),
		_mm_set_epi32_4(0xd6990624),
		_mm_set_epi32_4(0xf40e3585),
		_mm_set_epi32_4(0x106aa070),
		_mm_set_epi32_4(0x19a4c116),
		_mm_set_epi32_4(0x1e376c08),
		_mm_set_epi32_4(0x2748774c),
		_mm_set_epi32_4(0x34b0bcb5),
		_mm_set_epi32_4(0x391c0cb3),
		_mm_set_epi32_4(0x4ed8aa4a),
		_mm_set_epi32_4(0x5b9cca4f),
		_mm_set_epi32_4(0x682e6ff3),
		_mm_set_epi32_4(0x748f82ee),
		_mm_set_epi32_4(0x78a5636f),
		_mm_set_epi32_4(0x84c87814),
		_mm_set_epi32_4(0x8cc70208),
		_mm_set_epi32_4(0x90befffa),
		_mm_set_epi32_4(0xa4506ceb),
		_mm_set_epi32_4(0xbef9a3f7),
		_mm_set_epi32_4(0xc67178f2) };

	__m128i s0, s1, *Ki = (__m128i*)k;

    
    

    auto W0 = gather_be(message);
    auto W1 = gather_be(&message[1]);
    auto W2 = gather_be(&message[2]);
    auto W3 = gather_be(&message[3]);
    auto W4 = gather_be(&message[4]);
    auto W5 = gather_be(&message[5]);
    auto W6 = gather_be(&message[6]);
    auto W7 = gather_be(&message[7]);
    auto W8 = gather_be(&message[8]);
    auto W9 = gather_be(&message[9]);
    auto W10 = gather_be(&message[10]);
    auto W11 = gather_be(&message[11]);
    auto W12 = gather_be(&message[12]);
    auto W13 = gather_be(&message[13]);
    auto W14 = gather_be(&message[14]);
    auto W15 = gather_be(&message[15]);
    
    
    int i;
    for (i = 0; i < 32; ) {
		SCHEDULE_ROUND(W1, W14, W0, W9);
		SCHEDULE_ROUND(W2, W15, W1, W10);
		SCHEDULE_ROUND(W3, W0, W2, W11);
		SCHEDULE_ROUND(W4, W1, W3, W12);
		SCHEDULE_ROUND(W5, W2, W4, W13);
		SCHEDULE_ROUND(W6, W3, W5, W14);
		SCHEDULE_ROUND(W7, W4, W6, W15);
		SCHEDULE_ROUND(W8, W5, W7, W0);
		SCHEDULE_ROUND(W9, W6, W8, W1);
		SCHEDULE_ROUND(W10, W7, W9, W2);
		SCHEDULE_ROUND(W11, W8, W10, W3);
		SCHEDULE_ROUND(W12, W9, W11, W4);
		SCHEDULE_ROUND(W13, W10, W12, W5);
		SCHEDULE_ROUND(W14, W11, W13, W6);
		SCHEDULE_ROUND(W15, W12, W14, W7);
		SCHEDULE_ROUND(W0, W13, W15, W8);
	}

	SCHEDULE_ROUND(W1, W14, W0, W9);
	schedule[48] = _mm_add_epi32(W0, Ki[48]);
	SCHEDULE_ROUND(W2, W15, W1, W10);
	schedule[49] = _mm_add_epi32(W1, Ki[49]);
	SCHEDULE_ROUND(W3, W0, W2, W11);
	schedule[50] = _mm_add_epi32(W2, Ki[50]);
	SCHEDULE_ROUND(W4, W1, W3, W12);
	schedule[51] = _mm_add_epi32(W3, Ki[51]);
	SCHEDULE_ROUND(W5, W2, W4, W13);
	schedule[52] = _mm_add_epi32(W4, Ki[52]);
	SCHEDULE_ROUND(W6, W3, W5, W14);
	schedule[53] = _mm_add_epi32(W5, Ki[53]);
	SCHEDULE_ROUND(W7, W4, W6, W15);
	schedule[54] = _mm_add_epi32(W6, Ki[54]);
	SCHEDULE_ROUND(W8, W5, W7, W0);
	schedule[55] = _mm_add_epi32(W7, Ki[55]);
	SCHEDULE_ROUND(W9, W6, W8, W1);
	schedule[56] = _mm_add_epi32(W8, Ki[56]);
	SCHEDULE_ROUND(W10, W7, W9, W2);
	schedule[57] = _mm_add_epi32(W9, Ki[57]);
	SCHEDULE_ROUND(W11, W8, W10, W3);
	schedule[58] = _mm_add_epi32(W10, Ki[58]);
	SCHEDULE_ROUND(W12, W9, W11, W4);
	schedule[59] = _mm_add_epi32(W11, Ki[59]);
	SCHEDULE_ROUND(W13, W10, W12, W5);
	schedule[60] = _mm_add_epi32(W12, Ki[60]);
	SCHEDULE_ROUND(W14, W11, W13, W6);
	schedule[61] = _mm_add_epi32(W13, Ki[61]);
	SCHEDULE_ROUND(W15, W12, W14, W7);
	schedule[62] = _mm_add_epi32(W14, Ki[62]);
	SCHEDULE_ROUND(W0, W13, W15, W8);
	schedule[63] = _mm_add_epi32(W15, Ki[63]);



	//for (size_t i = 0; i < 64; ++i)
	//{
	//	printf("%#010x\n", schedule[i].m128i_i32[0]);  // gives 0x00000007
	//}

}




template <size_t idx>
void SHA256_SIMD_Step(uint32_t state[SHA256_STATE_LENGTH], __m128i schedule[64]) {

	uint32_t S[8];
	memcpy(S, state, 32);
	
	RNDr<idx>(S, 0, schedule);
	RNDr<idx>(S, 1, schedule);
	RNDr<idx>(S, 2, schedule);
	RNDr<idx>(S, 3, schedule);
	RNDr<idx>(S, 4, schedule);
	RNDr<idx>(S, 5, schedule);
	RNDr<idx>(S, 6, schedule);
	RNDr<idx>(S, 7, schedule);
	RNDr<idx>(S, 8, schedule);
	RNDr<idx>(S, 9, schedule);
	RNDr<idx>(S, 10, schedule);
	RNDr<idx>(S, 11, schedule);
	RNDr<idx>(S, 12, schedule);
	RNDr<idx>(S, 13, schedule);
	RNDr<idx>(S, 14, schedule);
	RNDr<idx>(S, 15, schedule);

	RNDr<idx>(S, 16, schedule);
	RNDr<idx>(S, 17, schedule);
	RNDr<idx>(S, 18, schedule);
	RNDr<idx>(S, 19, schedule);
	RNDr<idx>(S, 20, schedule);
	RNDr<idx>(S, 21, schedule);
	RNDr<idx>(S, 22, schedule);
	RNDr<idx>(S, 23, schedule);
	RNDr<idx>(S, 24, schedule);
	RNDr<idx>(S, 25, schedule);
	RNDr<idx>(S, 26, schedule);
	RNDr<idx>(S, 27, schedule);
	RNDr<idx>(S, 28, schedule);
	RNDr<idx>(S, 29, schedule);
	RNDr<idx>(S, 30, schedule);
	RNDr<idx>(S, 31, schedule);

	RNDr<idx>(S, 32, schedule);
	RNDr<idx>(S, 33, schedule);
	RNDr<idx>(S, 34, schedule);
	RNDr<idx>(S, 35, schedule);
	RNDr<idx>(S, 36, schedule);
	RNDr<idx>(S, 37, schedule);
	RNDr<idx>(S, 38, schedule);
	RNDr<idx>(S, 39, schedule);
	RNDr<idx>(S, 40, schedule);
	RNDr<idx>(S, 41, schedule);
	RNDr<idx>(S, 42, schedule);
	RNDr<idx>(S, 43, schedule);
	RNDr<idx>(S, 44, schedule);
	RNDr<idx>(S, 45, schedule);
	RNDr<idx>(S, 46, schedule);
	RNDr<idx>(S, 47, schedule);

	RNDr<idx>(S, 48, schedule);
	RNDr<idx>(S, 49, schedule);
	RNDr<idx>(S, 50, schedule);
	RNDr<idx>(S, 51, schedule);
	RNDr<idx>(S, 52, schedule);
	RNDr<idx>(S, 53, schedule);
	RNDr<idx>(S, 54, schedule);
	RNDr<idx>(S, 55, schedule);
	RNDr<idx>(S, 56, schedule);
	RNDr<idx>(S, 57, schedule);
	RNDr<idx>(S, 58, schedule);
	RNDr<idx>(S, 59, schedule);
	RNDr<idx>(S, 60, schedule);
	RNDr<idx>(S, 61, schedule);
	RNDr<idx>(S, 62, schedule);
	RNDr<idx>(S, 63, schedule);

	for (size_t i = 0; i < 8; ++i) {
		state[i] += S[i];
		//S[i] = context.state[i];
	}

}

void SHA256_SIMD2_Step(__m128i state[SHA256_STATE_LENGTH], __m128i schedule[64]) {

	__m128i S[8];
	memcpy(S, state, 32 * 4);

	RNDr_Simd(S, 0, schedule);
	RNDr_Simd(S, 1, schedule);
	RNDr_Simd(S, 2, schedule);
	RNDr_Simd(S, 3, schedule);
	RNDr_Simd(S, 4, schedule);
	RNDr_Simd(S, 5, schedule);
	RNDr_Simd(S, 6, schedule);
	RNDr_Simd(S, 7, schedule);
	RNDr_Simd(S, 8, schedule);
	RNDr_Simd(S, 9, schedule);
	RNDr_Simd(S, 10, schedule);
	RNDr_Simd(S, 11, schedule);
	RNDr_Simd(S, 12, schedule);
	RNDr_Simd(S, 13, schedule);
	RNDr_Simd(S, 14, schedule);
	RNDr_Simd(S, 15, schedule);

	RNDr_Simd(S, 16, schedule);
	RNDr_Simd(S, 17, schedule);
	RNDr_Simd(S, 18, schedule);
	RNDr_Simd(S, 19, schedule);
	RNDr_Simd(S, 20, schedule);
	RNDr_Simd(S, 21, schedule);
	RNDr_Simd(S, 22, schedule);
	RNDr_Simd(S, 23, schedule);
	RNDr_Simd(S, 24, schedule);
	RNDr_Simd(S, 25, schedule);
	RNDr_Simd(S, 26, schedule);
	RNDr_Simd(S, 27, schedule);
	RNDr_Simd(S, 28, schedule);
	RNDr_Simd(S, 29, schedule);
	RNDr_Simd(S, 30, schedule);
	RNDr_Simd(S, 31, schedule);

	RNDr_Simd(S, 32, schedule);
	RNDr_Simd(S, 33, schedule);
	RNDr_Simd(S, 34, schedule);
	RNDr_Simd(S, 35, schedule);
	RNDr_Simd(S, 36, schedule);
	RNDr_Simd(S, 37, schedule);
	RNDr_Simd(S, 38, schedule);
	RNDr_Simd(S, 39, schedule);
	RNDr_Simd(S, 40, schedule);
	RNDr_Simd(S, 41, schedule);
	RNDr_Simd(S, 42, schedule);
	RNDr_Simd(S, 43, schedule);
	RNDr_Simd(S, 44, schedule);
	RNDr_Simd(S, 45, schedule);
	RNDr_Simd(S, 46, schedule);
	RNDr_Simd(S, 47, schedule);

	RNDr_Simd(S, 48, schedule);
	RNDr_Simd(S, 49, schedule);
	RNDr_Simd(S, 50, schedule);
	RNDr_Simd(S, 51, schedule);
	RNDr_Simd(S, 52, schedule);
	RNDr_Simd(S, 53, schedule);
	RNDr_Simd(S, 54, schedule);
	RNDr_Simd(S, 55, schedule);
	RNDr_Simd(S, 56, schedule);
	RNDr_Simd(S, 57, schedule);
	RNDr_Simd(S, 58, schedule);
	RNDr_Simd(S, 59, schedule);
	RNDr_Simd(S, 60, schedule);
	RNDr_Simd(S, 61, schedule);
	RNDr_Simd(S, 62, schedule);
	RNDr_Simd(S, 63, schedule);

	for (size_t i = 0; i < 8; ++i) {
		//state[i] += S[i];
		state[i] = _mm_add_epi32(state[i], S[i]);
	}
}

void SHA256Transform_SIMD1(uint32_t state[SHA256_STATE_LENGTH], uint8_t const input[SHA256_BLOCK_LENGTH * 4]) {
	__m128i schedule[64] = {};

	//uint32_t message[64];
	//std::memcpy(message + 0, input + 0, 64);
	//std::memcpy(message + 16, input + 64, 64);
	//std::memcpy(message + 32, input + 128, 64);
	//std::memcpy(message + 48, input + 192, 64);
	//SHA256_QMS(schedule, message);

	SHA256_QMS(schedule, (uint32_t*)input);

	SHA256_SIMD_Step<0>(state, schedule);
	SHA256_SIMD_Step<1>(state, schedule);
	SHA256_SIMD_Step<2>(state, schedule);
	SHA256_SIMD_Step<3>(state, schedule);
}

void SHA256Transform_SIMD2(__m128i state[SHA256_STATE_LENGTH], uint8_t const input[SHA256_BLOCK_LENGTH * 4]) {
	__m128i schedule[64] = {};

	//uint32_t message[64];
	//std::memcpy(message + 0, input + 0, 64);
	//std::memcpy(message + 16, input + 64, 64);
	//std::memcpy(message + 32, input + 128, 64);
	//std::memcpy(message + 48, input + 192, 64);
	//SHA256_QMS(schedule, message);

	SHA256_QMS(schedule, (uint32_t*)input);
	SHA256_SIMD2_Step(state, schedule);
}

void copy_length_to_buffer(uint8_t* buffer, size_t length) {
	uint32_t bitlen[2];
	bitlen[1] = ((uint32_t)length) << 3;
	bitlen[0] = (uint32_t)(length >> 29);
	be32enc_vect(buffer, bitlen, 8);
}

void SHA256Init_State_SIMD(__m128i state[SHA256_STATE_LENGTH]) {
	state[0] = _mm_set_epi32_4(0x6A09E667);
	state[1] = _mm_set_epi32_4(0xBB67AE85);
	state[2] = _mm_set_epi32_4(0x3C6EF372);
	state[3] = _mm_set_epi32_4(0xA54FF53A);
	state[4] = _mm_set_epi32_4(0x510E527F);
	state[5] = _mm_set_epi32_4(0x9B05688C);
	state[6] = _mm_set_epi32_4(0x1F83D9AB);
	state[7] = _mm_set_epi32_4(0x5BE0CD19);
}

void SHA256Init_State_SIMD_x8(__m256i state[SHA256_STATE_LENGTH]) {
    state[0] = _mm_set_epi32_8(0x6A09E667);
    state[1] = _mm_set_epi32_8(0xBB67AE85);
    state[2] = _mm_set_epi32_8(0x3C6EF372);
    state[3] = _mm_set_epi32_8(0xA54FF53A);
    state[4] = _mm_set_epi32_8(0x510E527F);
    state[5] = _mm_set_epi32_8(0x9B05688C);
    state[6] = _mm_set_epi32_8(0x1F83D9AB);
    state[7] = _mm_set_epi32_8(0x5BE0CD19);
}



//----------------------------------------------------------------------------


void SHA256_SIMD_1(uint8_t const* input, size_t length, uint8_t digest[SHA256_DIGEST_LENGTH]) {
	SHA256CTX context;
	SHA256Init(&context);

	size_t orig_length = length;

	while (length >= 256) {
		SHA256Transform_SIMD1(context.state, input);
		input += 256;
		length -= 256;
	}

	while (length >= 64) {
		SHA256Transform(context.state, input);
		input += 64;
		length -= 64;
	}

	uint8_t buffer[SHA256_BLOCK_LENGTH] = {};
	std::memcpy(buffer, input, length);
	buffer[length] = 0x80;
	copy_length_to_buffer(buffer + 56, orig_length);
	SHA256Transform(context.state, buffer);
	be32enc_vect(digest, context.state, SHA256_DIGEST_LENGTH);
	//zeroize((void*)&context, sizeof(context));





	
	//uint8_t PADXX[SHA256_BLOCK_LENGTH] = {
	//	0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
	//	0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
	//	0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
	//	0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x08, 0
	//};

	//SHA256Transform(context.state, PADXX);

	//be32enc_vect(digest, context.state, SHA256_DIGEST_LENGTH);
	//zeroize((void*)&context, sizeof(context));
}


// Esta funcion sirve para procesar 4 "strings" de 64 bytes cada uno (a la vez).
// Cada uno de estos "strings" podría llegar a ser la concatenación de 2 hashes de 32 bytes.
// O sea, puede ser util para procesar el merkle-root
void SHA256_SIMD_2(
	uint8_t const* input1, 
	uint8_t const* input2, 
	uint8_t const* input3, 
	uint8_t const* input4, size_t length,
	uint8_t digest1[SHA256_DIGEST_LENGTH],
	uint8_t digest2[SHA256_DIGEST_LENGTH],
	uint8_t digest3[SHA256_DIGEST_LENGTH],
	uint8_t digest4[SHA256_DIGEST_LENGTH]) {

	__m128i state[SHA256_STATE_LENGTH];
	SHA256Init_State_SIMD(state);


	uint8_t input_tmp[256];
    std::memcpy(input_tmp + 0, input1, 64);
    std::memcpy(input_tmp + 64, input2, 64);
    std::memcpy(input_tmp + 128, input3, 64);
    std::memcpy(input_tmp + 192, input4, 64);
  
	uint8_t const* input = input_tmp;
//	size_t orig_length = length;

	while (length >= 64) {
		SHA256Transform_SIMD2(state, input);
		input += 64;
		length -= 64;
	}

	uint8_t buffer[SHA256_BLOCK_LENGTH * 4] = {
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0
    };

	SHA256Transform_SIMD2(state, buffer);

	uint32_t state1[SHA256_STATE_LENGTH];
	uint32_t state2[SHA256_STATE_LENGTH];
	uint32_t state3[SHA256_STATE_LENGTH];
	uint32_t state4[SHA256_STATE_LENGTH];

	for (size_t i = 0; i < 8; ++i) {
		uint32_t* val = (uint32_t*)&(state[i]);

		state1[i] = val[0];
		state2[i] = val[1];
		state3[i] = val[2];
		state4[i] = val[3];
	}

	be32enc_vect(digest1, state1, SHA256_DIGEST_LENGTH);
	be32enc_vect(digest2, state2, SHA256_DIGEST_LENGTH);
	be32enc_vect(digest3, state3, SHA256_DIGEST_LENGTH);
	be32enc_vect(digest4, state4, SHA256_DIGEST_LENGTH);
}




//--------------------------------------------------------------------------------




void SHA256_QMS_3(__m256i schedule[64], uint32_t message[64 * 2]) {

    static const __m256i k[64] = {
        _mm_set_epi32_8(0x428a2f98),
        _mm_set_epi32_8(0x71374491),
        _mm_set_epi32_8(0xb5c0fbcf),
        _mm_set_epi32_8(0xe9b5dba5),
        _mm_set_epi32_8(0x3956c25b),
        _mm_set_epi32_8(0x59f111f1),
        _mm_set_epi32_8(0x923f82a4),
        _mm_set_epi32_8(0xab1c5ed5),
        _mm_set_epi32_8(0xd807aa98),
        _mm_set_epi32_8(0x12835b01),
        _mm_set_epi32_8(0x243185be),
        _mm_set_epi32_8(0x550c7dc3),
        _mm_set_epi32_8(0x72be5d74),
        _mm_set_epi32_8(0x80deb1fe),
        _mm_set_epi32_8(0x9bdc06a7),
        _mm_set_epi32_8(0xc19bf174),
        _mm_set_epi32_8(0xe49b69c1),
        _mm_set_epi32_8(0xefbe4786),
        _mm_set_epi32_8(0x0fc19dc6),
        _mm_set_epi32_8(0x240ca1cc),
        _mm_set_epi32_8(0x2de92c6f),
        _mm_set_epi32_8(0x4a7484aa),
        _mm_set_epi32_8(0x5cb0a9dc),
        _mm_set_epi32_8(0x76f988da),
        _mm_set_epi32_8(0x983e5152),
        _mm_set_epi32_8(0xa831c66d),
        _mm_set_epi32_8(0xb00327c8),
        _mm_set_epi32_8(0xbf597fc7),
        _mm_set_epi32_8(0xc6e00bf3),
        _mm_set_epi32_8(0xd5a79147),
        _mm_set_epi32_8(0x06ca6351),
        _mm_set_epi32_8(0x14292967),
        _mm_set_epi32_8(0x27b70a85),
        _mm_set_epi32_8(0x2e1b2138),
        _mm_set_epi32_8(0x4d2c6dfc),
        _mm_set_epi32_8(0x53380d13),
        _mm_set_epi32_8(0x650a7354),
        _mm_set_epi32_8(0x766a0abb),
        _mm_set_epi32_8(0x81c2c92e),
        _mm_set_epi32_8(0x92722c85),
        _mm_set_epi32_8(0xa2bfe8a1),
        _mm_set_epi32_8(0xa81a664b),
        _mm_set_epi32_8(0xc24b8b70),
        _mm_set_epi32_8(0xc76c51a3),
        _mm_set_epi32_8(0xd192e819),
        _mm_set_epi32_8(0xd6990624),
        _mm_set_epi32_8(0xf40e3585),
        _mm_set_epi32_8(0x106aa070),
        _mm_set_epi32_8(0x19a4c116),
        _mm_set_epi32_8(0x1e376c08),
        _mm_set_epi32_8(0x2748774c),
        _mm_set_epi32_8(0x34b0bcb5),
        _mm_set_epi32_8(0x391c0cb3),
        _mm_set_epi32_8(0x4ed8aa4a),
        _mm_set_epi32_8(0x5b9cca4f),
        _mm_set_epi32_8(0x682e6ff3),
        _mm_set_epi32_8(0x748f82ee),
        _mm_set_epi32_8(0x78a5636f),
        _mm_set_epi32_8(0x84c87814),
        _mm_set_epi32_8(0x8cc70208),
        _mm_set_epi32_8(0x90befffa),
        _mm_set_epi32_8(0xa4506ceb),
        _mm_set_epi32_8(0xbef9a3f7),
        _mm_set_epi32_8(0xc67178f2) };
    
    __m256i s0, s1, *Ki = (__m256i*)k;
    
    
    
    // *1* Esto carga los primeros 16 elementos en los Wi en BigEndian
    auto W0 = gather_64_be(message);
    auto W1 = gather_64_be(&message[1]);
    auto W2 = gather_64_be(&message[2]);
    auto W3 = gather_64_be(&message[3]);
    auto W4 = gather_64_be(&message[4]);
    auto W5 = gather_64_be(&message[5]);
    auto W6 = gather_64_be(&message[6]);
    auto W7 = gather_64_be(&message[7]);
    auto W8 = gather_64_be(&message[8]);
    auto W9 = gather_64_be(&message[9]);
    auto W10 = gather_64_be(&message[10]);
    auto W11 = gather_64_be(&message[11]);
    auto W12 = gather_64_be(&message[12]);
    auto W13 = gather_64_be(&message[13]);
    auto W14 = gather_64_be(&message[14]);
    auto W15 = gather_64_be(&message[15]);
        // Fin *1*
    
    
    
    int i;
    for (i = 0; i < 32; ) {
        SCHEDULE_ROUND_8(W1, W14, W0, W9);
        SCHEDULE_ROUND_8(W2, W15, W1, W10);
        SCHEDULE_ROUND_8(W3, W0, W2, W11);
        SCHEDULE_ROUND_8(W4, W1, W3, W12);
        SCHEDULE_ROUND_8(W5, W2, W4, W13);
        SCHEDULE_ROUND_8(W6, W3, W5, W14);
        SCHEDULE_ROUND_8(W7, W4, W6, W15);
        SCHEDULE_ROUND_8(W8, W5, W7, W0);
        SCHEDULE_ROUND_8(W9, W6, W8, W1);
        SCHEDULE_ROUND_8(W10, W7, W9, W2);
        SCHEDULE_ROUND_8(W11, W8, W10, W3);
        SCHEDULE_ROUND_8(W12, W9, W11, W4);
        SCHEDULE_ROUND_8(W13, W10, W12, W5);
        SCHEDULE_ROUND_8(W14, W11, W13, W6);
        SCHEDULE_ROUND_8(W15, W12, W14, W7);
        SCHEDULE_ROUND_8(W0, W13, W15, W8);
    }
    
    SCHEDULE_ROUND_8(W1, W14, W0, W9);
    schedule[48] = _mm256_add_epi32(W0, Ki[48]);
    SCHEDULE_ROUND_8(W2, W15, W1, W10);
    schedule[49] = _mm256_add_epi32(W1, Ki[49]);
    SCHEDULE_ROUND_8(W3, W0, W2, W11);
    schedule[50] = _mm256_add_epi32(W2, Ki[50]);
    SCHEDULE_ROUND_8(W4, W1, W3, W12);
    schedule[51] = _mm256_add_epi32(W3, Ki[51]);
    SCHEDULE_ROUND_8(W5, W2, W4, W13);
    schedule[52] = _mm256_add_epi32(W4, Ki[52]);
    SCHEDULE_ROUND_8(W6, W3, W5, W14);
    schedule[53] = _mm256_add_epi32(W5, Ki[53]);
    SCHEDULE_ROUND_8(W7, W4, W6, W15);
    schedule[54] = _mm256_add_epi32(W6, Ki[54]);
    SCHEDULE_ROUND_8(W8, W5, W7, W0);
    schedule[55] = _mm256_add_epi32(W7, Ki[55]);
    SCHEDULE_ROUND_8(W9, W6, W8, W1);
    schedule[56] = _mm256_add_epi32(W8, Ki[56]);
    SCHEDULE_ROUND_8(W10, W7, W9, W2);
    schedule[57] = _mm256_add_epi32(W9, Ki[57]);
    SCHEDULE_ROUND_8(W11, W8, W10, W3);
    schedule[58] = _mm256_add_epi32(W10, Ki[58]);
    SCHEDULE_ROUND_8(W12, W9, W11, W4);
    schedule[59] = _mm256_add_epi32(W11, Ki[59]);
    SCHEDULE_ROUND_8(W13, W10, W12, W5);
    schedule[60] = _mm256_add_epi32(W12, Ki[60]);
    SCHEDULE_ROUND_8(W14, W11, W13, W6);
    schedule[61] = _mm256_add_epi32(W13, Ki[61]);
    SCHEDULE_ROUND_8(W15, W12, W14, W7);
    schedule[62] = _mm256_add_epi32(W14, Ki[62]);
    SCHEDULE_ROUND_8(W0, W13, W15, W8);
    schedule[63] = _mm256_add_epi32(W15, Ki[63]);
}


inline
__m256i Ch_Simd8(__m256i x, __m256i y, __m256i z) {
    return _mm256_xor_si256(_mm256_and_si256(x, _mm256_xor_si256(y, z)), z);
    //((x & (y ^ z)) ^ z)
}

inline
__m256i Maj_Simd8(__m256i x, __m256i y, __m256i z) {
    return _mm256_or_si256(_mm256_and_si256(x, _mm256_or_si256(y, z)), _mm256_and_si256(y, z));
    //((x & (y | z)) | (y & z))
}

inline
__m256i SHR_Simd8(__m256i x, int n) {
    return _mm256_srli_epi32(x, n);
    //(x >> n)
}

inline
__m256i ROTR_Simd8(__m256i x, int n) {
    return _mm256_or_si256(_mm256_srli_epi32(x, n), _mm256_slli_epi32(x, (32 - n)));
    //((x >> n) | (x << (32 - n)))
}

inline
__m256i S0_Simd8(__m256i x) {
    return _mm256_xor_si256(ROTR_Simd8(x, 2), _mm256_xor_si256(ROTR_Simd8(x, 13), ROTR_Simd8(x, 22)));
    //(ROTR_Simd(x,  2) ^ ROTR_Simd(x, 13) ^ ROTR_Simd(x, 22))
}

inline
__m256i S1_Simd8(__m256i x) {
    return _mm256_xor_si256(ROTR_Simd8(x, 6), _mm256_xor_si256(ROTR_Simd8(x, 11), ROTR_Simd8(x, 25)));
    //(ROTR_Simd(x,  6) ^ ROTR_Simd(x, 11) ^ ROTR_Simd(x, 25))
}

inline
__m256i s0_Simd8(__m256i x) {
    return _mm256_xor_si256(ROTR_Simd8(x, 7), _mm256_xor_si256(ROTR_Simd8(x, 18), SHR_Simd8(x, 3)));
    //(ROTR(x,  7) ^ ROTR(x, 18) ^  SHR(x,  3))
}

inline
__m256i s1_Simd8(__m256i x) {
    return _mm256_xor_si256(ROTR_Simd8(x, 17), _mm256_xor_si256(ROTR_Simd8(x, 19), SHR_Simd8(x, 10)));
    //(ROTR(x, 17) ^ ROTR(x, 19) ^  SHR(x, 10))
}



inline
void RND_Simd8(__m256i& a, __m256i& b, __m256i& c, __m256i& d, __m256i& e, __m256i& f, __m256i& g, __m256i& h, __m256i k) {
    auto t0 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(h, S1_Simd8(e)), Ch_Simd8(e, f, g)), k);
    auto t1 = _mm256_add_epi32(S0_Simd8(a), Maj_Simd8(a, b, c));
    d = _mm256_add_epi32(d, t0);
    h = _mm256_add_epi32(t0, t1);
}

inline
void RNDr_Simd8(__m256i S[SHA256_STATE_LENGTH], size_t i, __m256i sch[64]) {
    RND_Simd8(S[(64 - i) % 8],
             S[(65 - i) % 8],
             S[(66 - i) % 8],
             S[(67 - i) % 8],
             S[(68 - i) % 8],
             S[(69 - i) % 8],
             S[(70 - i) % 8],
             S[(71 - i) % 8],
             sch[i]);
}

void SHA256_SIMD3_Step(__m256i state[SHA256_STATE_LENGTH], __m256i schedule[64]) {
    
    __m256i S[8];
    memcpy(S, state, 32 * 8);
    
    RNDr_Simd8(S, 0, schedule);
    RNDr_Simd8(S, 1, schedule);
    RNDr_Simd8(S, 2, schedule);
    RNDr_Simd8(S, 3, schedule);
    RNDr_Simd8(S, 4, schedule);
    RNDr_Simd8(S, 5, schedule);
    RNDr_Simd8(S, 6, schedule);
    RNDr_Simd8(S, 7, schedule);
    RNDr_Simd8(S, 8, schedule);
    RNDr_Simd8(S, 9, schedule);
    RNDr_Simd8(S, 10, schedule);
    RNDr_Simd8(S, 11, schedule);
    RNDr_Simd8(S, 12, schedule);
    RNDr_Simd8(S, 13, schedule);
    RNDr_Simd8(S, 14, schedule);
    RNDr_Simd8(S, 15, schedule);
    
    RNDr_Simd8(S, 16, schedule);
    RNDr_Simd8(S, 17, schedule);
    RNDr_Simd8(S, 18, schedule);
    RNDr_Simd8(S, 19, schedule);
    RNDr_Simd8(S, 20, schedule);
    RNDr_Simd8(S, 21, schedule);
    RNDr_Simd8(S, 22, schedule);
    RNDr_Simd8(S, 23, schedule);
    RNDr_Simd8(S, 24, schedule);
    RNDr_Simd8(S, 25, schedule);
    RNDr_Simd8(S, 26, schedule);
    RNDr_Simd8(S, 27, schedule);
    RNDr_Simd8(S, 28, schedule);
    RNDr_Simd8(S, 29, schedule);
    RNDr_Simd8(S, 30, schedule);
    RNDr_Simd8(S, 31, schedule);
    
    RNDr_Simd8(S, 32, schedule);
    RNDr_Simd8(S, 33, schedule);
    RNDr_Simd8(S, 34, schedule);
    RNDr_Simd8(S, 35, schedule);
    RNDr_Simd8(S, 36, schedule);
    RNDr_Simd8(S, 37, schedule);
    RNDr_Simd8(S, 38, schedule);
    RNDr_Simd8(S, 39, schedule);
    RNDr_Simd8(S, 40, schedule);
    RNDr_Simd8(S, 41, schedule);
    RNDr_Simd8(S, 42, schedule);
    RNDr_Simd8(S, 43, schedule);
    RNDr_Simd8(S, 44, schedule);
    RNDr_Simd8(S, 45, schedule);
    RNDr_Simd8(S, 46, schedule);
    RNDr_Simd8(S, 47, schedule);
    
    RNDr_Simd8(S, 48, schedule);
    RNDr_Simd8(S, 49, schedule);
    RNDr_Simd8(S, 50, schedule);
    RNDr_Simd8(S, 51, schedule);
    RNDr_Simd8(S, 52, schedule);
    RNDr_Simd8(S, 53, schedule);
    RNDr_Simd8(S, 54, schedule);
    RNDr_Simd8(S, 55, schedule);
    RNDr_Simd8(S, 56, schedule);
    RNDr_Simd8(S, 57, schedule);
    RNDr_Simd8(S, 58, schedule);
    RNDr_Simd8(S, 59, schedule);
    RNDr_Simd8(S, 60, schedule);
    RNDr_Simd8(S, 61, schedule);
    RNDr_Simd8(S, 62, schedule);
    RNDr_Simd8(S, 63, schedule);
    
    for (size_t i = 0; i < 8; ++i) {
        //state[i] += S[i];
        state[i] = _mm256_add_epi32(state[i], S[i]);
    }
}

void SHA256Transform_SIMD3(__m256i state[SHA256_STATE_LENGTH], uint8_t const input[SHA256_BLOCK_LENGTH * 8]) {
    __m256i schedule[64] = {};

	//uint32_t message[128];
 //   std::memcpy(message + 0,   input + 0, 64);
 //   std::memcpy(message + 16,  input + 64, 64);
 //   std::memcpy(message + 32,  input + 128, 64);
 //   std::memcpy(message + 48,  input + 192, 64);
 //   std::memcpy(message + 64,  input + 256, 64);
 //   std::memcpy(message + 80,  input + 320, 64);
 //   std::memcpy(message + 96,  input + 384, 64);
 //   std::memcpy(message + 112, input + 448, 64);
 //   SHA256_QMS_3(schedule, message);

	SHA256_QMS_3(schedule, (uint32_t*)input);

    SHA256_SIMD3_Step(state, schedule);
}

// Esta funcion sirve para procesar 8 "strings" de 64 bytes cada uno (a la vez).
// Cada uno de estos "strings" podría llegar a ser la concatenación de 2 hashes de 32 bytes.
// O sea, puede ser util para procesar el merkle-root
void SHA256_SIMD_3(
                   uint8_t const* input1,
                   uint8_t const* input2,
                   uint8_t const* input3,
                   uint8_t const* input4,
                   uint8_t const* input5,
                   uint8_t const* input6,
                   uint8_t const* input7,
                   uint8_t const* input8, size_t length,
                   uint8_t digest1[SHA256_DIGEST_LENGTH],
                   uint8_t digest2[SHA256_DIGEST_LENGTH],
                   uint8_t digest3[SHA256_DIGEST_LENGTH],
                   uint8_t digest4[SHA256_DIGEST_LENGTH],
                   uint8_t digest5[SHA256_DIGEST_LENGTH],
                   uint8_t digest6[SHA256_DIGEST_LENGTH],
                   uint8_t digest7[SHA256_DIGEST_LENGTH],
                   uint8_t digest8[SHA256_DIGEST_LENGTH]) {
    
    __m256i state[SHA256_STATE_LENGTH];
    SHA256Init_State_SIMD_x8(state);
    
    
    uint8_t input_tmp[512];
    std::memcpy(input_tmp + 0, input1, 64);
    std::memcpy(input_tmp + 64, input2, 64);
    std::memcpy(input_tmp + 128, input3, 64);
    std::memcpy(input_tmp + 192, input4, 64);
    std::memcpy(input_tmp + 256, input5, 64);
    std::memcpy(input_tmp + 320, input6, 64);
    std::memcpy(input_tmp + 384, input7, 64);
    std::memcpy(input_tmp + 448, input8, 64);
    
    uint8_t const* input = input_tmp;
//    size_t orig_length = length;
    
    while (length >= 64) {
        SHA256Transform_SIMD3(state, input);
        input += 64;
        length -= 64;
    }
    
    uint8_t buffer[SHA256_BLOCK_LENGTH * 8] = {
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0
    };
    
    SHA256Transform_SIMD3(state, buffer);
    
    uint32_t state1[SHA256_STATE_LENGTH];
    uint32_t state2[SHA256_STATE_LENGTH];
    uint32_t state3[SHA256_STATE_LENGTH];
    uint32_t state4[SHA256_STATE_LENGTH];
    uint32_t state5[SHA256_STATE_LENGTH];
    uint32_t state6[SHA256_STATE_LENGTH];
    uint32_t state7[SHA256_STATE_LENGTH];
    uint32_t state8[SHA256_STATE_LENGTH];
    
    for (size_t i = 0; i < 8; ++i) {
        uint32_t* val = (uint32_t*)&(state[i]);
        
        state1[i] = val[0];
        state2[i] = val[1];
        state3[i] = val[2];
        state4[i] = val[3];
        state5[i] = val[4];
        state6[i] = val[5];
        state7[i] = val[6];
        state8[i] = val[7];
    }
    
    be32enc_vect(digest1, state1, SHA256_DIGEST_LENGTH);
    be32enc_vect(digest2, state2, SHA256_DIGEST_LENGTH);
    be32enc_vect(digest3, state3, SHA256_DIGEST_LENGTH);
    be32enc_vect(digest4, state4, SHA256_DIGEST_LENGTH);
    be32enc_vect(digest5, state5, SHA256_DIGEST_LENGTH);
    be32enc_vect(digest6, state6, SHA256_DIGEST_LENGTH);
    be32enc_vect(digest7, state7, SHA256_DIGEST_LENGTH);
    be32enc_vect(digest8, state8, SHA256_DIGEST_LENGTH);
}


// Esta funcion sirve para procesar 4 "strings" de 64 bytes cada uno (a la vez).
// Cada uno de estos "strings" podría llegar a ser la concatenación de 2 hashes de 32 bytes.
// O sea, puede ser util para procesar el merkle-root
void SHA256_SIMD_2_ONE_BUFFER(uint8_t* input_output) {
    
    __m128i state[SHA256_STATE_LENGTH];
    SHA256Init_State_SIMD(state);
    
    SHA256Transform_SIMD2(state, input_output);
    
    uint8_t buffer[SHA256_BLOCK_LENGTH * 4] = {
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0
    };
    
    SHA256Transform_SIMD2(state, buffer);
    
    uint32_t state1[SHA256_STATE_LENGTH];
    uint32_t state2[SHA256_STATE_LENGTH];
    uint32_t state3[SHA256_STATE_LENGTH];
    uint32_t state4[SHA256_STATE_LENGTH];
    
    for (size_t i = 0; i < 8; ++i) {
        uint32_t* val = (uint32_t*)&(state[i]);
        
        state1[i] = val[0];
        state2[i] = val[1];
        state3[i] = val[2];
        state4[i] = val[3];
    }
    
//    be32enc_vect(digest1, state1, SHA256_DIGEST_LENGTH);
//    be32enc_vect(digest2, state2, SHA256_DIGEST_LENGTH);
//    be32enc_vect(digest3, state3, SHA256_DIGEST_LENGTH);
//    be32enc_vect(digest4, state4, SHA256_DIGEST_LENGTH);

    be32enc_vect(input_output + 0,   state1, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 32,  state2, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 64,  state3, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 96,  state4, SHA256_DIGEST_LENGTH);
}

// Esta funcion sirve para procesar 8 "strings" de 64 bytes cada uno (a la vez).
// Cada uno de estos "strings" podría llegar a ser la concatenación de 2 hashes de 32 bytes.
// O sea, puede ser util para procesar el merkle-root
void SHA256_SIMD_3_ONE_BUFFER(uint8_t* input_output) {
    
    __m256i state[SHA256_STATE_LENGTH];
    SHA256Init_State_SIMD_x8(state);
    
    SHA256Transform_SIMD3(state, input_output);
    
    
    uint8_t buffer[SHA256_BLOCK_LENGTH * 8] = {
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0
    };
    
    SHA256Transform_SIMD3(state, buffer);
    
    
    //TODO: Fer: ver como mejorar esta parte!
    uint32_t state1[SHA256_STATE_LENGTH];
    uint32_t state2[SHA256_STATE_LENGTH];
    uint32_t state3[SHA256_STATE_LENGTH];
    uint32_t state4[SHA256_STATE_LENGTH];
    uint32_t state5[SHA256_STATE_LENGTH];
    uint32_t state6[SHA256_STATE_LENGTH];
    uint32_t state7[SHA256_STATE_LENGTH];
    uint32_t state8[SHA256_STATE_LENGTH];
    
    for (size_t i = 0; i < 8; ++i) {
        uint32_t* val = (uint32_t*)&(state[i]);
        
        state1[i] = val[0];
        state2[i] = val[1];
        state3[i] = val[2];
        state4[i] = val[3];
        state5[i] = val[4];
        state6[i] = val[5];
        state7[i] = val[6];
        state8[i] = val[7];
    }

    be32enc_vect(input_output + 0,   state1, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 32,  state2, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 64,  state3, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 96,  state4, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 128, state5, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 160, state6, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 192, state7, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 224, state8, SHA256_DIGEST_LENGTH);
}

// Esta funcion sirve para procesar 8 "strings" de 64 bytes cada uno (a la vez).
// Cada uno de estos "strings" podría llegar a ser la concatenación de 2 hashes de 32 bytes.
// O sea, puede ser util para procesar el merkle-root
void SHA256_SIMD_3_DOUBLE_ONE_BUFFER(uint8_t* input_output) {
    
    __m256i state[SHA256_STATE_LENGTH];
    SHA256Init_State_SIMD_x8(state);
    
    SHA256Transform_SIMD3(state, input_output);
    
    
    uint8_t buffer[SHA256_BLOCK_LENGTH * 8] = {
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0
    };
    
    SHA256Transform_SIMD3(state, buffer);
    
    
    //TODO: Fer: ver como mejorar esta parte!
    uint32_t state1[SHA256_STATE_LENGTH];
    uint32_t state2[SHA256_STATE_LENGTH];
    uint32_t state3[SHA256_STATE_LENGTH];
    uint32_t state4[SHA256_STATE_LENGTH];
    uint32_t state5[SHA256_STATE_LENGTH];
    uint32_t state6[SHA256_STATE_LENGTH];
    uint32_t state7[SHA256_STATE_LENGTH];
    uint32_t state8[SHA256_STATE_LENGTH];
    
    for (size_t i = 0; i < 8; ++i) {
        uint32_t* val = (uint32_t*)&(state[i]);
        
        state1[i] = val[0];
        state2[i] = val[1];
        state3[i] = val[2];
        state4[i] = val[3];
        state5[i] = val[4];
        state6[i] = val[5];
        state7[i] = val[6];
        state8[i] = val[7];
    }
    
    uint8_t buffer2[] = {
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0
    };
    
    
    std::memcpy(input_output, buffer2, 512);
    
    be32enc_vect(input_output + 0,   state1, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 64,  state2, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 128, state3, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 192, state4, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 256, state5, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 320, state6, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 384, state7, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 448, state8, SHA256_DIGEST_LENGTH);
    

    // Second Execution of Sha256
    SHA256Init_State_SIMD_x8(state);
    SHA256Transform_SIMD3(state, input_output);
    
    for (size_t i = 0; i < 8; ++i) {
        uint32_t* val = (uint32_t*)&(state[i]);
        
        state1[i] = val[0];
        state2[i] = val[1];
        state3[i] = val[2];
        state4[i] = val[3];
        state5[i] = val[4];
        state6[i] = val[5];
        state7[i] = val[6];
        state8[i] = val[7];
    }
    
    be32enc_vect(input_output + 0,   state1, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 32,  state2, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 64,  state3, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 96,  state4, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 128, state5, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 160, state6, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 192, state7, SHA256_DIGEST_LENGTH);
    be32enc_vect(input_output + 224, state8, SHA256_DIGEST_LENGTH);
}

// Esta funcion sirve para procesar 8 "strings" de 64 bytes cada uno (a la vez).
// Cada uno de estos "strings" podría llegar a ser la concatenación de 2 hashes de 32 bytes.
// O sea, puede ser util para procesar el merkle-root
void SHA256_SIMD_3_DOUBLE_TWO_BUFFERS(uint8_t const* input, uint8_t* output) {
    
    __m256i state[SHA256_STATE_LENGTH];
    SHA256Init_State_SIMD_x8(state);
    
    SHA256Transform_SIMD3(state, input);
    
    
	//uint8_t buffer[SHA256_BLOCK_LENGTH * 8] = {
    uint8_t buffer[] = {
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0
    };
    
    SHA256Transform_SIMD3(state, buffer);
    
    
    //TODO: Fer: ver como mejorar esta parte!
    uint32_t state1[SHA256_STATE_LENGTH];
    uint32_t state2[SHA256_STATE_LENGTH];
    uint32_t state3[SHA256_STATE_LENGTH];
    uint32_t state4[SHA256_STATE_LENGTH];
    uint32_t state5[SHA256_STATE_LENGTH];
    uint32_t state6[SHA256_STATE_LENGTH];
    uint32_t state7[SHA256_STATE_LENGTH];
    uint32_t state8[SHA256_STATE_LENGTH];
    
    for (size_t i = 0; i < 8; ++i) {
        uint32_t* val = (uint32_t*)&(state[i]);
        
        state1[i] = val[0];
        state2[i] = val[1];
        state3[i] = val[2];
        state4[i] = val[3];
        state5[i] = val[4];
        state6[i] = val[5];
        state7[i] = val[6];
        state8[i] = val[7];
    }
    
    uint8_t buffer2[] = {
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
        0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0
    };
    
    be32enc_vect(buffer2 + 0,   state1, SHA256_DIGEST_LENGTH);
    be32enc_vect(buffer2 + 64,  state2, SHA256_DIGEST_LENGTH);
    be32enc_vect(buffer2 + 128, state3, SHA256_DIGEST_LENGTH);
    be32enc_vect(buffer2 + 192, state4, SHA256_DIGEST_LENGTH);
    be32enc_vect(buffer2 + 256, state5, SHA256_DIGEST_LENGTH);
    be32enc_vect(buffer2 + 320, state6, SHA256_DIGEST_LENGTH);
    be32enc_vect(buffer2 + 384, state7, SHA256_DIGEST_LENGTH);
    be32enc_vect(buffer2 + 448, state8, SHA256_DIGEST_LENGTH);
    
    
    // Second Execution of Sha256
    SHA256Init_State_SIMD_x8(state);
    SHA256Transform_SIMD3(state, buffer2);
    
    for (size_t i = 0; i < 8; ++i) {
        uint32_t* val = (uint32_t*)&(state[i]);
        
        state1[i] = val[0];
        state2[i] = val[1];
        state3[i] = val[2];
        state4[i] = val[3];
        state5[i] = val[4];
        state6[i] = val[5];
        state7[i] = val[6];
        state8[i] = val[7];
    }
    
    be32enc_vect(output + 0,   state1, SHA256_DIGEST_LENGTH);
    be32enc_vect(output + 32,  state2, SHA256_DIGEST_LENGTH);
    be32enc_vect(output + 64,  state3, SHA256_DIGEST_LENGTH);
    be32enc_vect(output + 96,  state4, SHA256_DIGEST_LENGTH);
    be32enc_vect(output + 128, state5, SHA256_DIGEST_LENGTH);
    be32enc_vect(output + 160, state6, SHA256_DIGEST_LENGTH);
    be32enc_vect(output + 192, state7, SHA256_DIGEST_LENGTH);
    be32enc_vect(output + 224, state8, SHA256_DIGEST_LENGTH);
}





//-----------------------------------------------------------------------------------------------

// sha256_schedule_round ********************************************************
template <size_t Bits>
auto sha256_schedule_round(simd_type<Bits> schedule[64], int& i, simd_type<Bits> const* Ki, simd_type<Bits>& w1, simd_type<Bits>& w2, simd_type<Bits>& w3, simd_type<Bits>& w4) {
	//s0 = simd_sigma_0<Bits>(w1);
	//1 = simd_sigma_1<Bits>(w2);

	auto s0 = simd_sigma_0<Bits>(w1);
	auto s1 = simd_sigma_1<Bits>(w2);

	schedule[i] = simd_add(w3, Ki[i]);
	w3 = simd_add(simd_add(w3, w4), simd_add(s0, s1));
	i++;
}
 
template <size_t Bits>
void sha256_qms(simd_type<Bits> schedule[64], uint32_t message[64 * 2]) {
	using simd_t = simd_type<Bits>;
	static const auto k = sha256_k_constants<Bits>();

	//simd_t s0, s1;
	//simd_t const* Ki = (simd_t const*)k;
	simd_t const* Ki = k;

	auto W0 = simd_gather_be<Bits>(message);
	auto W1 = simd_gather_be<Bits>(&message[1]);
	auto W2 = simd_gather_be<Bits>(&message[2]);
	auto W3 = simd_gather_be<Bits>(&message[3]);
	auto W4 = simd_gather_be<Bits>(&message[4]);
	auto W5 = simd_gather_be<Bits>(&message[5]);
	auto W6 = simd_gather_be<Bits>(&message[6]);
	auto W7 = simd_gather_be<Bits>(&message[7]);
	auto W8 = simd_gather_be<Bits>(&message[8]);
	auto W9 = simd_gather_be<Bits>(&message[9]);
	auto W10 = simd_gather_be<Bits>(&message[10]);
	auto W11 = simd_gather_be<Bits>(&message[11]);
	auto W12 = simd_gather_be<Bits>(&message[12]);
	auto W13 = simd_gather_be<Bits>(&message[13]);
	auto W14 = simd_gather_be<Bits>(&message[14]);
	auto W15 = simd_gather_be<Bits>(&message[15]);

	int i;
	for (i = 0; i < 32; ) {
		sha256_schedule_round<Bits>(schedule, i, Ki, W1, W14, W0, W9);
		sha256_schedule_round<Bits>(schedule, i, Ki, W2, W15, W1, W10);
		sha256_schedule_round<Bits>(schedule, i, Ki, W3, W0, W2, W11);
		sha256_schedule_round<Bits>(schedule, i, Ki, W4, W1, W3, W12);
		sha256_schedule_round<Bits>(schedule, i, Ki, W5, W2, W4, W13);
		sha256_schedule_round<Bits>(schedule, i, Ki, W6, W3, W5, W14);
		sha256_schedule_round<Bits>(schedule, i, Ki, W7, W4, W6, W15);
		sha256_schedule_round<Bits>(schedule, i, Ki, W8, W5, W7, W0);
		sha256_schedule_round<Bits>(schedule, i, Ki, W9, W6, W8, W1);
		sha256_schedule_round<Bits>(schedule, i, Ki, W10, W7, W9, W2);
		sha256_schedule_round<Bits>(schedule, i, Ki, W11, W8, W10, W3);
		sha256_schedule_round<Bits>(schedule, i, Ki, W12, W9, W11, W4);
		sha256_schedule_round<Bits>(schedule, i, Ki, W13, W10, W12, W5);
		sha256_schedule_round<Bits>(schedule, i, Ki, W14, W11, W13, W6);
		sha256_schedule_round<Bits>(schedule, i, Ki, W15, W12, W14, W7);
		sha256_schedule_round<Bits>(schedule, i, Ki, W0, W13, W15, W8);
	}

	sha256_schedule_round<Bits>(schedule, i, Ki, W1, W14, W0, W9);
	schedule[48] = simd_add(W0, Ki[48]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W2, W15, W1, W10);
	schedule[49] = simd_add(W1, Ki[49]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W3, W0, W2, W11);
	schedule[50] = simd_add(W2, Ki[50]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W4, W1, W3, W12);
	schedule[51] = simd_add(W3, Ki[51]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W5, W2, W4, W13);
	schedule[52] = simd_add(W4, Ki[52]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W6, W3, W5, W14);
	schedule[53] = simd_add(W5, Ki[53]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W7, W4, W6, W15);
	schedule[54] = simd_add(W6, Ki[54]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W8, W5, W7, W0);
	schedule[55] = simd_add(W7, Ki[55]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W9, W6, W8, W1);
	schedule[56] = simd_add(W8, Ki[56]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W10, W7, W9, W2);
	schedule[57] = simd_add(W9, Ki[57]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W11, W8, W10, W3);
	schedule[58] = simd_add(W10, Ki[58]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W12, W9, W11, W4);
	schedule[59] = simd_add(W11, Ki[59]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W13, W10, W12, W5);
	schedule[60] = simd_add(W12, Ki[60]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W14, W11, W13, W6);
	schedule[61] = simd_add(W13, Ki[61]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W15, W12, W14, W7);
	schedule[62] = simd_add(W14, Ki[62]);
	sha256_schedule_round<Bits>(schedule, i, Ki, W0, W13, W15, W8);
	schedule[63] = simd_add(W15, Ki[63]);
}




//void SHA256_SIMD3_Step(__m256i state[SHA256_STATE_LENGTH], __m256i schedule[64]) {
//
//	__m256i S[8];
//	memcpy(S, state, 32 * 8);
//
//	RNDr_Simd8(S, 0, schedule);
//	RNDr_Simd8(S, 1, schedule);
//	RNDr_Simd8(S, 2, schedule);
//	RNDr_Simd8(S, 3, schedule);
//	RNDr_Simd8(S, 4, schedule);
//	RNDr_Simd8(S, 5, schedule);
//	RNDr_Simd8(S, 6, schedule);
//	RNDr_Simd8(S, 7, schedule);
//	RNDr_Simd8(S, 8, schedule);
//	RNDr_Simd8(S, 9, schedule);
//	RNDr_Simd8(S, 10, schedule);
//	RNDr_Simd8(S, 11, schedule);
//	RNDr_Simd8(S, 12, schedule);
//	RNDr_Simd8(S, 13, schedule);
//	RNDr_Simd8(S, 14, schedule);
//	RNDr_Simd8(S, 15, schedule);
//
//	RNDr_Simd8(S, 16, schedule);
//	RNDr_Simd8(S, 17, schedule);
//	RNDr_Simd8(S, 18, schedule);
//	RNDr_Simd8(S, 19, schedule);
//	RNDr_Simd8(S, 20, schedule);
//	RNDr_Simd8(S, 21, schedule);
//	RNDr_Simd8(S, 22, schedule);
//	RNDr_Simd8(S, 23, schedule);
//	RNDr_Simd8(S, 24, schedule);
//	RNDr_Simd8(S, 25, schedule);
//	RNDr_Simd8(S, 26, schedule);
//	RNDr_Simd8(S, 27, schedule);
//	RNDr_Simd8(S, 28, schedule);
//	RNDr_Simd8(S, 29, schedule);
//	RNDr_Simd8(S, 30, schedule);
//	RNDr_Simd8(S, 31, schedule);
//
//	RNDr_Simd8(S, 32, schedule);
//	RNDr_Simd8(S, 33, schedule);
//	RNDr_Simd8(S, 34, schedule);
//	RNDr_Simd8(S, 35, schedule);
//	RNDr_Simd8(S, 36, schedule);
//	RNDr_Simd8(S, 37, schedule);
//	RNDr_Simd8(S, 38, schedule);
//	RNDr_Simd8(S, 39, schedule);
//	RNDr_Simd8(S, 40, schedule);
//	RNDr_Simd8(S, 41, schedule);
//	RNDr_Simd8(S, 42, schedule);
//	RNDr_Simd8(S, 43, schedule);
//	RNDr_Simd8(S, 44, schedule);
//	RNDr_Simd8(S, 45, schedule);
//	RNDr_Simd8(S, 46, schedule);
//	RNDr_Simd8(S, 47, schedule);
//
//	RNDr_Simd8(S, 48, schedule);
//	RNDr_Simd8(S, 49, schedule);
//	RNDr_Simd8(S, 50, schedule);
//	RNDr_Simd8(S, 51, schedule);
//	RNDr_Simd8(S, 52, schedule);
//	RNDr_Simd8(S, 53, schedule);
//	RNDr_Simd8(S, 54, schedule);
//	RNDr_Simd8(S, 55, schedule);
//	RNDr_Simd8(S, 56, schedule);
//	RNDr_Simd8(S, 57, schedule);
//	RNDr_Simd8(S, 58, schedule);
//	RNDr_Simd8(S, 59, schedule);
//	RNDr_Simd8(S, 60, schedule);
//	RNDr_Simd8(S, 61, schedule);
//	RNDr_Simd8(S, 62, schedule);
//	RNDr_Simd8(S, 63, schedule);
//
//	for (size_t i = 0; i < 8; ++i) {
//		//state[i] += S[i];
//		state[i] = _mm256_add_epi32(state[i], S[i]);
//	}
//}



// sha256_ch ***************************************************************
//template <size_t Bits>
//inline
//auto sha256_ch(simd_type<Bits> x, simd_type<Bits> y, simd_type<Bits> z) {
//	return simd_xor(simd_and(x, simd_xor(y, z)), z);  //((x & (y ^ z)) ^ z)
//}

template <size_t Bits>
inline
auto sha256_ch(typename simd_type_impl<Bits>::type x, typename simd_type_impl<Bits>::type y, typename simd_type_impl<Bits>::type z) {
    return simd_xor(simd_and(x, simd_xor(y, z)), z);  //((x & (y ^ z)) ^ z)
}





// sha256_maj ***************************************************************
template <size_t Bits>
inline
auto sha256_maj(simd_type<Bits> x, simd_type<Bits> y, simd_type<Bits> z) {
	return simd_or(simd_and(x, simd_or(y, z)), simd_and(y, z)); //((x & (y | z)) | (y & z))

}

// sha256_shr ***************************************************************
template <size_t Bits>
inline
auto sha256_shr(simd_type<Bits> x, int n) {
	return simd_rshift(x, n); //(x >> n)
}

// sha256_rotr ***************************************************************
template <size_t Bits>
inline
auto sha256_rotr(simd_type<Bits> x, int n) {
	return simd_or(simd_rshift(x, n), simd_lshift(x, (32 - n))); //((x >> n) | (x << (32 - n)))
}

// sha256_S0 ***************************************************************
template <size_t Bits>
inline
auto sha256_S0(simd_type<Bits> x) {
	return simd_xor(sha256_rotr(x, 2), simd_xor(sha256_rotr(x, 13), sha256_rotr(x, 22)));
	//(ROTR_Simd(x,  2) ^ ROTR_Simd(x, 13) ^ ROTR_Simd(x, 22))
}


// sha256_S1 ***************************************************************
template <size_t Bits>
inline
auto sha256_S1(simd_type<Bits> x) {
	return simd_xor(sha256_rotr<Bits>(x, 6), simd_xor(sha256_rotr<Bits>(x, 11), sha256_rotr<Bits>(x, 25)));
	//(ROTR_Simd(x,  6) ^ ROTR_Simd(x, 11) ^ ROTR_Simd(x, 25))
}

// sha256_s0 ***************************************************************
template <size_t Bits>
inline
auto sha256_s0(simd_type<Bits> x) {
	return simd_xor(sha256_rotr(x, 7), simd_xor(sha256_rotr(x, 18), sha256_shr(x, 3)));
	//(ROTR(x,  7) ^ ROTR(x, 18) ^  SHR(x,  3))
}

// sha256_s1 ***************************************************************
template <size_t Bits>
inline
auto sha256_s1(simd_type<Bits> x) {
	return simd_xor(sha256_rotr(x, 17), simd_xor(sha256_rotr(x, 19), sha256_shr(x, 10)));
	//(ROTR(x, 17) ^ ROTR(x, 19) ^  SHR(x, 10))
}



template <size_t Bits>
inline
void sha256_rnd(simd_type<Bits>& a, simd_type<Bits>& b, simd_type<Bits>& c, simd_type<Bits>& d, 
			    simd_type<Bits>& e, simd_type<Bits>& f, simd_type<Bits>& g, simd_type<Bits>& h, 
	            simd_type<Bits> k) {
	auto t0 = simd_add(simd_add(simd_add(h, sha256_S1<Bits>(e)), sha256_ch(e, f, g)), k);
	auto t1 = simd_add(sha256_S0<Bits>(a), sha256_maj(a, b, c));
	d = simd_add(d, t0);
	h = simd_add(t0, t1);
}

template <size_t Bits>
inline
void sha256_rndr(simd_type<Bits> S[SHA256_STATE_LENGTH], size_t i, simd_type<Bits> sch[64]) {

	sha256_rnd<Bits>(S[(64 - i) % 8],
		S[(65 - i) % 8],
		S[(66 - i) % 8],
		S[(67 - i) % 8],
		S[(68 - i) % 8],
		S[(69 - i) % 8],
		S[(70 - i) % 8],
		S[(71 - i) % 8],
		sch[i]);

	//RND_Simd8(S[(64 - i) % 8],
	//	S[(65 - i) % 8],
	//	S[(66 - i) % 8],
	//	S[(67 - i) % 8],
	//	S[(68 - i) % 8],
	//	S[(69 - i) % 8],
	//	S[(70 - i) % 8],
	//	S[(71 - i) % 8],
	//	sch[i]);
}

template <size_t Bits>
void sha256_step(simd_type<Bits> state[SHA256_STATE_LENGTH], simd_type<Bits> schedule[64]) {
	simd_type<Bits> S[8];
	//memcpy(S, state, 32 * 8); //TODO:
	memcpy(S, state, Bits);

	sha256_rndr<Bits>(S, 0, schedule);
	sha256_rndr<Bits>(S, 1, schedule);
	sha256_rndr<Bits>(S, 2, schedule);
	sha256_rndr<Bits>(S, 3, schedule);
	sha256_rndr<Bits>(S, 4, schedule);
	sha256_rndr<Bits>(S, 5, schedule);
	sha256_rndr<Bits>(S, 6, schedule);
	sha256_rndr<Bits>(S, 7, schedule);
	sha256_rndr<Bits>(S, 8, schedule);
	sha256_rndr<Bits>(S, 9, schedule);
	sha256_rndr<Bits>(S, 10, schedule);
	sha256_rndr<Bits>(S, 11, schedule);
	sha256_rndr<Bits>(S, 12, schedule);
	sha256_rndr<Bits>(S, 13, schedule);
	sha256_rndr<Bits>(S, 14, schedule);
	sha256_rndr<Bits>(S, 15, schedule);

	sha256_rndr<Bits>(S, 16, schedule);
	sha256_rndr<Bits>(S, 17, schedule);
	sha256_rndr<Bits>(S, 18, schedule);
	sha256_rndr<Bits>(S, 19, schedule);
	sha256_rndr<Bits>(S, 20, schedule);
	sha256_rndr<Bits>(S, 21, schedule);
	sha256_rndr<Bits>(S, 22, schedule);
	sha256_rndr<Bits>(S, 23, schedule);
	sha256_rndr<Bits>(S, 24, schedule);
	sha256_rndr<Bits>(S, 25, schedule);
	sha256_rndr<Bits>(S, 26, schedule);
	sha256_rndr<Bits>(S, 27, schedule);
	sha256_rndr<Bits>(S, 28, schedule);
	sha256_rndr<Bits>(S, 29, schedule);
	sha256_rndr<Bits>(S, 30, schedule);
	sha256_rndr<Bits>(S, 31, schedule);

	sha256_rndr<Bits>(S, 32, schedule);
	sha256_rndr<Bits>(S, 33, schedule);
	sha256_rndr<Bits>(S, 34, schedule);
	sha256_rndr<Bits>(S, 35, schedule);
	sha256_rndr<Bits>(S, 36, schedule);
	sha256_rndr<Bits>(S, 37, schedule);
	sha256_rndr<Bits>(S, 38, schedule);
	sha256_rndr<Bits>(S, 39, schedule);
	sha256_rndr<Bits>(S, 40, schedule);
	sha256_rndr<Bits>(S, 41, schedule);
	sha256_rndr<Bits>(S, 42, schedule);
	sha256_rndr<Bits>(S, 43, schedule);
	sha256_rndr<Bits>(S, 44, schedule);
	sha256_rndr<Bits>(S, 45, schedule);
	sha256_rndr<Bits>(S, 46, schedule);
	sha256_rndr<Bits>(S, 47, schedule);

	sha256_rndr<Bits>(S, 48, schedule);
	sha256_rndr<Bits>(S, 49, schedule);
	sha256_rndr<Bits>(S, 50, schedule);
	sha256_rndr<Bits>(S, 51, schedule);
	sha256_rndr<Bits>(S, 52, schedule);
	sha256_rndr<Bits>(S, 53, schedule);
	sha256_rndr<Bits>(S, 54, schedule);
	sha256_rndr<Bits>(S, 55, schedule);
	sha256_rndr<Bits>(S, 56, schedule);
	sha256_rndr<Bits>(S, 57, schedule);
	sha256_rndr<Bits>(S, 58, schedule);
	sha256_rndr<Bits>(S, 59, schedule);
	sha256_rndr<Bits>(S, 60, schedule);
	sha256_rndr<Bits>(S, 61, schedule);
	sha256_rndr<Bits>(S, 62, schedule);
	sha256_rndr<Bits>(S, 63, schedule);

	for (size_t i = 0; i < 8; ++i) {
		//state[i] += S[i];
		state[i] = simd_add(state[i], S[i]);
	}
}

template <size_t Bits>
void sha256_transform(simd_type<Bits> state[SHA256_STATE_LENGTH], 
	uint8_t const input[Bits * 2])  //256 or 512
{
	simd_type<Bits> schedule[64] = {};
	sha256_qms<Bits>(schedule, (uint32_t*)input);
	sha256_step<Bits>(state, schedule);
}

//template <size_t Bits>
//void sha256_transform_2(std::array<simd_type<Bits>, SHA256_STATE_LENGTH> state,
//                      uint8_t const input[Bits * 2])  //256 or 512
//{
//    simd_type<Bits> schedule[64] = {};
//    sha256_qms<Bits>(schedule, (uint32_t*)input);
//    sha256_step<Bits>(state, schedule);
//}


template <size_t Bits>
void sha256_xxxxx(uint8_t const* input, uint8_t* output) {
	auto state = sha256_init_state<Bits>();
    
    sha256_transform<Bits>(state.data(), input);
//	sha256_transform_2(state, input);
    
    
	
	//uint8_t buffer[SHA256_BLOCK_LENGTH * 8] = {
	uint8_t buffer[] = {
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0
	};

	//sha256_transform(state, buffer);
	sha256_transform<Bits>(state.data(), buffer);


	//TODO: Fer: ver como mejorar esta parte!
	uint32_t state1[SHA256_STATE_LENGTH];
	uint32_t state2[SHA256_STATE_LENGTH];
	uint32_t state3[SHA256_STATE_LENGTH];
	uint32_t state4[SHA256_STATE_LENGTH];
	uint32_t state5[SHA256_STATE_LENGTH];
	uint32_t state6[SHA256_STATE_LENGTH];
	uint32_t state7[SHA256_STATE_LENGTH];
	uint32_t state8[SHA256_STATE_LENGTH];

	for (size_t i = 0; i < 8; ++i) {
		uint32_t* val = (uint32_t*)&(state[i]);

		state1[i] = val[0];
		state2[i] = val[1];
		state3[i] = val[2];
		state4[i] = val[3];
		state5[i] = val[4];
		state6[i] = val[5];
		state7[i] = val[6];
		state8[i] = val[7];
	}

	uint8_t buffer2[] = {
		0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
		0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
		0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
		0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
		0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
		0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
		0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0,
		0   , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
		0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0
	};

	be32enc_vect(buffer2 + 0, state1, SHA256_DIGEST_LENGTH);
	be32enc_vect(buffer2 + 64, state2, SHA256_DIGEST_LENGTH);
	be32enc_vect(buffer2 + 128, state3, SHA256_DIGEST_LENGTH);
	be32enc_vect(buffer2 + 192, state4, SHA256_DIGEST_LENGTH);
	be32enc_vect(buffer2 + 256, state5, SHA256_DIGEST_LENGTH);
	be32enc_vect(buffer2 + 320, state6, SHA256_DIGEST_LENGTH);
	be32enc_vect(buffer2 + 384, state7, SHA256_DIGEST_LENGTH);
	be32enc_vect(buffer2 + 448, state8, SHA256_DIGEST_LENGTH);


	// Second Execution of Sha256
	//SHA256Init_State_SIMD_x8(state);
	state = sha256_init_state<Bits>();

	//sha256_transform(state, buffer2);
	sha256_transform<Bits>(state.data(), buffer2);

	for (size_t i = 0; i < 8; ++i) {
		uint32_t* val = (uint32_t*)&(state[i]);

		state1[i] = val[0];
		state2[i] = val[1];
		state3[i] = val[2];
		state4[i] = val[3];
		state5[i] = val[4];
		state6[i] = val[5];
		state7[i] = val[6];
		state8[i] = val[7];
	}

	be32enc_vect(output + 0, state1, SHA256_DIGEST_LENGTH);
	be32enc_vect(output + 32, state2, SHA256_DIGEST_LENGTH);
	be32enc_vect(output + 64, state3, SHA256_DIGEST_LENGTH);
	be32enc_vect(output + 96, state4, SHA256_DIGEST_LENGTH);
	be32enc_vect(output + 128, state5, SHA256_DIGEST_LENGTH);
	be32enc_vect(output + 160, state6, SHA256_DIGEST_LENGTH);
	be32enc_vect(output + 192, state7, SHA256_DIGEST_LENGTH);
	be32enc_vect(output + 224, state8, SHA256_DIGEST_LENGTH);
}



//--------------------------------------------------------------------------------


#include <array>

#define Container typename

template <size_t Size>
using byte_array = std::array<uint8_t, Size>;

constexpr size_t hash_size = 32;
constexpr size_t hash_double_size = hash_size * 2;

using hash_t = byte_array<hash_size>;
using hash_double_t = byte_array<hash_double_size>;

template <Container C>
inline
hash_t sha256_serial(C const& c) {
    hash_t hash;
	SHA256_(reinterpret_cast<uint8_t const*>(c.data()), c.size(), hash.data());
    return hash;
}

template <Container C>
inline
hash_t sha256_double_serial(C const& c) {
    return sha256_serial(sha256_serial(c));
}


template <Container C>
inline
hash_t sha256_simd(C const& c) {
	hash_t hash;
	SHA256_SIMD_1(reinterpret_cast<uint8_t const*>(c.data()), c.size(), hash.data());
	return hash;
}

template <Container C1, Container C2, Container C3, Container C4>
inline
std::tuple<hash_t, hash_t, hash_t, hash_t>
sha256_simd_multiple(C1 const& c1, C2 const& c2, C3 const& c3, C4 const& c4) {
	//precondition: size(c1) == size(c2) == size(c3) == size(c4)
	hash_t hash1;
	hash_t hash2;
	hash_t hash3;
	hash_t hash4;

	SHA256_SIMD_2(
		reinterpret_cast<uint8_t const*>(c1.data()), 
		reinterpret_cast<uint8_t const*>(c2.data()),
		reinterpret_cast<uint8_t const*>(c3.data()),
		reinterpret_cast<uint8_t const*>(c4.data()),
		c1.size(),
		hash1.data(),
		hash2.data(),
		hash3.data(),
		hash4.data()
	);

	return std::make_tuple(hash1, hash2, hash3, hash4);
}



template <Container C1, Container C2, Container C3, Container C4, Container C5, Container C6, Container C7, Container C8>
inline
std::tuple<hash_t, hash_t, hash_t, hash_t, hash_t, hash_t, hash_t, hash_t>
sha256_simd_multiple(C1 const& c1, C2 const& c2, C3 const& c3, C4 const& c4,
                     C5 const& c5, C6 const& c6, C7 const& c7, C8 const& c8) {
    //precondition: size(c1) == size(c2) == size(c3) == size(c4) == size(c5) == size(c6) == size(c7) == size(c8)
    hash_t hash1;
    hash_t hash2;
    hash_t hash3;
    hash_t hash4;
    hash_t hash5;
    hash_t hash6;
    hash_t hash7;
    hash_t hash8;
    
    SHA256_SIMD_3(
                  reinterpret_cast<uint8_t const*>(c1.data()),
                  reinterpret_cast<uint8_t const*>(c2.data()),
                  reinterpret_cast<uint8_t const*>(c3.data()),
                  reinterpret_cast<uint8_t const*>(c4.data()),
                  reinterpret_cast<uint8_t const*>(c5.data()),
                  reinterpret_cast<uint8_t const*>(c6.data()),
                  reinterpret_cast<uint8_t const*>(c7.data()),
                  reinterpret_cast<uint8_t const*>(c8.data()),
                  c1.size(),
                  hash1.data(),
                  hash2.data(),
                  hash3.data(),
                  hash4.data(),
                  hash5.data(),
                  hash6.data(),
                  hash7.data(),
                  hash8.data()
                  );
    
    return std::make_tuple(hash1, hash2, hash3, hash4, hash5, hash6, hash7, hash8);
}



template <typename HashType>
void print_hash(HashType const& hash) {
    for (auto x : hash) {
        printf("%02x", x);
    }
	printf("\n");
}


inline
hash_double_t concat_hash(hash_t const& a, hash_t const& b) {
    hash_double_t res;
    
    std::memcpy(res.data(),            a.data(), a.size() * sizeof(hash_t::value_type));
    std::memcpy(res.data() + a.size(), b.data(), b.size() * sizeof(hash_t::value_type));
    
    return res;
}


void test_sha256_simd() {
    {
        std::string data = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijkl"
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKL"
                           "1234567890123456789012345678901234567890123456789012345678901234"
                           "!@#$%^&*()_+=-{}?;!@#$%^&*()_+=-{}?;!@#$%^&*()_+=-{}?;!@#$%^&*()";
        
        auto y = sha256_serial(data);
        auto x = sha256_simd(data);
        
        if (x != y) std::cout << "test_sha256_simd error, data: " << data << std::endl;
    }

    
    {
        std::string data = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijkl";
        auto y = sha256_serial(data);
        auto x = sha256_simd(data);
        
        if (x != y) std::cout << "test_sha256_simd error, data: " << data << std::endl;
    }
    
    
    {
        std::string data(256, 'a');
        auto y = sha256_serial(data);
        auto x = sha256_simd(data);

        if (x != y) std::cout << "test_sha256_simd error, data: " << data << std::endl;
    }

    {
        std::string data = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
        auto y = sha256_serial(data);
        auto x = sha256_simd(data);
        
        if (x != y) std::cout << "test_sha256_simd error, data: " << data << std::endl;
    }
    
    {
        std::string data = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
        auto y = sha256_serial(data);
        auto x = sha256_simd(data);
        
        if (x != y) std::cout << "test_sha256_simd error, data: " << data << std::endl;
    }

    {
        std::string data = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeef";
        
        auto y = sha256_serial(data);
        auto x = sha256_simd(data);
        
        if (x != y) std::cout << "test_sha256_simd error, data: " << data << std::endl;
    }

    {
        std::string data = "a";
        auto y = sha256_serial(data);
        auto x = sha256_simd(data);
        
        if (x != y) std::cout << "test_sha256_simd error, data: " << data << std::endl;
    }

    {
        std::string data = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        auto y = sha256_serial(data);
        auto x = sha256_simd(data);
        
        if (x != y) std::cout << "test_sha256_simd error, data: " << data << std::endl;
    }


    
//    print_hash(y);
    
}


#include <chrono>
#include <random>
#include <vector>
#define Integer typename

using namespace std;

template <Integer I = unsigned int, I From = 0, I To = std::numeric_limits<I>::max()>
struct random_int_generator {
	using dis_t = uniform_int_distribution<I>;
	static constexpr I from = From;
	static constexpr I to = To;

	random_int_generator()
		// : mt {rd()}
		: dis{ from, to }  // closed range [1, 1000]
	{}

	random_int_generator(random_int_generator const&) = default;

	auto operator()() {
		return dis(eng);
	}

	// random_device rd;
	// mt19937 eng;
	std::mt19937 eng{ std::random_device{}() };
	// std::mt19937 eng{std::chrono::system_clock::now().time_since_epoch().count()};
	// std::mt19937 eng(std::chrono::system_clock::now().time_since_epoch().count());

	dis_t dis;
}; // Models: RandomIntGenerator

using transaction_t = array<uint8_t, 256>;

inline
uint8_t const* as_bytes(uint32_t const& x) {
	return static_cast<uint8_t const*>(static_cast<void const*>(&x));
}

inline
uint8_t const* as_bytes(uint32_t const* x) {
	return static_cast<uint8_t const*>(static_cast<void const*>(x));
}


transaction_t random_transaction_t(random_int_generator<>& gen) {
	uint32_t temp[] = { gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(),
		gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(),
		gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(),
		gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen() };

	transaction_t res;
	//std::copy(as_bytes(temp), as_bytes(temp) + sizeof(temp), res.data());
	//std::copy_n(as_bytes(temp), sizeof(temp), res.data());
	std::memcpy(res.data(), as_bytes(temp), sizeof(temp));
	return res;
}

std::vector<transaction_t> make_transactions(random_int_generator<>& gen, size_t n) {
	// random_int_generator<> gen;
	std::vector<transaction_t> res(n);
	// res.reserve(n);
	std::generate(std::begin(res), std::end(res), [&gen] { return random_transaction_t(gen); });
	return res;
}


void test_merkle() {
//    constexpr size_t tx_count = 2048;
	constexpr size_t tx_count = 64;
	    
	random_int_generator<> gen;
	auto data = make_transactions(gen, tx_count);
	vector<hash_t> hashes;
	hashes.reserve(tx_count);
	    
	for (auto& tx : data) {
	    auto hash = sha256_double_serial(tx);
	    hashes.push_back(hash);
	}
	    
	for (size_t i = 0; i < 32; ++i) {
	    auto hash = sha256_double_serial(concat_hash(hashes[i * 2], hashes[i * 2 + 1]));
	    hashes[i] = hash;
	}
	for (size_t i = 0; i < 16; ++i) {
	    auto hash = sha256_double_serial(concat_hash(hashes[i * 2], hashes[i * 2 + 1]));
	    hashes[i] = hash;
	}
	for (size_t i = 0; i < 8; ++i) {
	    auto hash = sha256_double_serial(concat_hash(hashes[i * 2], hashes[i * 2 + 1]));
	    hashes[i] = hash;
	}
	for (size_t i = 0; i < 4; ++i) {
	    auto hash = sha256_double_serial(concat_hash(hashes[i * 2], hashes[i * 2 + 1]));
	    hashes[i] = hash;
	}
	for (size_t i = 0; i < 2; ++i) {
	    auto hash = sha256_double_serial(concat_hash(hashes[i * 2], hashes[i * 2 + 1]));
	    hashes[i] = hash;
	}
	auto hash = sha256_double_serial(concat_hash(hashes[0], hashes[1]));
	hashes[0] = hash;
	print_bytes(hashes[0].data(),  32);
	
	    
	std::cout << "------------------------------------------------------\n";
	    
	// ---------------------------------------
	    
	uint8_t buffer[tx_count  * 32];
	uint8_t* buffer_it = buffer;
	size_t len = 0;
	    
	for (auto& tx : data) {
	    auto hash = sha256_double_serial(tx);
	    std::memcpy(buffer_it, hash.data(), 32);
	    buffer_it += 32;
	    len += 32;
	}
	
	while (len >= 512) {
		buffer_it = buffer;
		uint8_t* buffer_out = buffer;
		size_t lentmp = len;
		while (lentmp >= 512) {
			//SHA256_SIMD_3_DOUBLE_TWO_BUFFERS(buffer_it, buffer_out);
			sha256_xxxxx<256>(buffer_it, buffer_out);
			buffer_it += 512;
			buffer_out += 256;
			lentmp -= 512;
			len -= 256;
		}
	}
	
	if (len == 256) {
		SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer); // 8 -> 4
		SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer); // 4 -> 2
		SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer); // 2 -> 1
	}
	print_bytes(buffer, 32);
}


void tests() {
	test_sha256_simd();
	test_merkle();
}







int main() {

	/*constexpr*/ 
	auto xxx = sha256_init_state<256>();
	auto yyy = sha256_init_state<128>();
    
    tests();



//
//
//	//----------------------------------------------------------
////
//////	std::string d1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
//////	std::string d2 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
//////	std::string d3 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
//////	std::string d4 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
////
////    std::string d1 = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijkl";
////    std::string d2 = "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKL";
////    std::string d3 = "1234567890123456789012345678901234567890123456789012345678901234";
////    std::string d4 = "!@#$%^&*()_+=-{}?;!@#$%^&*()_+=-{}?;!@#$%^&*()_+=-{}?;!@#$%^&*()";
////    
////    
////	print_hash(sha256_serial(d1));
////	print_hash(sha256_serial(d2));
////	print_hash(sha256_serial(d3));
////	print_hash(sha256_serial(d4));
////
////	hash_t x, y, z, w;
////	std::tie(x, y, z, w) = sha256_simd_multiple(d1, d2, d3, d4);
////
////	print_hash(x);
////	print_hash(y);
////	print_hash(z);
////	print_hash(w);
//
//    //----------------------------------------------------------
//////    
//////    std::string d1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
//////    std::string d2 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
//////    std::string d3 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
//////    std::string d4 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
//////    std::string d5 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
//////    std::string d6 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
//////    std::string d7 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
//////    std::string d8 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
////    
////    std::string d1 = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijkl";
////    std::string d2 = "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKL";
////    std::string d3 = "1234567890123456789012345678901234567890123456789012345678901234";
////    std::string d4 = "!@#$%^&*()_+=-{}?;!@#$%^&*()_+=-{}?;!@#$%^&*()_+=-{}?;!@#$%^&*()";
////    std::string d5 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
////    std::string d6 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
////    std::string d7 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
////    std::string d8 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
////
////    
//////    std::string d1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
//////    std::string d2 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
//////    std::string d3 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
//////    std::string d4 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
//////    std::string d5 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
//////    std::string d6 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
//////    std::string d7 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
//////    std::string d8 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
////    
////    
////    print_hash(sha256_serial(d1));
////    print_hash(sha256_serial(d2));
////    print_hash(sha256_serial(d3));
////    print_hash(sha256_serial(d4));
////    print_hash(sha256_serial(d5));
////    print_hash(sha256_serial(d6));
////    print_hash(sha256_serial(d7));
////    print_hash(sha256_serial(d8));
////    
////    hash_t x, y, z, w, a, b, c, d;
////
////    std::tie(x, y, z, w) = sha256_simd_multiple(d1, d2, d3, d4);
////    std::tie(a, b, c, d) = sha256_simd_multiple(d5, d6, d7, d8);
////    print_hash(x);
////    print_hash(y);
////    print_hash(z);
////    print_hash(w);
////    print_hash(a);
////    print_hash(b);
////    print_hash(c);
////    print_hash(d);
////
////    
////    std::tie(x, y, z, w, a, b, c, d) = sha256_simd_multiple(d1, d2, d3, d4, d5, d6, d7, d8);
////    print_hash(x);
////    print_hash(y);
////    print_hash(z);
////    print_hash(w);
////    print_hash(a);
////    print_hash(b);
////    print_hash(c);
////    print_hash(d);
////    
////    
//    //-------------------------------------------------------
////
////    std::string d1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
////    std::string d2 = "ccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddd";
////    std::string d3 = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeffffffffffffffffffffffffffffffff";
////    std::string d4 = "gggggggggggggggggggggggggggggggghhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh";
////
////    
////    auto hd1 = sha256_serial(d1);
////    auto hd2 = sha256_serial(d2);
////    auto hd3 = sha256_serial(d3);
////    auto hd4 = sha256_serial(d4);
////    
//////    print_hash(hd1);
//////    print_hash(hd2);
//////    print_hash(hd3);
//////    print_hash(hd4);
////    
////    
////    auto hd1d2 = sha256_serial(concat_hash(hd1, hd2));
////    auto hd3d4 = sha256_serial(concat_hash(hd3, hd4));
////
//////    print_hash(hd1d2);
//////    print_hash(hd3d4);
////
////    auto hd1d2d3d4 = sha256_serial(concat_hash(hd1d2, hd3d4));
////    
////    print_hash(hd1d2d3d4);
////
////    
////    std::cout << "------------------------------------------------------\n";
////    
////    
////    std::string temp = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
////                       "ccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddd"
////                       "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeffffffffffffffffffffffffffffffff"
////                       "gggggggggggggggggggggggggggggggghhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh";
////    
////    std::array<uint8_t, 256> da;
////    std::memcpy(da.data(), temp.data(), 256);
////    
//////    print_bytes(da.data(), 256);
////    
////    SHA256_SIMD_2(
////                  reinterpret_cast<uint8_t const*>(da.data() + 0),
////                  reinterpret_cast<uint8_t const*>(da.data() + 64),
////                  reinterpret_cast<uint8_t const*>(da.data() + 128),
////                  reinterpret_cast<uint8_t const*>(da.data() + 192),
////                  da.size() / 4,
////                  reinterpret_cast<uint8_t*>(da.data() + 0),
////                  reinterpret_cast<uint8_t*>(da.data() + 32),
////                  reinterpret_cast<uint8_t*>(da.data() + 64),
////                  reinterpret_cast<uint8_t*>(da.data() + 96)
////                  );
////
////    print_bytes(da.data() + 0, 32);
////    print_bytes(da.data() + 32, 32);
////    print_bytes(da.data() + 64, 32);
////    print_bytes(da.data() + 96, 32);
////
////    std::cout << "------------------------------------------------------\n";
////
////    
////    SHA256_SIMD_2(
////                  reinterpret_cast<uint8_t const*>(da.data() + 0),
////                  reinterpret_cast<uint8_t const*>(da.data() + 64),
////                  reinterpret_cast<uint8_t const*>(da.data() + 128),
////                  reinterpret_cast<uint8_t const*>(da.data() + 192),
////                  da.size() / 4,
////                  reinterpret_cast<uint8_t*>(da.data() + 0),
////                  reinterpret_cast<uint8_t*>(da.data() + 32),
////                  reinterpret_cast<uint8_t*>(da.data() + 64),
////                  reinterpret_cast<uint8_t*>(da.data() + 96)
////                  );
////    
////    print_bytes(da.data() + 0, 32);
////    print_bytes(da.data() + 32, 32);
//////    print_bytes(da.data() + 64, 32);
//////    print_bytes(da.data() + 96, 32);
////
////    std::cout << "------------------------------------------------------\n";
////
////    
////    
////    SHA256_SIMD_2(
////                  reinterpret_cast<uint8_t const*>(da.data() + 0),
////                  reinterpret_cast<uint8_t const*>(da.data() + 64),
////                  reinterpret_cast<uint8_t const*>(da.data() + 128),
////                  reinterpret_cast<uint8_t const*>(da.data() + 192),
////                  da.size() / 4,
////                  reinterpret_cast<uint8_t*>(da.data() + 0),
////                  reinterpret_cast<uint8_t*>(da.data() + 32),
////                  reinterpret_cast<uint8_t*>(da.data() + 64),
////                  reinterpret_cast<uint8_t*>(da.data() + 96)
////                  );
////    
////    print_bytes(da.data() + 0, 32);
//////    print_bytes(da.data() + 32, 32);
//////    print_bytes(da.data() + 64, 32);
//////    print_bytes(da.data() + 96, 32);
////    
////    std::cout << "------------------------------------------------------\n";
////    
////
////    
////    std::string d1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
////    std::string d2 = "ccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddd";
////    std::string d3 = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeffffffffffffffffffffffffffffffff";
////    std::string d4 = "gggggggggggggggggggggggggggggggghhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh";
////    std::string d5 = "iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiijjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj";
////    std::string d6 = "kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkllllllllllllllllllllllllllllllll";
////    std::string d7 = "mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn";
////    std::string d8 = "oooooooooooooooooooooooooooooooopppppppppppppppppppppppppppppppp";
////    
////    auto hd1 = sha256_double_serial(d1);
////    auto hd2 = sha256_double_serial(d2);
////    auto hd3 = sha256_double_serial(d3);
////    auto hd4 = sha256_double_serial(d4);
////    auto hd5 = sha256_double_serial(d5);
////    auto hd6 = sha256_double_serial(d6);
////    auto hd7 = sha256_double_serial(d7);
////    auto hd8 = sha256_double_serial(d8);
////    
////    auto hd1d2 = sha256_double_serial(concat_hash(hd1, hd2));
////    auto hd3d4 = sha256_double_serial(concat_hash(hd3, hd4));
////    auto hd5d6 = sha256_double_serial(concat_hash(hd5, hd6));
////    auto hd7d8 = sha256_double_serial(concat_hash(hd7, hd8));
////    
////    auto hd1d2d3d4 = sha256_double_serial(concat_hash(hd1d2, hd3d4));
////    auto hd5d6d7d8 = sha256_double_serial(concat_hash(hd5d6, hd7d8));
////
////    auto hd1d2d3d4d5d6d7d8 = sha256_double_serial(concat_hash(hd1d2d3d4, hd5d6d7d8));
////    
////    print_hash(hd1d2d3d4d5d6d7d8);
////    
////    
////    
////    
////    std::cout << "------------------------------------------------------\n";
////
////    
////    
////    
////    std::string temp = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8;
////    std::array<uint8_t, 512> da;
////    std::memcpy(da.data(), temp.data(), da.size());
////    
////    //    print_bytes(da.data(), 256);
////    
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(reinterpret_cast<uint8_t*>(da.data() + 0)); //512 -> 256        16 -> 8 | 0  -> 0
//////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(reinterpret_cast<uint8_t*>(da.data() + 0)); //256 -> 128      8  -> 4 | 8  -> 4
//////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(reinterpret_cast<uint8_t*>(da.data() + 0)); //128 -> 64       4  -> 2 | 12 -> 6
//////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(reinterpret_cast<uint8_t*>(da.data() + 0)); //64 -> 32        2  -> 1 | 14 -> 7
////    
////    print_bytes(da.data() + 0, 32);
////    print_bytes(da.data() + 32, 32);
////    print_bytes(da.data() + 64, 32);
////    print_bytes(da.data() + 96, 32);
////    print_bytes(da.data() + 128, 32);
////    print_bytes(da.data() + 160, 32);
////    print_bytes(da.data() + 192, 32);
////    print_bytes(da.data() + 224, 32);
//////    print_bytes(da.data() + 256, 32);
//////    print_bytes(da.data() + 288, 32);
//////    print_bytes(da.data() + 320, 32);
//////    print_bytes(da.data() + 352, 32);
//////    print_bytes(da.data() + 384, 32);
//////    print_bytes(da.data() + 416, 32);
//////    print_bytes(da.data() + 448, 32);
//////    print_bytes(da.data() + 480, 32);
//    
//    






//
//
//	// ---------------------------------------
//
//    
////    print_bytes(buffer, 32);
//    
////    //First 16 Tx's
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);
////
////    //2nd 16 Tx's
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 512);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 512);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 512);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 512);
////
////    //3th 16 Tx's
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1024);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1024);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1024);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1024);
////
////    //3th 16 Tx's
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1536);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1536);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1536);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1536);
////    
////    hash_t h1;
////    std::memcpy(h1.data(), buffer, 32);
////    hash_t h2;
////    std::memcpy(h2.data(), buffer + 512, 32);
////    hash_t h3;
////    std::memcpy(h3.data(), buffer + 1024, 32);
////    hash_t h4;
////    std::memcpy(h4.data(), buffer + 1536, 32);
////
////    auto h1h2 = sha256_double_serial(concat_hash(h1, h2));
////    auto h3h4 = sha256_double_serial(concat_hash(h3, h4));
////
////    auto h1h2h3h4 = sha256_double_serial(concat_hash(h1h2, h3h4));
////    print_bytes(h1h2h3h4.data(),  32);
//
//
//    
//    //----------------------------------------------------------------
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);        //16
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 512);  //16
////    std::memcpy(buffer +  256, buffer +  512, 256);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);        //16
////    
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1024); //16
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1536); //16
////    std::memcpy(buffer + 1280, buffer + 1536, 256);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1024); //16
////    
////    std::memcpy(buffer +  256, buffer + 1024, 256);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);        //16
////    
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);
////    SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);
//
////----------------------------------------------------------------
//
//	
//	//SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);                        //16
// //   SHA256_SIMD_3_DOUBLE_TWO_BUFFERS(buffer + 512, buffer + 256);  //16
// //   
// //   SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer + 1024);                 //16
// //   SHA256_SIMD_3_DOUBLE_TWO_BUFFERS(buffer + 1536, buffer + 1280); //16
// //   
// //   SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);        //16
// //   SHA256_SIMD_3_DOUBLE_TWO_BUFFERS(buffer + 1024, buffer + 256); //16
// //   
// //   SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);        //16
// //   
// //   SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);
// //   SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);
// //   SHA256_SIMD_3_DOUBLE_ONE_BUFFER(buffer);
//    //print_bytes(buffer, 32);
    
    return 0;
}

