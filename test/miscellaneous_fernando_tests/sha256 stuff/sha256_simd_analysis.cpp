// clang++ -O3 -std=c++14 sha256_simd_analysis.cpp
// g++ -O3 -std=c++14 sha256_simd_analysis.cpp



// ----------------------------------------------------------------
// ----------------------------------------------------------------
// SIMD - SSE3 (from Parallelizing message schedules to accelerate the computations of hash functions)
// ----------------------------------------------------------------
// ----------------------------------------------------------------


// this function takes a word from each chunk, and puts it in a single register
inline 
__m128i gather(unsigned int* address) {
    __m128i temp;
    temp = _mm_cvtsi32_si128(address[0]);
    temp = _mm_insert_epi32(temp, address[16], 1);
    temp = _mm_insert_epi32(temp, address[32], 2);
    temp = _mm_insert_epi32(temp, address[48], 3);
    return temp;
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

void SHA256_QMS(__m128i schedule[64], uint32_t message[64]) {
    __m128i bswap_mask = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    __m128i W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15;
    __m128i s0, s1, Wi, *Ki = (__m128i*)k;

    W0 = gather(message);
    W1 = gather(&message[1]);
    W2 = gather(&message[2]);
    W3 = gather(&message[3]);
    W4 = gather(&message[4]);
    W5 = gather(&message[5]);
    W6 = gather(&message[6]);
    W7 = gather(&message[7]);
    W8 = gather(&message[8]);
    W9 = gather(&message[9]);
    W10 = gather(&message[10]);
    W11 = gather(&message[11]);
    W12 = gather(&message[12]);
    W13 = gather(&message[13]);
    W14 = gather(&message[14]);
    W15 = gather(&message[15]);

    W0  = _mm_shuffle_epi8(W0, bswap_mask);
    W1  = _mm_shuffle_epi8(W1, bswap_mask);
    W2  = _mm_shuffle_epi8(W2, bswap_mask);
    W3  = _mm_shuffle_epi8(W3, bswap_mask);
    W4  = _mm_shuffle_epi8(W4, bswap_mask);
    W5  = _mm_shuffle_epi8(W5, bswap_mask);
    W6  = _mm_shuffle_epi8(W6, bswap_mask);
    W7  = _mm_shuffle_epi8(W7, bswap_mask);
    W8  = _mm_shuffle_epi8(W8, bswap_mask);
    W9  = _mm_shuffle_epi8(W9, bswap_mask);
    W10 = _mm_shuffle_epi8(W10, bswap_mask);
    W11 = _mm_shuffle_epi8(W11, bswap_mask);
    W12 = _mm_shuffle_epi8(W12, bswap_mask);
    W13 = _mm_shuffle_epi8(W13, bswap_mask);
    W14 = _mm_shuffle_epi8(W14, bswap_mask);
    W15 = _mm_shuffle_epi8(W15, bswap_mask);

    int i;
    for (i = 0; i<32; ) {
        SCHEDULE_ROUND(W1 , W14, W0 , W9 );
        SCHEDULE_ROUND(W2 , W15, W1 , W10);
        SCHEDULE_ROUND(W3 , W0 , W2 , W11);
        SCHEDULE_ROUND(W4 , W1 , W3 , W12);
        SCHEDULE_ROUND(W5 , W2 , W4 , W13);
        SCHEDULE_ROUND(W6 , W3 , W5 , W14);
        SCHEDULE_ROUND(W7 , W4 , W6 , W15);
        SCHEDULE_ROUND(W8 , W5 , W7 , W0 );
        SCHEDULE_ROUND(W9 , W6 , W8 , W1 );
        SCHEDULE_ROUND(W10, W7 , W9 , W2 );
        SCHEDULE_ROUND(W11, W8 , W10, W3 );
        SCHEDULE_ROUND(W12, W9 , W11, W4 );
        SCHEDULE_ROUND(W13, W10, W12, W5 );
        SCHEDULE_ROUND(W14, W11, W13, W6 );
        SCHEDULE_ROUND(W15, W12, W14, W7 );
        SCHEDULE_ROUND(W0 , W13, W15, W8 );
    }

    SCHEDULE_ROUND(W1 , W14, W0 , W9 );
    schedule[48] = _mm_add_epi32(W0, Ki[48]);
    SCHEDULE_ROUND(W2 , W15, W1 , W10);
    schedule[49] = _mm_add_epi32(W1, Ki[49]);
    SCHEDULE_ROUND(W3 , W0 , W2 , W11);
    schedule[50] = _mm_add_epi32(W2, Ki[50]);
    SCHEDULE_ROUND(W4 , W1 , W3 , W12);
    schedule[51] = _mm_add_epi32(W3, Ki[51]);
    SCHEDULE_ROUND(W5 , W2 , W4 , W13);
    schedule[52] = _mm_add_epi32(W4, Ki[52]);
    SCHEDULE_ROUND(W6 , W3 , W5 , W14);
    schedule[53] = _mm_add_epi32(W5, Ki[53]);
    SCHEDULE_ROUND(W7 , W4 , W6 , W15);
    schedule[54] = _mm_add_epi32(W6, Ki[54]);
    SCHEDULE_ROUND(W8 , W5 , W7 , W0 );
    schedule[55] = _mm_add_epi32(W7, Ki[55]);
    SCHEDULE_ROUND(W9 , W6 , W8 , W1 );
    schedule[56] = _mm_add_epi32(W8, Ki[56]);
    SCHEDULE_ROUND(W10, W7 , W9 , W2 );
    schedule[57] = _mm_add_epi32(W9, Ki[57]);
    SCHEDULE_ROUND(W11, W8 , W10, W3 );
    schedule[58] = _mm_add_epi32(W10, Ki[58]);
    SCHEDULE_ROUND(W12, W9 , W11, W4 );
    schedule[59] = _mm_add_epi32(W11, Ki[59]);
    SCHEDULE_ROUND(W13, W10, W12, W5 );
    schedule[60] = _mm_add_epi32(W12, Ki[60]);
    SCHEDULE_ROUND(W14, W11, W13, W6 );
    schedule[61] = _mm_add_epi32(W13, Ki[61]);
    SCHEDULE_ROUND(W15, W12, W14, W7 );
    schedule[62] = _mm_add_epi32(W14, Ki[62]);
    SCHEDULE_ROUND(W0 , W13, W15, W8 );
    schedule[63] = _mm_add_epi32(W15, Ki[63]);
}




// ----------------------------------------------------------------
// ----------------------------------------------------------------
// SIMD - AVX2 
// (from Parallelizing message schedules to accelerate the computations of hash functions)
// ----------------------------------------------------------------
// ----------------------------------------------------------------


// #define vpbroadcastq(vec, k) vec = _mm256_broadcastq_epi64(*(__m128i*)k)

// // this function calculates the small sigma 0 transformation
// inline 
// __m256i sigma_0(__m256i W) {
//     return
//         _mm256_xor_si256(
//             _mm256_xor_si256(
//                 _mm256_xor_si256(
//                     _mm256_srli_epi64(W, 7),
//                     _mm256_srli_epi64(W, 8)
//                 ),
//                 _mm256_xor_si256(
//                     _mm256_srli_epi64(W, 1),
//                     _mm256_slli_epi64(W, 56)
//                 )
//             ),
//             _mm256_slli_epi64(W, 63)
//         );
// }

// // this function calculates the small sigma 1 transformation
// inline
// __m256i sigma_1(__m256i W) {
//     return
//         _mm256_xor_si256(
//             _mm256_xor_si256(
//                 _mm256_xor_si256(
//                     _mm256_srli_epi64(W, 6),
//                     _mm256_srli_epi64(W, 61)
//                 ),
//                 _mm256_xor_si256(
//                     _mm256_srli_epi64(W, 19),
//                     _mm256_slli_epi64(W, 3)
//                 )
//             ),
//             _mm256_slli_epi64(W, 45)
//         );
// }

// // the message scheduling round
// #define SCHEDULE_ROUND(w1, w2, w3, w4) \
//     vpbroadcastq(Ki, &k[i]);\
//     s0 = sigma_0(w1); \
//     s1 = sigma_1(w2); \
//     schedule[i] = _mm256_add_epi64(w3, Ki); \
//     w3 = _mm256_add_epi64( \
//         _mm256_add_epi64(w3, w4), \
//         _mm256_add_epi64(s0, s1) \
//     ); \
//     i++;

// void SHA512_QMS(__m256i schedule[80], uint64_t message[64]) {
//     __m256i gather_mask = _mm256_setr_epi64x(0, 16, 32, 48);
//     __m256i bswap_mask = _mm256_set_epi8(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7);
//     __m256i W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15;
//     __m256i s0, s1, Ki, Wi;

//     int i;

//     W0 = _mm256_i64gather_epi64((const long*)message, gather_mask, 8);
//     W0 = _mm256_shuffle_epi8(W0, bswap_mask);
//     W1 = _mm256_i64gather_epi64((const long*)&message[1], gather_mask, 8);
//     W1 = _mm256_shuffle_epi8(W1, bswap_mask);
//     W2 = _mm256_i64gather_epi64((const long*)&message[2], gather_mask, 8);
//     W2 = _mm256_shuffle_epi8(W2, bswap_mask);
//     W3 = _mm256_i64gather_epi64((const long*)&message[3], gather_mask, 8);
//     W3 = _mm256_shuffle_epi8(W3, bswap_mask);
//     W4 = _mm256_i64gather_epi64((const long*)&message[4], gather_mask, 8);
//     W4 = _mm256_shuffle_epi8(W4, bswap_mask);
//     W5 = _mm256_i64gather_epi64((const long*)&message[5], gather_mask, 8);
//     W5 = _mm256_shuffle_epi8(W5, bswap_mask);
//     W6 = _mm256_i64gather_epi64((const long*)&message[6], gather_mask, 8);
//     W6 = _mm256_shuffle_epi8(W6, bswap_mask);
//     W7 = _mm256_i64gather_epi64((const long*)&message[7], gather_mask, 8);
//     W7 = _mm256_shuffle_epi8(W7, bswap_mask);
//     W8 = _mm256_i64gather_epi64((const long*)&message[8], gather_mask, 8);
//     W8 = _mm256_shuffle_epi8(W8, bswap_mask);
//     W9 = _mm256_i64gather_epi64((const long*)&message[9], gather_mask, 8);
//     W9 = _mm256_shuffle_epi8(W9, bswap_mask);
//     W10 = _mm256_i64gather_epi64((const long*)&message[10], gather_mask, 8);
//     W10 = _mm256_shuffle_epi8(W10, bswap_mask);
//     W11 = _mm256_i64gather_epi64((const long*)&message[11], gather_mask, 8);
//     W11 = _mm256_shuffle_epi8(W11, bswap_mask);
//     W12 = _mm256_i64gather_epi64((const long*)&message[12], gather_mask, 8);

//     W12 = _mm256_shuffle_epi8(W12, bswap_mask);
//     W13 = _mm256_i64gather_epi64((const long*)&message[13], gather_mask, 8);
//     W13 = _mm256_shuffle_epi8(W13, bswap_mask);
//     W14 = _mm256_i64gather_epi64((const long*)&message[14], gather_mask, 8);
//     W14 = _mm256_shuffle_epi8(W14, bswap_mask);
//     W15 = _mm256_i64gather_epi64((const long*)&message[15], gather_mask, 8);
//     W15 = _mm256_shuffle_epi8(W15, bswap_mask);

//     for(i=0; i<64; ) {
//         SCHEDULE_ROUND(W1, W14, W0, W9);
//         SCHEDULE_ROUND(W2, W15, W1, W10);
//         SCHEDULE_ROUND(W3, W0, W2, W11);
//         SCHEDULE_ROUND(W4, W1, W3, W12);
//         SCHEDULE_ROUND(W5, W2, W4, W13);
//         SCHEDULE_ROUND(W6, W3, W5, W14);
//         SCHEDULE_ROUND(W7, W4, W6, W15);
//         SCHEDULE_ROUND(W8, W5, W7, W0);
//         SCHEDULE_ROUND(W9, W6, W8, W1);
//         SCHEDULE_ROUND(W10, W7, W9, W2);
//         SCHEDULE_ROUND(W11, W8, W10, W3);
//         SCHEDULE_ROUND(W12, W9, W11, W4);
//         SCHEDULE_ROUND(W13, W10, W12, W5);
//         SCHEDULE_ROUND(W14, W11, W13, W6);
//         SCHEDULE_ROUND(W15, W12, W14, W7);
//         SCHEDULE_ROUND(W0, W13, W15, W8);
//     }

//     schedule[64] = _mm256_add_epi64(W0, _mm256_broadcastq_epi64(*(__m128i*)&k[64]));
//     schedule[65] = _mm256_add_epi64(W1, _mm256_broadcastq_epi64(*(__m128i*)&k[65]));
//     schedule[66] = _mm256_add_epi64(W2, _mm256_broadcastq_epi64(*(__m128i*)&k[66]));
//     schedule[67] = _mm256_add_epi64(W3, _mm256_broadcastq_epi64(*(__m128i*)&k[67]));
//     schedule[68] = _mm256_add_epi64(W4, _mm256_broadcastq_epi64(*(__m128i*)&k[68]));
//     schedule[69] = _mm256_add_epi64(W5, _mm256_broadcastq_epi64(*(__m128i*)&k[69]));
//     schedule[70] = _mm256_add_epi64(W6, _mm256_broadcastq_epi64(*(__m128i*)&k[70]));
//     schedule[71] = _mm256_add_epi64(W7, _mm256_broadcastq_epi64(*(__m128i*)&k[71]));
//     schedule[72] = _mm256_add_epi64(W8, _mm256_broadcastq_epi64(*(__m128i*)&k[72]));
//     schedule[73] = _mm256_add_epi64(W9, _mm256_broadcastq_epi64(*(__m128i*)&k[73]));
//     schedule[74] = _mm256_add_epi64(W10, _mm256_broadcastq_epi64(*(__m128i*)&k[74]));
//     schedule[75] = _mm256_add_epi64(W11, _mm256_broadcastq_epi64(*(__m128i*)&k[75]));
//     schedule[76] = _mm256_add_epi64(W12, _mm256_broadcastq_epi64(*(__m128i*)&k[76]));
//     schedule[77] = _mm256_add_epi64(W13, _mm256_broadcastq_epi64(*(__m128i*)&k[77]));
//     schedule[78] = _mm256_add_epi64(W14, _mm256_broadcastq_epi64(*(__m128i*)&k[78]));
//     schedule[79] = _mm256_add_epi64(W15, _mm256_broadcastq_epi64(*(__m128i*)&k[79]));
// }


