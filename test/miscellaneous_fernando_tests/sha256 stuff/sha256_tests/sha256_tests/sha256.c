/* libsodium: hash_sha256.c, v0.4.5 2014/04/16 */
/**
 * Copyright 2005,2007,2009 Colin Percival. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
#include "sha256.h"

#include <stdint.h>
#include <string.h>
#include "zeroize.h"

/*static*/ uint32_t be32dec(const void* pp)
{
    const uint8_t* p = (uint8_t const*)pp;

    return ((uint32_t)(p[3]) + ((uint32_t)(p[2]) << 8) +
        ((uint32_t)(p[1]) << 16) + ((uint32_t)(p[0]) << 24));
}

/*static*/ void be32enc(void* pp, uint32_t x)
{
    uint8_t* p = (uint8_t*)pp;

    p[3] = x & 0xff;
    p[2] = (x >> 8) & 0xff;
    p[1] = (x >> 16) & 0xff;
    p[0] = (x >> 24) & 0xff;
}

/*static*/ void be32enc_vect(uint8_t* dst, const uint32_t* src, size_t len)
{
    size_t i;
    for (i = 0; i < len / 4; i++) 
    {
        be32enc(dst + i * 4, src[i]);
    }
}

/*static*/ void be32dec_vect(uint32_t* dst, const uint8_t* src, size_t len)
{
    size_t i;
    for (i = 0; i < len / 4; i++) 
    {
        dst[i] = be32dec(src + i * 4);
    }
}


inline
uint8_t const* as_bytes(uint32_t const* x) {
    return (uint8_t const*)(void const*)x;
}
inline
uint32_t const* as_words(uint8_t const* x) {
    return (uint32_t const*)(void const*)x;
}

static void be32dec_vect_optimized(uint32_t* dst, const uint8_t* src, size_t len)
{
    memcpy(dst, as_words(src), len / 4);
}


#define Ch(x, y, z)  ((x & (y ^ z)) ^ z)
#define Maj(x, y, z) ((x & (y | z)) | (y & z))
#define SHR(x, n)    (x >> n)
#define ROTR(x, n)   ((x >> n) | (x << (32 - n)))
#define S0(x)        (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x)        (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define s0(x)        (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define s1(x)        (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

#define RND(a, b, c, d, e, f, g, h, k) \
    t0 = h + S1(e) + Ch(e, f, g) + k;  \
    t1 = S0(a) + Maj(a, b, c); \
    d += t0; \
    h = t0 + t1;

//printf("i: %d, k: %#010x, W[%d]: %#010x, W[%d] + k: %#010x\n", i, k, i, W[i], i, W[i] + k); 

#define RNDr(S, W, i, k) \
	RND(S[(64 - i) % 8], S[(65 - i) % 8], \
    S[(66 - i) % 8], S[(67 - i) % 8], \
    S[(68 - i) % 8], S[(69 - i) % 8], \
    S[(70 - i) % 8], S[(71 - i) % 8], \
    W[i] + k)

static unsigned char PAD[SHA256_BLOCK_LENGTH] =
{
    0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

void SHA256_(const uint8_t* input, size_t length,
    uint8_t digest[SHA256_DIGEST_LENGTH])
{
    SHA256CTX context;
	
    SHA256Init(&context);
    SHA256Update(&context, input, length);
    SHA256Final(&context, digest);
}

void SHA256Final(SHA256CTX* context, uint8_t digest[SHA256_DIGEST_LENGTH])
{
    SHA256Pad(context);
    be32enc_vect(digest, context->state, SHA256_DIGEST_LENGTH);
    zeroize((void*)context, sizeof *context);
}

void SHA256Init(SHA256CTX* context)
{
    context->count[0] = context->count[1] = 0;

    context->state[0] = 0x6A09E667;
    context->state[1] = 0xBB67AE85;
    context->state[2] = 0x3C6EF372;
    context->state[3] = 0xA54FF53A;
    context->state[4] = 0x510E527F;
    context->state[5] = 0x9B05688C;
    context->state[6] = 0x1F83D9AB;
    context->state[7] = 0x5BE0CD19;
}



void SHA256Pad(SHA256CTX* context)
{
    uint8_t len[8];
    uint32_t r, plen;

    be32enc_vect(len, context->count, 8);

    r = (context->count[1] >> 3) & 0x3f;
    plen = (r < 56) ? (56 - r) : (120 - r);

    SHA256Update(context, PAD, plen);
    SHA256Update(context, len, 8);
}

void SHA256Transform(uint32_t state[SHA256_STATE_LENGTH],   //uint32_t state[8]         = 256 bits
    const uint8_t block[SHA256_BLOCK_LENGTH])				//const uint8_t block[64]   = 512 bits
{
    int i;
    uint32_t W[64];
    uint32_t S[8];
    uint32_t t0, t1;

    be32dec_vect(W, block, SHA256_BLOCK_LENGTH);

    for (i = 16; i < 64; i++)
    {
        W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
    }

    memcpy(S, state, 32);
	
    RNDr(S, W, 0, 0x428a2f98);
    RNDr(S, W, 1, 0x71374491);
	RNDr(S, W, 2, 0xb5c0fbcf);
	RNDr(S, W, 3, 0xe9b5dba5);
	RNDr(S, W, 4, 0x3956c25b);
	RNDr(S, W, 5, 0x59f111f1);
	RNDr(S, W, 6, 0x923f82a4);
	RNDr(S, W, 7, 0xab1c5ed5);
	RNDr(S, W, 8, 0xd807aa98);
	RNDr(S, W, 9, 0x12835b01);
	RNDr(S, W, 10, 0x243185be);
	RNDr(S, W, 11, 0x550c7dc3);
	RNDr(S, W, 12, 0x72be5d74);
	RNDr(S, W, 13, 0x80deb1fe);
	RNDr(S, W, 14, 0x9bdc06a7);
	RNDr(S, W, 15, 0xc19bf174);

    RNDr(S, W, 16, 0xe49b69c1);
	RNDr(S, W, 17, 0xefbe4786);
	RNDr(S, W, 18, 0x0fc19dc6);
	RNDr(S, W, 19, 0x240ca1cc);
	RNDr(S, W, 20, 0x2de92c6f);
	RNDr(S, W, 21, 0x4a7484aa);
	RNDr(S, W, 22, 0x5cb0a9dc);
	RNDr(S, W, 23, 0x76f988da);
	RNDr(S, W, 24, 0x983e5152);
	RNDr(S, W, 25, 0xa831c66d);
	RNDr(S, W, 26, 0xb00327c8);
	RNDr(S, W, 27, 0xbf597fc7);
	RNDr(S, W, 28, 0xc6e00bf3);
	RNDr(S, W, 29, 0xd5a79147);
	RNDr(S, W, 30, 0x06ca6351);
	RNDr(S, W, 31, 0x14292967);

    RNDr(S, W, 32, 0x27b70a85);
	RNDr(S, W, 33, 0x2e1b2138);
	RNDr(S, W, 34, 0x4d2c6dfc);
	RNDr(S, W, 35, 0x53380d13);
	RNDr(S, W, 36, 0x650a7354);
	RNDr(S, W, 37, 0x766a0abb);
	RNDr(S, W, 38, 0x81c2c92e);
	RNDr(S, W, 39, 0x92722c85);
	RNDr(S, W, 40, 0xa2bfe8a1);
	RNDr(S, W, 41, 0xa81a664b);
	RNDr(S, W, 42, 0xc24b8b70);
	RNDr(S, W, 43, 0xc76c51a3);
	RNDr(S, W, 44, 0xd192e819);
	RNDr(S, W, 45, 0xd6990624);
	RNDr(S, W, 46, 0xf40e3585);
	RNDr(S, W, 47, 0x106aa070);

    RNDr(S, W, 48, 0x19a4c116);
	RNDr(S, W, 49, 0x1e376c08);
	RNDr(S, W, 50, 0x2748774c);
	RNDr(S, W, 51, 0x34b0bcb5);
	RNDr(S, W, 52, 0x391c0cb3);
	RNDr(S, W, 53, 0x4ed8aa4a);
	RNDr(S, W, 54, 0x5b9cca4f);
	RNDr(S, W, 55, 0x682e6ff3);
	RNDr(S, W, 56, 0x748f82ee);
	RNDr(S, W, 57, 0x78a5636f);
	RNDr(S, W, 58, 0x84c87814);
	RNDr(S, W, 59, 0x8cc70208);
	RNDr(S, W, 60, 0x90befffa);
	RNDr(S, W, 61, 0xa4506ceb);
	RNDr(S, W, 62, 0xbef9a3f7);
	RNDr(S, W, 63, 0xc67178f2);
	
	//printf("---------------------------------------------------------------------\n");

	for (i = 0; i < 8; i++)
	{
		state[i] += S[i];
	}

	//for (size_t i = 0; i < 8; ++i) {
	//	printf("state[%d]: %#010x\n", i, state[i]);
	//}
	//printf("---------------------------------------------------------\n");




    zeroize((void*)W, sizeof W);
    zeroize((void*)S, sizeof S);
    zeroize((void*)&t0, sizeof t0);
    zeroize((void*)&t1, sizeof t1);
}

void SHA256Update(SHA256CTX* context, const uint8_t* input, size_t length)
{
    uint32_t bitlen[2];
    uint32_t r = (context->count[1] >> 3) & 0x3f;

    bitlen[1] = ((uint32_t)length) << 3;
    bitlen[0] = (uint32_t)(length >> 29);

    if ((context->count[1] += bitlen[1]) < bitlen[1])
    {
        context->count[0]++;
    }

    context->count[0] += bitlen[0];

    if (length < 64 - r)
    {
        memcpy(&context->buf[r], input, length);
        return;
    }

    memcpy(&context->buf[r], input, 64 - r);
    SHA256Transform(context->state, context->buf);

    input += 64 - r;
    length -= 64 - r;

    while (length >= 64)
    {
        SHA256Transform(context->state, input);
        input += 64;
        length -= 64;
    }

    memcpy(context->buf, input, length);
}



// ----------------------------------------------------------------



void SHA256Opt1_(const uint8_t* input, size_t length,
             uint8_t digest[SHA256_DIGEST_LENGTH])
{
    SHA256CTX context;
    SHA256Init(&context);
    SHA256UpdateOpt1(&context, input, length);
    SHA256FinalOpt1(&context, digest);
}

void SHA256OptDouble_(const uint8_t* input, size_t length,
                 uint8_t digest[SHA256_DIGEST_LENGTH])
{
    SHA256CTX context;

    SHA256Init(&context);
    SHA256UpdateOpt1(&context, input, length);
    SHA256FinalOpt1(&context, digest);
    
    SHA256Init(&context);
    SHA256UpdateOptDouble(&context, digest);
    SHA256FinalOptDouble(&context, digest);
}

void SHA256OptDoubleDualBuffer_(uint8_t const* input1, uint8_t const* input2, uint8_t digest[SHA256_DIGEST_LENGTH]) {
    //precondition: length of input1 == 32 && length of input2 == 32

    SHA256CTX context;

    SHA256Init(&context);
    SHA256UpdateDualFixed32(&context, input1, input2);

    uint8_t PADXX[SHA256_BLOCK_LENGTH] = {
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0
    };

    SHA256TransformOpt1(context.state, PADXX);

    be32enc_vect(digest, context.state, SHA256_DIGEST_LENGTH);
    zeroize((void*)&context, sizeof(context));

    //SHA256FinalOpt1(&context, digest);
    
    SHA256Init(&context);
    SHA256UpdateOptDouble(&context, digest);
    SHA256FinalOptDouble(&context, digest);
}

void SHA256FinalOpt1(SHA256CTX* context, uint8_t digest[SHA256_DIGEST_LENGTH])
{
    SHA256PadOpt1(context);
    be32enc_vect(digest, context->state, SHA256_DIGEST_LENGTH); //TODO: ver si se puede optimizar
    zeroize((void*)context, sizeof *context);                   //TODO: ver si se puede aplicar C++ para eliminar
}

void SHA256FinalOptDouble(SHA256CTX* context, uint8_t digest[SHA256_DIGEST_LENGTH])
{
    be32enc_vect(digest, context->state, SHA256_DIGEST_LENGTH); //TODO: ver si se puede optimizar
    zeroize((void*)context, sizeof *context);                   //TODO: ver si se puede aplicar C++ para eliminar
}

void SHA256PadOpt1(SHA256CTX* context)
{
    uint8_t len[8];
    uint32_t r, plen;
    
    be32enc_vect(len, context->count, 8);
    
    r = (context->count[1] >> 3) & 0x3f;
    plen = (r < 56) ? (56 - r) : (120 - r);
    
    SHA256UpdateOpt1(context, PAD, plen);
    SHA256UpdateOpt1(context, len, 8);
}

void SHA256TransformOpt1(uint32_t state[SHA256_STATE_LENGTH],
                     const uint8_t block[SHA256_BLOCK_LENGTH])
{
    int i;
    uint32_t W[64];
    uint32_t S[8];
    uint32_t t0, t1;
    
    be32dec_vect(W, block, SHA256_BLOCK_LENGTH);
    
    for (i = 16; i < 64; i++) {
        W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
    }
    
    memcpy(S, state, 32);
    
    RNDr(S, W, 0, 0x428a2f98);
    RNDr(S, W, 1, 0x71374491);
    RNDr(S, W, 2, 0xb5c0fbcf);
    RNDr(S, W, 3, 0xe9b5dba5);
    RNDr(S, W, 4, 0x3956c25b);
    RNDr(S, W, 5, 0x59f111f1);
    RNDr(S, W, 6, 0x923f82a4);
    RNDr(S, W, 7, 0xab1c5ed5);
    RNDr(S, W, 8, 0xd807aa98);
    RNDr(S, W, 9, 0x12835b01);
    RNDr(S, W, 10, 0x243185be);
    RNDr(S, W, 11, 0x550c7dc3);
    RNDr(S, W, 12, 0x72be5d74);
    RNDr(S, W, 13, 0x80deb1fe);
    RNDr(S, W, 14, 0x9bdc06a7);
    RNDr(S, W, 15, 0xc19bf174);
    RNDr(S, W, 16, 0xe49b69c1);
    RNDr(S, W, 17, 0xefbe4786);
    RNDr(S, W, 18, 0x0fc19dc6);
    RNDr(S, W, 19, 0x240ca1cc);
    RNDr(S, W, 20, 0x2de92c6f);
    RNDr(S, W, 21, 0x4a7484aa);
    RNDr(S, W, 22, 0x5cb0a9dc);
    RNDr(S, W, 23, 0x76f988da);
    RNDr(S, W, 24, 0x983e5152);
    RNDr(S, W, 25, 0xa831c66d);
    RNDr(S, W, 26, 0xb00327c8);
    RNDr(S, W, 27, 0xbf597fc7);
    RNDr(S, W, 28, 0xc6e00bf3);
    RNDr(S, W, 29, 0xd5a79147);
    RNDr(S, W, 30, 0x06ca6351);
    RNDr(S, W, 31, 0x14292967);
    RNDr(S, W, 32, 0x27b70a85);
    RNDr(S, W, 33, 0x2e1b2138);
    RNDr(S, W, 34, 0x4d2c6dfc);
    RNDr(S, W, 35, 0x53380d13);
    RNDr(S, W, 36, 0x650a7354);
    RNDr(S, W, 37, 0x766a0abb);
    RNDr(S, W, 38, 0x81c2c92e);
    RNDr(S, W, 39, 0x92722c85);
    RNDr(S, W, 40, 0xa2bfe8a1);
    RNDr(S, W, 41, 0xa81a664b);
    RNDr(S, W, 42, 0xc24b8b70);
    RNDr(S, W, 43, 0xc76c51a3);
    RNDr(S, W, 44, 0xd192e819);
    RNDr(S, W, 45, 0xd6990624);
    RNDr(S, W, 46, 0xf40e3585);
    RNDr(S, W, 47, 0x106aa070);
    RNDr(S, W, 48, 0x19a4c116);
    RNDr(S, W, 49, 0x1e376c08);
    RNDr(S, W, 50, 0x2748774c);
    RNDr(S, W, 51, 0x34b0bcb5);
    RNDr(S, W, 52, 0x391c0cb3);
    RNDr(S, W, 53, 0x4ed8aa4a);
    RNDr(S, W, 54, 0x5b9cca4f);
    RNDr(S, W, 55, 0x682e6ff3);
    RNDr(S, W, 56, 0x748f82ee);
    RNDr(S, W, 57, 0x78a5636f);
    RNDr(S, W, 58, 0x84c87814);
    RNDr(S, W, 59, 0x8cc70208);
    RNDr(S, W, 60, 0x90befffa);
    RNDr(S, W, 61, 0xa4506ceb);
    RNDr(S, W, 62, 0xbef9a3f7);
    RNDr(S, W, 63, 0xc67178f2);
    
    for (i = 0; i < 8; i++) {
        state[i] += S[i];
    }
    
    zeroize((void*)W, sizeof W);
    zeroize((void*)S, sizeof S);
    zeroize((void*)&t0, sizeof t0);
    zeroize((void*)&t1, sizeof t1);
}

void SHA256TransformOptDouble(uint32_t state[SHA256_STATE_LENGTH],      // 32 bytes ... 256 bits
                         const uint8_t block[SHA256_BLOCK_LENGTH])      // 64 bytes ... 512 bits
{
    int i;
    uint32_t W[64];
    uint32_t S[8];
    uint32_t t0, t1;
    
    be32dec_vect(W, block, SHA256_BLOCK_LENGTH);
    
    for (i = 16; i < 64; i++)
    {
        W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
    }
    
    memcpy(S, state, 32);
    
    RNDr(S, W, 0, 0x428a2f98);
    RNDr(S, W, 1, 0x71374491);
    RNDr(S, W, 2, 0xb5c0fbcf);
    RNDr(S, W, 3, 0xe9b5dba5);
    RNDr(S, W, 4, 0x3956c25b);
    RNDr(S, W, 5, 0x59f111f1);
    RNDr(S, W, 6, 0x923f82a4);
    RNDr(S, W, 7, 0xab1c5ed5);
    RNDr(S, W, 8, 0xd807aa98);
    RNDr(S, W, 9, 0x12835b01);
    RNDr(S, W, 10, 0x243185be);
    RNDr(S, W, 11, 0x550c7dc3);
    RNDr(S, W, 12, 0x72be5d74);
    RNDr(S, W, 13, 0x80deb1fe);
    RNDr(S, W, 14, 0x9bdc06a7);
    RNDr(S, W, 15, 0xc19bf174);
    RNDr(S, W, 16, 0xe49b69c1);
    RNDr(S, W, 17, 0xefbe4786);
    RNDr(S, W, 18, 0x0fc19dc6);
    RNDr(S, W, 19, 0x240ca1cc);
    RNDr(S, W, 20, 0x2de92c6f);
    RNDr(S, W, 21, 0x4a7484aa);
    RNDr(S, W, 22, 0x5cb0a9dc);
    RNDr(S, W, 23, 0x76f988da);
    RNDr(S, W, 24, 0x983e5152);
    RNDr(S, W, 25, 0xa831c66d);
    RNDr(S, W, 26, 0xb00327c8);
    RNDr(S, W, 27, 0xbf597fc7);
    RNDr(S, W, 28, 0xc6e00bf3);
    RNDr(S, W, 29, 0xd5a79147);
    RNDr(S, W, 30, 0x06ca6351);
    RNDr(S, W, 31, 0x14292967);
    RNDr(S, W, 32, 0x27b70a85);
    RNDr(S, W, 33, 0x2e1b2138);
    RNDr(S, W, 34, 0x4d2c6dfc);
    RNDr(S, W, 35, 0x53380d13);
    RNDr(S, W, 36, 0x650a7354);
    RNDr(S, W, 37, 0x766a0abb);
    RNDr(S, W, 38, 0x81c2c92e);
    RNDr(S, W, 39, 0x92722c85);
    RNDr(S, W, 40, 0xa2bfe8a1);
    RNDr(S, W, 41, 0xa81a664b);
    RNDr(S, W, 42, 0xc24b8b70);
    RNDr(S, W, 43, 0xc76c51a3);
    RNDr(S, W, 44, 0xd192e819);
    RNDr(S, W, 45, 0xd6990624);
    RNDr(S, W, 46, 0xf40e3585);
    RNDr(S, W, 47, 0x106aa070);
    RNDr(S, W, 48, 0x19a4c116);
    RNDr(S, W, 49, 0x1e376c08);
    RNDr(S, W, 50, 0x2748774c);
    RNDr(S, W, 51, 0x34b0bcb5);
    RNDr(S, W, 52, 0x391c0cb3);
    RNDr(S, W, 53, 0x4ed8aa4a);
    RNDr(S, W, 54, 0x5b9cca4f);
    RNDr(S, W, 55, 0x682e6ff3);
    RNDr(S, W, 56, 0x748f82ee);
    RNDr(S, W, 57, 0x78a5636f);
    RNDr(S, W, 58, 0x84c87814);
    RNDr(S, W, 59, 0x8cc70208);
    RNDr(S, W, 60, 0x90befffa);
    RNDr(S, W, 61, 0xa4506ceb);
    RNDr(S, W, 62, 0xbef9a3f7);
    RNDr(S, W, 63, 0xc67178f2);
    
    for (i = 0; i < 8; i++)
    {
        state[i] += S[i];
    }
    
    zeroize((void*)W, sizeof W);
    zeroize((void*)S, sizeof S);
    zeroize((void*)&t0, sizeof t0);
    zeroize((void*)&t1, sizeof t1);
}

void SHA256UpdateOpt1(SHA256CTX* context, const uint8_t* input, size_t length)
{
    uint32_t bitlen[2];
    uint32_t r = (context->count[1] >> 3) & 0x3f;
    
    bitlen[1] = ((uint32_t)length) << 3;
    bitlen[0] = (uint32_t)(length >> 29);
    
    if ((context->count[1] += bitlen[1]) < bitlen[1])
    {
        context->count[0]++;
    }
    
    context->count[0] += bitlen[0];
    
    if (length < 64 - r)
    {
        memcpy(&context->buf[r], input, length);
        return;
    }
    
    memcpy(&context->buf[r], input, 64 - r);
    SHA256TransformOpt1(context->state, context->buf);
    
    input += 64 - r;
    length -= 64 - r;
    
    while (length >= 64)
    {
        SHA256TransformOpt1(context->state, input);
        input += 64;
        length -= 64;
    }
    
    memcpy(context->buf, input, length);
}


void SHA256UpdateOptDouble(SHA256CTX* context, const uint8_t* input) {
    //precondition: length of input == 32
    
//    uint8_t data[SHA256_BLOCK_LENGTH] =
//    {
//        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//        0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
//    };
//    
//    memcpy(data, input, 32);
//    SHA256TransformOptDouble(context->state, data);


    memcpy(&context->buf[0], input, 32);

        uint8_t data[SHA256_BLOCK_LENGTH] =
        {
            0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
        };

    memcpy(&context->buf[32], data, 32);
    SHA256TransformOptDouble(context->state, context->buf);

}

void SHA256UpdateDualFixed32(SHA256CTX* context, uint8_t const* input1, uint8_t const* input2) {
    memcpy(&context->buf[0], input1, 32);
    memcpy(&context->buf[32], input2, 32);

    SHA256TransformOpt1(context->state, context->buf);
}


// ----------------------------------------------------------------
// ----------------------------------------------------------------
// SIMD - SSE3 (from Parallelizing message schedules to accelerate the computations of hash functions)
// ----------------------------------------------------------------
// ----------------------------------------------------------------

//
//// this function takes a word from each chunk, and puts it in a single register
//inline 
//__m128i gather(unsigned int* address) {
//    __m128i temp;
//    temp = _mm_cvtsi32_si128(address[0]);
//    temp = _mm_insert_epi32(temp, address[16], 1);
//    temp = _mm_insert_epi32(temp, address[32], 2);
//    temp = _mm_insert_epi32(temp, address[48], 3);
//    return temp;
//}
//
//// this function calculates the small sigma 0 transformation
//inline 
//__m128i sigma_0(__m128i W) {
//    return
//        _mm_xor_si128(
//            _mm_xor_si128(
//                _mm_xor_si128(
//                    _mm_srli_epi32(W, 7),
//                    _mm_srli_epi32(W, 18)
//                ),
//                _mm_xor_si128(
//                    _mm_srli_epi32(W, 3),
//                    _mm_slli_epi32(W, 25)
//                )
//            ),
//            _mm_slli_epi32(W, 14)
//        );
//}
//
//// this function calculates the small sigma 1 transformation
//inline 
//__m128i sigma_1(__m128i W) {
//    return
//        _mm_xor_si128(
//            _mm_xor_si128(
//                _mm_xor_si128(
//                    _mm_srli_epi32(W, 17),
//                    _mm_srli_epi32(W, 10)
//                ),
//                _mm_xor_si128(
//                    _mm_srli_epi32(W, 19),
//                    _mm_slli_epi32(W, 15)
//                )
//            ),
//            _mm_slli_epi32(W, 13)
//        );
//}
//
//// the message scheduling round
//#define SCHEDULE_ROUND(w1, w2, w3, w4) \
//    s0 = sigma_0(w1); \
//    s1 = sigma_1(w2); \
//    schedule[i] = _mm_add_epi32(w3, Ki[i]); \
//    w3 = _mm_add_epi32( \
//        _mm_add_epi32(w3, w4), \
//        _mm_add_epi32(s0, s1) \
//    ); \
//    i++;
//
//void SHA256_QMS(__m128i schedule[64], uint32_t message[64]) {
//    __m128i bswap_mask = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
//    __m128i W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15;
//    __m128i s0, s1, Wi, *Ki = (__m128i*)k;
//
//    W0 = gather(message);
//    W1 = gather(&message[1]);
//    W2 = gather(&message[2]);
//    W3 = gather(&message[3]);
//    W4 = gather(&message[4]);
//    W5 = gather(&message[5]);
//    W6 = gather(&message[6]);
//    W7 = gather(&message[7]);
//    W8 = gather(&message[8]);
//    W9 = gather(&message[9]);
//    W10 = gather(&message[10]);
//    W11 = gather(&message[11]);
//    W12 = gather(&message[12]);
//    W13 = gather(&message[13]);
//    W14 = gather(&message[14]);
//    W15 = gather(&message[15]);
//
//    W0  = _mm_shuffle_epi8(W0, bswap_mask);
//    W1  = _mm_shuffle_epi8(W1, bswap_mask);
//    W2  = _mm_shuffle_epi8(W2, bswap_mask);
//    W3  = _mm_shuffle_epi8(W3, bswap_mask);
//    W4  = _mm_shuffle_epi8(W4, bswap_mask);
//    W5  = _mm_shuffle_epi8(W5, bswap_mask);
//    W6  = _mm_shuffle_epi8(W6, bswap_mask);
//    W7  = _mm_shuffle_epi8(W7, bswap_mask);
//    W8  = _mm_shuffle_epi8(W8, bswap_mask);
//    W9  = _mm_shuffle_epi8(W9, bswap_mask);
//    W10 = _mm_shuffle_epi8(W10, bswap_mask);
//    W11 = _mm_shuffle_epi8(W11, bswap_mask);
//    W12 = _mm_shuffle_epi8(W12, bswap_mask);
//    W13 = _mm_shuffle_epi8(W13, bswap_mask);
//    W14 = _mm_shuffle_epi8(W14, bswap_mask);
//    W15 = _mm_shuffle_epi8(W15, bswap_mask);
//
//    int i;
//    for (i = 0; i<32; ) {
//        SCHEDULE_ROUND(W1 , W14, W0 , W9 );
//        SCHEDULE_ROUND(W2 , W15, W1 , W10);
//        SCHEDULE_ROUND(W3 , W0 , W2 , W11);
//        SCHEDULE_ROUND(W4 , W1 , W3 , W12);
//        SCHEDULE_ROUND(W5 , W2 , W4 , W13);
//        SCHEDULE_ROUND(W6 , W3 , W5 , W14);
//        SCHEDULE_ROUND(W7 , W4 , W6 , W15);
//        SCHEDULE_ROUND(W8 , W5 , W7 , W0 );
//        SCHEDULE_ROUND(W9 , W6 , W8 , W1 );
//        SCHEDULE_ROUND(W10, W7 , W9 , W2 );
//        SCHEDULE_ROUND(W11, W8 , W10, W3 );
//        SCHEDULE_ROUND(W12, W9 , W11, W4 );
//        SCHEDULE_ROUND(W13, W10, W12, W5 );
//        SCHEDULE_ROUND(W14, W11, W13, W6 );
//        SCHEDULE_ROUND(W15, W12, W14, W7 );
//        SCHEDULE_ROUND(W0 , W13, W15, W8 );
//    }
//
//    SCHEDULE_ROUND(W1 , W14, W0 , W9 );
//    schedule[48] = _mm_add_epi32(W0, Ki[48]);
//    SCHEDULE_ROUND(W2 , W15, W1 , W10);
//    schedule[49] = _mm_add_epi32(W1, Ki[49]);
//    SCHEDULE_ROUND(W3 , W0 , W2 , W11);
//    schedule[50] = _mm_add_epi32(W2, Ki[50]);
//    SCHEDULE_ROUND(W4 , W1 , W3 , W12);
//    schedule[51] = _mm_add_epi32(W3, Ki[51]);
//    SCHEDULE_ROUND(W5 , W2 , W4 , W13);
//    schedule[52] = _mm_add_epi32(W4, Ki[52]);
//    SCHEDULE_ROUND(W6 , W3 , W5 , W14);
//    schedule[53] = _mm_add_epi32(W5, Ki[53]);
//    SCHEDULE_ROUND(W7 , W4 , W6 , W15);
//    schedule[54] = _mm_add_epi32(W6, Ki[54]);
//    SCHEDULE_ROUND(W8 , W5 , W7 , W0 );
//    schedule[55] = _mm_add_epi32(W7, Ki[55]);
//    SCHEDULE_ROUND(W9 , W6 , W8 , W1 );
//    schedule[56] = _mm_add_epi32(W8, Ki[56]);
//    SCHEDULE_ROUND(W10, W7 , W9 , W2 );
//    schedule[57] = _mm_add_epi32(W9, Ki[57]);
//    SCHEDULE_ROUND(W11, W8 , W10, W3 );
//    schedule[58] = _mm_add_epi32(W10, Ki[58]);
//    SCHEDULE_ROUND(W12, W9 , W11, W4 );
//    schedule[59] = _mm_add_epi32(W11, Ki[59]);
//    SCHEDULE_ROUND(W13, W10, W12, W5 );
//    schedule[60] = _mm_add_epi32(W12, Ki[60]);
//    SCHEDULE_ROUND(W14, W11, W13, W6 );
//    schedule[61] = _mm_add_epi32(W13, Ki[61]);
//    SCHEDULE_ROUND(W15, W12, W14, W7 );
//    schedule[62] = _mm_add_epi32(W14, Ki[62]);
//    SCHEDULE_ROUND(W0 , W13, W15, W8 );
//    schedule[63] = _mm_add_epi32(W15, Ki[63]);
//}
//



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






// --------------------------------------------------------------------------

//
//void check_fer(uint32_t a[64], uint32_t S[8]) {
//
//	for (size_t i = 0; i < 8; i++)
//	{
//		for (size_t j = 0; j < 64; j++)
//		{
//			if (a[j] == S[i]) {
//				printf("i: %d, j:%d, value: %#010x\n", i, j, a[j]);
//			}
//		}
//
//	}
//
//}

//
//
//
//
//uint32_t temp[64] = { 0x428a2f98,
//0x72374491,
//0xb7c0fbcf,
//0xecb5dba5,
//0x3d56c25b,
//0x5ef111f1,
//0x983f82a4,
//0xb21c5ed5,
//0xe007aa98,
//0x1b835b01,
//0x2e3185be,
//0x600c7dc3,
//0x7ebe5d74,
//0x8ddeb1fe,
//0xa9dc06a7,
//0xd09bf174,
//0xedc0f0c1,
//0xfb060e66,
//0x9089eb5d,
//0x4f437806,
//0x6eec301d,
//0x10112372,
//0x545153fd,
//0x64bb2f34,
//0xed214780,
//0x1b2b71b5,
//0x0c77d912,
//0xedc60fbe,
//0x68d22693,
//0xb23328c0,
//0x52b645ac,
//0x9268106d,
//0x6210e8bd,
//0x71d191de,
//0x95f51540,
//0xab01f850,
//0x787c4952,
//0xc872a93d,
//0x89db7eef,
//0x02d1a08a,
//0x7c1ccd18,
//0x6bdfa326,
//0x2d79ec2d,
//0x16e6af8a,
//0xcfd78788,
//0x68165d98,
//0x47b95c53,
//0x6a28789e,
//0xe84617aa,
//0xcc7a5d9e,
//0x4bd0ce8b,
//0x3cc8b77e,
//0x28c22554,
//0xa52f7666,
//0x66dc885e,
//0x344c41ef,
//0x05bde125,
//0xa382ba9f,
//0x2d10b198,
//0xb54c0532,
//0xb003b2c7,
//0xc26998fe,
//0xcf8af101,
//0xcad675b2 };
