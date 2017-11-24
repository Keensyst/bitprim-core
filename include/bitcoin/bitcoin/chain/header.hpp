/**
 * Copyright (c) 2011-2017 libbitcoin developers (see AUTHORS)
 *
 * This file is part of libbitcoin.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef LIBBITCOIN_CHAIN_HEADER_HPP_
#define LIBBITCOIN_CHAIN_HEADER_HPP_

#include <cstddef>
#include <cstdint>
#include <istream>
#include <string>
#include <memory>
#include <vector>
#include <bitcoin/bitcoin/chain/chain_state.hpp>
#include <bitcoin/bitcoin/define.hpp>
#include <bitcoin/bitcoin/error.hpp>
#include <bitcoin/bitcoin/math/hash.hpp>
#include <bitcoin/bitcoin/utility/data.hpp>
#include <bitcoin/bitcoin/utility/reader.hpp>
#include <bitcoin/bitcoin/utility/thread.hpp>
#include <bitcoin/bitcoin/utility/writer.hpp>

namespace libbitcoin { namespace chain {

class BC_API header_raw {
public:
    using list = std::vector<header_raw>;
    using ptr = std::shared_ptr<header_raw>;
    using const_ptr = std::shared_ptr<header_raw const>;
    using ptr_list = std::vector<header_raw>;
    using const_ptr_list = std::vector<const_ptr>;

    // Constructors.
    //-----------------------------------------------------------------------------
    header_raw();
    header_raw(uint32_t version, hash_digest const& previous_block_hash, hash_digest const& merkle, uint32_t timestamp, uint32_t bits, uint32_t nonce);
    header_raw(header_raw const& other);
    //TODO(fernando): check how the move ctor is generated by the compiler

    // Operators.
    //-----------------------------------------------------------------------------

    /// This class is move and copy assignable.
    header_raw& operator=(header_raw const& other);

    friend
    bool operator==(header_raw const&, header_raw const&);
    friend
    bool operator!=(header_raw const&, header_raw const&);

    //-----------------------------------------------------------------------------
    bool valid() const;

    //-----------------------------------------------------------------------------

    uint32_t version() const;
    void set_version(uint32_t value);

    // Deprecated (unsafe).
    // hash_digest& previous_block_hash();
    hash_digest const& previous_block_hash() const;
    void set_previous_block_hash(hash_digest const& value);

    // Deprecated (unsafe).
    // hash_digest& merkle();
    hash_digest const& merkle() const;
    void set_merkle(hash_digest const& value);

    uint32_t timestamp() const;
    void set_timestamp(uint32_t value);

    uint32_t bits() const;
    void set_bits(uint32_t value);

    uint32_t nonce() const;
    void set_nonce(uint32_t value);

    // Deserialization.
    //-----------------------------------------------------------------------------
    static 
    boost::optional<header_raw> factory_from_data(const data_chunk& data, bool wire = true);

    static 
    boost::optional<header_raw> factory_from_data(std::istream& stream, bool wire = true);

    static 
    boost::optional<header_raw> factory_from_data(reader& source, bool wire = true);


private:
    //TODO(fernando):  check object alignment

    uint32_t version_;
    hash_digest previous_block_hash_;
    hash_digest merkle_;
    uint32_t timestamp_;
    uint32_t bits_;
    uint32_t nonce_;
};


class BC_API header {
public:
    using list = std::vector<header>;
    using ptr = std::shared_ptr<header>;
    using const_ptr = std::shared_ptr<header const>;
    using ptr_list = std::vector<header>;
    using const_ptr_list = std::vector<const_ptr>;

    // THIS IS FOR LIBRARY USE ONLY, DO NOT CREATE A DEPENDENCY ON IT.
    struct validation {
        size_t height = 0;
        uint32_t median_time_past = 0;
    };

    // Constructors.
    //-----------------------------------------------------------------------------

    header();
    header(uint32_t version, hash_digest const& previous_block_hash, hash_digest const& merkle, uint32_t timestamp, uint32_t bits, uint32_t nonce);
    // header(header const& other, hash_digest const& hash);
    header(header const& other);
    //TODO(fernando): check how the move ctor is generated by the compiler

    // Operators.
    //-----------------------------------------------------------------------------

    /// This class is move and copy assignable.
    header& operator=(header const& other);

    friend
    bool operator==(header const&, header const&);
    friend
    bool operator!=(header const&, header const&);

    bool is_valid() const;

    // Serialization.
    //-----------------------------------------------------------------------------

    data_chunk to_data(bool wire = true) const;
    void to_data(std::ostream& stream, bool wire = true) const;
    void to_data(writer& sink, bool wire = true) const;

    // Properties (size, accessors, cache).
    //-----------------------------------------------------------------------------

    static size_t satoshi_fixed_size();
    size_t serialized_size(bool wire = true) const;

    uint32_t version() const;
    // void set_version(uint32_t value);

    // Deprecated (unsafe).
    hash_digest& previous_block_hash();

    hash_digest const& previous_block_hash() const;
    // void set_previous_block_hash(hash_digest const& value);

    // Deprecated (unsafe).
    hash_digest& merkle();

    hash_digest const& merkle() const;
    // void set_merkle(hash_digest const& value);

    uint32_t timestamp() const;
    // void set_timestamp(uint32_t value);

    uint32_t bits() const;
    // void set_bits(uint32_t value);

    uint32_t nonce() const;
    // void set_nonce(uint32_t value);

    hash_digest hash() const;
#ifdef LITECOIN //TODO(fernando): rename this macro
    hash_digest litecoin_proof_of_work_hash() const;
#endif

    // Validation.
    //-----------------------------------------------------------------------------

    bool is_valid_timestamp() const;
    bool is_valid_proof_of_work() const;

    code check() const;
    code accept(const chain_state& state) const;

    // THIS IS FOR LIBRARY USE ONLY, DO NOT CREATE A DEPENDENCY ON IT.
    mutable validation validation;



    // Deserialization.
    //-----------------------------------------------------------------------------

    static 
    boost::optional<header> factory_from_data(const data_chunk& data, bool wire = true);

    static 
    boost::optional<header> factory_from_data(std::istream& stream, bool wire = true);

    static 
    boost::optional<header> factory_from_data(reader& source, bool wire = true);

    // bool from_data(const data_chunk& data, bool wire = true);
    // bool from_data(std::istream& stream, bool wire = true);
    // bool from_data(reader& source, bool wire = true);

protected:
    // So that block may call reset from its own.
    friend class block;

    // void reset();
    // void invalidate_cache() const;

private:
    //TODO(fernando):  check object alignment
    //TODO(fernando):  why mutex here?

    mutable upgrade_mutex mutex_;
    mutable std::shared_ptr<hash_digest> hash_;

    uint32_t version_;
    hash_digest previous_block_hash_;
    hash_digest merkle_;
    uint32_t timestamp_;
    uint32_t bits_;
    uint32_t nonce_;
};




// Serialization.
//-----------------------------------------------------------------------------

data_chunk to_data(header_raw const& header, bool wire = true) const;
void to_data(header_raw const& header, std::ostream& stream, bool wire = true) const;
void to_data(header_raw const& header, writer& sink, bool wire = true) const;

// Properties (size, accessors, cache).
//-----------------------------------------------------------------------------
static size_t satoshi_fixed_size(header_raw const& header);
size_t serialized_size(header_raw const& header, bool wire = true) const;

hash_digest hash(header_raw const& header) const;
#ifdef LITECOIN //TODO(fernando): rename this macro
hash_digest litecoin_proof_of_work_hash(header_raw const& header) const;
#endif


// Validation.
//-----------------------------------------------------------------------------

bool is_valid_timestamp(header_raw const& header) const;
bool is_valid_proof_of_work(header_raw const& header) const;

code check(header_raw const& header) const;
code accept(header_raw const& header, const chain_state& state) const;















// class BC_API header {
// public:
//     using list = std::vector<header>;
//     using ptr = std::shared_ptr<header>;
//     using const_ptr = std::shared_ptr<header const>;
//     using ptr_list = std::vector<header>;
//     using const_ptr_list = std::vector<const_ptr>;

//     // THIS IS FOR LIBRARY USE ONLY, DO NOT CREATE A DEPENDENCY ON IT.
//     struct validation {
//         size_t height = 0;
//         uint32_t median_time_past = 0;
//     };

//     // Constructors.
//     //-----------------------------------------------------------------------------

//     header();
//     header(uint32_t version, hash_digest const& previous_block_hash, hash_digest const& merkle, uint32_t timestamp, uint32_t bits, uint32_t nonce);
//     // header(header const& other, hash_digest const& hash);
//     header(header const& other);
//     //TODO(fernando): check how the move ctor is generated by the compiler

//     // Operators.
//     //-----------------------------------------------------------------------------

//     /// This class is move and copy assignable.
//     header& operator=(header const& other);

//     friend
//     bool operator==(header const&, header const&);
//     friend
//     bool operator!=(header const&, header const&);

//     bool is_valid() const;

//     // Serialization.
//     //-----------------------------------------------------------------------------

//     data_chunk to_data(bool wire = true) const;
//     void to_data(std::ostream& stream, bool wire = true) const;
//     void to_data(writer& sink, bool wire = true) const;

//     // Properties (size, accessors, cache).
//     //-----------------------------------------------------------------------------

//     static size_t satoshi_fixed_size();
//     size_t serialized_size(bool wire = true) const;

//     uint32_t version() const;
//     // void set_version(uint32_t value);

//     // Deprecated (unsafe).
//     hash_digest& previous_block_hash();

//     hash_digest const& previous_block_hash() const;
//     // void set_previous_block_hash(hash_digest const& value);

//     // Deprecated (unsafe).
//     hash_digest& merkle();

//     hash_digest const& merkle() const;
//     // void set_merkle(hash_digest const& value);

//     uint32_t timestamp() const;
//     // void set_timestamp(uint32_t value);

//     uint32_t bits() const;
//     // void set_bits(uint32_t value);

//     uint32_t nonce() const;
//     // void set_nonce(uint32_t value);

//     hash_digest hash() const;
// #ifdef LITECOIN //TODO(fernando): rename this macro
//     hash_digest litecoin_proof_of_work_hash() const;
// #endif

//     // Validation.
//     //-----------------------------------------------------------------------------

//     bool is_valid_timestamp() const;
//     bool is_valid_proof_of_work() const;

//     code check() const;
//     code accept(const chain_state& state) const;

//     // THIS IS FOR LIBRARY USE ONLY, DO NOT CREATE A DEPENDENCY ON IT.
//     mutable validation validation;



//     // Deserialization.
//     //-----------------------------------------------------------------------------

//     static 
//     boost::optional<header> factory_from_data(const data_chunk& data, bool wire = true);

//     static 
//     boost::optional<header> factory_from_data(std::istream& stream, bool wire = true);

//     static 
//     boost::optional<header> factory_from_data(reader& source, bool wire = true);

//     // bool from_data(const data_chunk& data, bool wire = true);
//     // bool from_data(std::istream& stream, bool wire = true);
//     // bool from_data(reader& source, bool wire = true);

// protected:
//     // So that block may call reset from its own.
//     friend class block;

//     // void reset();
//     // void invalidate_cache() const;

// private:
//     //TODO(fernando):  check object alignment
//     //TODO(fernando):  why mutex here?

//     mutable upgrade_mutex mutex_;
//     mutable std::shared_ptr<hash_digest> hash_;

//     uint32_t version_;
//     hash_digest previous_block_hash_;
//     hash_digest merkle_;
//     uint32_t timestamp_;
//     uint32_t bits_;
//     uint32_t nonce_;
// };

}} // namespace libbitcoin::chain

#endif /*LIBBITCOIN_CHAIN_HEADER_HPP_*/
