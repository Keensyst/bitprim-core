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
#ifndef LIBBITCOIN_CHAIN_TRANSACTION_HPP_
#define LIBBITCOIN_CHAIN_TRANSACTION_HPP_

#include <cstddef>
#include <cstdint>
#include <istream>
#include <memory>
#include <string>
#include <vector>
#include <boost/optional.hpp>
#include <bitcoin/bitcoin/chain/chain_state.hpp>
#include <bitcoin/bitcoin/chain/input.hpp>
#include <bitcoin/bitcoin/chain/output.hpp>
#include <bitcoin/bitcoin/chain/point.hpp>
#include <bitcoin/bitcoin/define.hpp>
#include <bitcoin/bitcoin/error.hpp>
#include <bitcoin/bitcoin/math/elliptic_curve.hpp>
#include <bitcoin/bitcoin/math/hash.hpp>
#include <bitcoin/bitcoin/machine/opcode.hpp>
#include <bitcoin/bitcoin/machine/rule_fork.hpp>
#include <bitcoin/bitcoin/utility/reader.hpp>
#include <bitcoin/bitcoin/utility/thread.hpp>
#include <bitcoin/bitcoin/utility/writer.hpp>

namespace libbitcoin { namespace chain {

class BC_API transaction_raw {
public:
    using tx_opt = boost::optional<transaction_raw>;
    using ins = input::list;
    using outs = output::list;
    using list = std::vector<transaction_raw>;

    // Constructors.
    //-----------------------------------------------------------------------------

    transaction_raw();

    // transaction_raw(uint32_t version, uint32_t locktime, ins&& inputs, outs&& outputs);
    // transaction_raw(uint32_t version, uint32_t locktime, ins const& inputs, outs const& outputs);

    template <typename Ins, typename Outs>
    //  requires SameType<Ins, ins> && SameType<Outs, outs>
    transaction_raw(uint32_t version, uint32_t locktime, Ins&& inputs, Outs&& outputs);

    transaction_raw(transaction_raw const& other);
    transaction_raw(transaction_raw&& other);
    /// This class is move assignable and copy assignable [TODO: remove copy].
    transaction_raw& operator=(transaction_raw const& other);
    transaction_raw& operator=(transaction_raw&& other);


    // Operators.
    //-----------------------------------------------------------------------------

    friend
    bool operator==(transaction_raw const& x, transaction_raw const& y);
    friend
    bool operator!=(transaction_raw const& x, transaction_raw const& y);


    //-----------------------------------------------------------------------------

    bool valid() const;

    // Accessors
    //-----------------------------------------------------------------------------

    uint32_t version() const;
    void set_version(uint32_t value);

    uint32_t locktime() const;
    void set_locktime(uint32_t value);

    // Deprecated (unsafe).
    ins& inputs();

    const ins& inputs() const;
    void set_inputs(const ins& value);
    void set_inputs(ins&& value);

    // Deprecated (unsafe).
    outs& outputs();

    const outs& outputs() const;
    void set_outputs(const outs& value);
    void set_outputs(outs&& value);

    // Deserialization.
    //-----------------------------------------------------------------------------

    static 
    tx_opt factory_from_data(data_chunk const& data, bool wire = true);
    static 
    tx_opt factory_from_data(std::istream& stream, bool wire = true);
    static 
    tx_opt factory_from_data(reader& source, bool wire = true);

// protected:
//     void reset();

private:
    uint32_t version_;
    uint32_t locktime_;
    input::list inputs_;
    output::list outputs_;
    //TODO(fernando): Segwit
};

// Immutable Transaction!
class BC_API transaction {
public:
    using ins = input::list;
    using outs = output::list;
    using list = std::vector<transaction>;
    using tx_opt = boost::optional<transaction>;

    //TODO(fernando): Remove this validation, breaks regularity
    // THIS IS FOR LIBRARY USE ONLY, DO NOT CREATE A DEPENDENCY ON IT.
    struct validation {
        uint64_t originator = 0;
        code error = error::not_found;
        chain_state::ptr state = nullptr;

        // The transaction is an unspent duplicate.
        bool duplicate = false;

        // The unconfirmed tx exists in the store.
        bool pooled = false;

        // The unconfirmed tx is validated at the block's current fork state.
        bool current = false;

        // Similate organization and instead just validate the transaction.
        bool simulate = false;
    };

    // Constructors.
    //-----------------------------------------------------------------------------

    transaction();

    transaction(transaction_raw const& other);
    transaction(transaction_raw&& other);

    // transaction(transaction_raw const& other, hash_digest const& hash);
    // transaction(transaction_raw&& other, hash_digest const& hash);

    template <typename Ins, typename Outs>
    //  requires SameType<Ins, ins> && SameType<Outs, outs>
    transaction(uint32_t version, uint32_t locktime, Ins&& inputs, Outs&& outputs);

    // transaction(uint32_t version, uint32_t locktime, ins&& inputs, outs&& outputs);
    // transaction(uint32_t version, uint32_t locktime, const ins& inputs, const outs& outputs);

    transaction(transaction const& other);
    transaction(transaction&& other);


    // Operators.
    //-----------------------------------------------------------------------------

    /// This class is move assignable and copy assignable [TODO: remove copy].
    transaction& operator=(transaction const& other);
    transaction& operator=(transaction&& other);

    friend
    bool operator==(transaction const& x, transaction const& y);
    friend
    bool operator!=(transaction const& x, transaction const& y);

    // explicit
    operator transaction_raw() const;

    //-----------------------------------------------------------------------------
    bool valid() const;

    // Accessors.
    //-----------------------------------------------------------------------------
    uint32_t version() const;
    uint32_t locktime() const;
    ins const& inputs() const;
    outs const& outputs() const;
    hash_digest hash() const;
    // hash_digest hash(uint32_t sighash_type) const;
    transaction_raw const& raw() const;

    // Deserialization.
    //-----------------------------------------------------------------------------
    static 
    tx_opt factory_from_data(data_chunk const& data, bool wire = true);
    static 
    tx_opt factory_from_data(std::istream& stream, bool wire = true);
    static 
    tx_opt factory_from_data(reader& source, bool wire = true);

    // Validation.
    //-----------------------------------------------------------------------------
    // THIS IS FOR LIBRARY USE ONLY, DO NOT CREATE A DEPENDENCY ON IT.
    mutable validation validation;

// protected:
//     void reset();

private:
    //TODO(fernando): check alignment
    transaction_raw tx_raw_;
    hash_digest hash_;
};


// Computations.
//-----------------------------------------------------------------------------
hash_digest hash_compute(transaction_raw const& tx);
hash_digest hash_compute(transaction_raw const& tx, uint32_t sighash_type);

// Serialization.
//-----------------------------------------------------------------------------
data_chunk to_data(transaction_raw const& tx, bool wire = true);
void to_data(transaction_raw const& tx, std::ostream& stream, bool wire = true);
void to_data(transaction_raw const& tx, writer& sink, bool wire = true);

// Properties (size, accessors, cache).
//-----------------------------------------------------------------------------
size_t serialized_size(transaction_raw const& tx, bool wire = true);


// Validation.
//-----------------------------------------------------------------------------
uint64_t fees(transaction_raw const& tx);
point::list previous_outputs(transaction_raw const& tx);
point::list missing_previous_outputs(transaction_raw const& tx);
hash_list missing_previous_transactions(transaction_raw const& tx);
uint64_t total_input_value(transaction_raw const& tx);
uint64_t total_output_value(transaction_raw const& tx);
size_t signature_operations(transaction_raw const& tx, chain_state::ptr const& state);
size_t signature_operations(transaction_raw const& tx, bool bip16_active);

bool is_coinbase(transaction_raw const& tx);
bool is_null_non_coinbase(transaction_raw const& tx);
bool is_oversized_coinbase(transaction_raw const& tx);
bool is_mature(transaction_raw const& tx, size_t height);
bool is_overspent(transaction_raw const& tx);
bool is_internal_double_spend(transaction_raw const& tx);
bool is_double_spend(transaction_raw const& tx, bool include_unconfirmed);
bool is_dusty(transaction_raw const& tx, uint64_t minimum_output_value);
bool is_missing_previous_outputs(transaction_raw const& tx);
bool is_final(transaction_raw const& tx, size_t block_height, uint32_t block_time);
bool is_locked(transaction_raw const& tx, size_t block_height, uint32_t median_time_past);
bool is_locktime_conflict(transaction_raw const& tx);

code check(transaction_raw const& tx, bool transaction_pool = true);

// code accept(transaction_raw const& tx, bool transaction_pool = true);
code accept(transaction_raw const& tx, chain_state const& state, bool duplicate, bool transaction_pool = true);

// code connect(transaction_raw const& tx);
code connect(transaction_raw const& tx, chain_state const& state);
code connect_input(transaction_raw const& tx, chain_state const& state, size_t input_index);


}}  // namespace libbitcoin::chain

#endif /*LIBBITCOIN_CHAIN_TRANSACTION_HPP_*/



// // Immutable Transaction!
// class BC_API transaction {
// public:
//     using ins = input::list;
//     using outs = output::list;
//     using list = std::vector<transaction>;

//     // THIS IS FOR LIBRARY USE ONLY, DO NOT CREATE A DEPENDENCY ON IT.
//     struct validation {
//         uint64_t originator = 0;
//         code error = error::not_found;
//         chain_state::ptr state = nullptr;

//         // The transaction is an unspent duplicate.
//         bool duplicate = false;

//         // The unconfirmed tx exists in the store.
//         bool pooled = false;

//         // The unconfirmed tx is validated at the block's current fork state.
//         bool current = false;

//         // Similate organization and instead just validate the transaction.
//         bool simulate = false;
//     };

//     // Constructors.
//     //-----------------------------------------------------------------------------

//     transaction();

//     transaction(transaction const& other);
//     transaction(transaction&& other);

//     transaction(transaction const& other, hash_digest const& hash);
//     transaction(transaction&& other, hash_digest const& hash);

//     transaction(uint32_t version, uint32_t locktime, ins&& inputs, outs&& outputs);
//     transaction(uint32_t version, uint32_t locktime, const ins& inputs, const outs& outputs);

//     // Operators.
//     //-----------------------------------------------------------------------------

//     /// This class is move assignable and copy assignable [TODO: remove copy].
//     transaction& operator=(transaction const& other);
//     transaction& operator=(transaction&& other);

//     bool operator==(transaction const& other) const;
//     bool operator!=(transaction const& other) const;

//     // Deserialization.
//     //-----------------------------------------------------------------------------

//     static transaction factory_from_data(data_chunk const& data, bool wire = true);
//     static transaction factory_from_data(std::istream& stream, bool wire = true);
//     static transaction factory_from_data(reader& source, bool wire = true);

//     bool from_data(data_chunk const& data, bool wire = true);
//     bool from_data(std::istream& stream, bool wire = true);
//     bool from_data(reader& source, bool wire = true);

//     bool is_valid() const;

//     // Serialization.
//     //-----------------------------------------------------------------------------

//     data_chunk to_data(bool wire = true) const;
//     void to_data(std::ostream& stream, bool wire = true) const;
//     void to_data(writer& sink, bool wire = true) const;

//     // Properties (size, accessors, cache).
//     //-----------------------------------------------------------------------------

//     size_t serialized_size(bool wire = true) const;

//     uint32_t version() const;
//     void set_version(uint32_t value);

//     uint32_t locktime() const;
//     void set_locktime(uint32_t value);

//     // Deprecated (unsafe).
//     ins& inputs();

//     const ins& inputs() const;
//     void set_inputs(const ins& value);
//     void set_inputs(ins&& value);

//     // Deprecated (unsafe).
//     outs& outputs();

//     const outs& outputs() const;
//     void set_outputs(const outs& value);
//     void set_outputs(outs&& value);

//     hash_digest hash() const;
//     hash_digest hash(uint32_t sighash_type) const;

//     void recompute_hash();

//     // Validation.
//     //-----------------------------------------------------------------------------

//     uint64_t fees() const;
//     point::list previous_outputs() const;
//     point::list missing_previous_outputs() const;
//     hash_list missing_previous_transactions() const;
//     uint64_t total_input_value() const;
//     uint64_t total_output_value() const;
//     size_t signature_operations() const;
//     size_t signature_operations(bool bip16_active) const;

//     bool is_coinbase() const;
//     bool is_null_non_coinbase() const;
//     bool is_oversized_coinbase() const;
//     bool is_mature(size_t height) const;
//     bool is_overspent() const;
//     bool is_internal_double_spend() const;
//     bool is_double_spend(bool include_unconfirmed) const;
//     bool is_dusty(uint64_t minimum_output_value) const;
//     bool is_missing_previous_outputs() const;
//     bool is_final(size_t block_height, uint32_t block_time) const;
//     bool is_locked(size_t block_height, uint32_t median_time_past) const;
//     bool is_locktime_conflict() const;

//     code check(bool transaction_pool=true) const;
//     code accept(bool transaction_pool=true) const;
//     code accept(chain_state const& state, bool transaction_pool=true) const;
//     code connect() const;
//     code connect(chain_state const& state) const;
//     code connect_input(chain_state const& state, size_t input_index) const;

//     // THIS IS FOR LIBRARY USE ONLY, DO NOT CREATE A DEPENDENCY ON IT.
//     mutable validation validation;

// protected:
//     void reset();
//     void invalidate_cache() const;
//     bool all_inputs_final() const;

// private:
//     uint32_t version_;
//     uint32_t locktime_;
//     input::list inputs_;
//     output::list outputs_;

//     // These share a mutex as they are not expected to conflict.
//     mutable boost::optional<uint64_t> total_input_value_;
//     mutable boost::optional<uint64_t> total_output_value_;
//     mutable std::shared_ptr<hash_digest> hash_;
//     mutable upgrade_mutex mutex_;
// };
