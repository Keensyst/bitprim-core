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
#include <bitcoin/bitcoin/chain/transaction.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <sstream>
#include <utility>
#include <vector>
#include <boost/optional.hpp>
#include <bitcoin/bitcoin/chain/chain_state.hpp>
#include <bitcoin/bitcoin/chain/input.hpp>
#include <bitcoin/bitcoin/chain/output.hpp>
#include <bitcoin/bitcoin/chain/script.hpp>
#include <bitcoin/bitcoin/constants.hpp>
#include <bitcoin/bitcoin/error.hpp>
#include <bitcoin/bitcoin/math/limits.hpp>
#include <bitcoin/bitcoin/machine/opcode.hpp>
#include <bitcoin/bitcoin/machine/operation.hpp>
#include <bitcoin/bitcoin/machine/rule_fork.hpp>
#include <bitcoin/bitcoin/message/messages.hpp>
#include <bitcoin/bitcoin/utility/collection.hpp>
#include <bitcoin/bitcoin/utility/container_sink.hpp>
#include <bitcoin/bitcoin/utility/container_source.hpp>
#include <bitcoin/bitcoin/utility/endian.hpp>
#include <bitcoin/bitcoin/utility/istream_reader.hpp>
#include <bitcoin/bitcoin/utility/ostream_writer.hpp>

#include <bitcoin/bitcoin/bitcoin_cash_support.hpp>

namespace libbitcoin {
namespace chain {

using namespace bc::machine;

// Read a length-prefixed collection of inputs or outputs from the source.
template <typename Source, typename Put>
bool read(Source& source, std::vector<Put>& puts, bool wire) {
    auto result = true;
    auto const count = source.read_size_little_endian();

    // Guard against potential for arbitary memory allocation.
    if (count > get_max_block_size(is_bitcoin_cash())) {
        source.invalidate();
    } else {
        puts.resize(count);
    }

    auto const deserialize = [&result, &source, wire](Put& put) {
        result = result && put.from_data(source, wire);
    };

    std::for_each(puts.begin(), puts.end(), deserialize);
    return result;
}

// Write a length-prefixed collection of inputs or outputs to the sink.
template <typename Sink, typename Put>
void write(Sink& sink, const std::vector<Put>& puts, bool wire) {
    sink.write_variable_little_endian(puts.size());

    auto const serialize = [&sink, wire](const Put& put) {
        put.to_data(sink, wire);
    };

    std::for_each(puts.begin(), puts.end(), serialize);
}

// Constructors.
//-----------------------------------------------------------------------------

transaction_raw::transaction_raw()
  : version_(0), locktime_(0)
{}

// transaction_raw::transaction_raw(uint32_t version, uint32_t locktime, input::list const& inputs, output::list const& outputs)
//     : version_(version)
//     , locktime_(locktime)
//     , inputs_(inputs)
//     , outputs_(outputs)
// {}

// transaction_raw::transaction_raw(uint32_t version, uint32_t locktime, input::list&& inputs, output::list&& outputs)
//     : version_(version)
//     , locktime_(locktime)
//     , inputs_(std::move(inputs))
//     , outputs_(std::move(outputs))
// {}

template <typename Ins, typename Outs>
transaction_raw::transaction_raw(uint32_t version, uint32_t locktime, Ins&& inputs, Outs&& outputs)
    : version_(version)
    , locktime_(locktime)
    , inputs_(std::forward<Ins>(inputs))
    , outputs_(std::forward<Outs>(outputs))
{}

transaction_raw::transaction_raw(transaction_raw const& other)
    : transaction_raw(other.version_, other.locktime_, other.inputs_, other.outputs_)
{}

transaction_raw::transaction_raw(transaction_raw&& other)
    : transaction_raw(other.version_, other.locktime_, std::move(other.inputs_), std::move(other.outputs_))
{}

// TODO(libbitcoin): eliminate blockchain transaction_raw copies and then delete this.
transaction_raw& transaction_raw::operator=(transaction_raw const& other) {
    version_ = other.version_;
    locktime_ = other.locktime_;
    inputs_ = other.inputs_;
    outputs_ = other.outputs_;
    return *this;
}

transaction_raw& transaction_raw::operator=(transaction_raw&& other) {
    // TODO: implement safe private accessor for conditional cache transfer.
    version_ = other.version_;
    locktime_ = other.locktime_;
    inputs_ = std::move(other.inputs_);
    outputs_ = std::move(other.outputs_);
    return *this;
}

// TODO(fernando): implement swap

// Operators.
//-----------------------------------------------------------------------------

bool operator==(transaction_raw const& x, transaction_raw const& y) {
    return (x.version_ == y.version_)
        && (x.locktime_ == y.locktime_)
        && (x.inputs_ == y.inputs_)
        && (x.outputs_ == y.outputs_);
}

bool operator!=(transaction_raw const& x, transaction_raw const& y) {
    return !(x == y);
}

//-----------------------------------------------------------------------------

bool transaction_raw::valid() const {
    return version_ != 0 || 
           locktime_ != 0 || 
           ! inputs_.empty() ||
           ! outputs_.empty();
}

// Accessors.
//-----------------------------------------------------------------------------

uint32_t transaction_raw::version() const {
    return version_;
}

void transaction_raw::set_version(uint32_t value) {
    version_ = value;
}

uint32_t transaction_raw::locktime() const {
    return locktime_;
}

void transaction_raw::set_locktime(uint32_t value) {
    locktime_ = value;
}

input::list& transaction_raw::inputs() {
    return inputs_;
}

input::list const& transaction_raw::inputs() const {
    return inputs_;
}

void transaction_raw::set_inputs(input::list const& value) {
    inputs_ = value;
}

void transaction_raw::set_inputs(input::list&& value) {
    inputs_ = std::move(value);
}

output::list& transaction_raw::outputs() {
    return outputs_;
}

output::list const& transaction_raw::outputs() const {
    return outputs_;
}

void transaction_raw::set_outputs(output::list const& value) {
    outputs_ = value;
}

void transaction_raw::set_outputs(output::list&& value) {
    outputs_ = std::move(value);
}

// Deserialization.
//-----------------------------------------------------------------------------

// static
transaction_raw::tx_opt factory_from_data_wire(reader& source) {
    //TODO(fernando): Implement Segwit

    // Wire (satoshi protocol) deserialization.
    auto version = source.read_4_bytes_little_endian();

    input::list inputs;
    output::list outputs;
    read(source, inputs, true) && read(source, outputs, true);

    auto locktime = source.read_4_bytes_little_endian();

    if ( ! source) return {};

    return transaction_raw(version, locktime, std::move(inputs), std::move(outputs));
}

// static
transaction_raw::tx_opt factory_from_data_not_wire(reader& source) {
    // Database (outputs forward) serialization.

    input::list inputs;
    output::list outputs;

    read(source, outputs, false) && read(source, inputs, false);

    auto locktime = source.read_variable_little_endian();
    auto version = source.read_variable_little_endian();

    if (locktime > max_uint32 || version > max_uint32) {
        source.invalidate();
    }

    if ( ! source) return {};

    return transaction_raw(static_cast<uint32_t>(version), static_cast<uint32_t>(locktime), std::move(inputs), std::move(outputs));
}

// static
inline
transaction_raw::tx_opt transaction_raw::factory_from_data(reader& source, bool wire) {
    return wire ? factory_from_data_wire(source) : factory_from_data_not_wire(source);
}

// static
inline
transaction_raw::tx_opt transaction_raw::factory_from_data(data_chunk const& data, bool wire) {
    data_source source(data);
    return factory_from_data(source, wire);
}

// static
inline
transaction_raw::tx_opt transaction_raw::factory_from_data(std::istream& stream, bool wire) {
    istream_reader reader(stream);
    return factory_from_data(reader, wire);
}

// // protected
// void transaction_raw::reset() {
//     version_ = 0;
//     locktime_ = 0;
//     inputs_.clear();
//     inputs_.shrink_to_fit();
//     outputs_.clear();
//     outputs_.shrink_to_fit();
//     invalidate_cache();
// }

// Size.
//-----------------------------------------------------------------------------

size_t serialized_size(transaction_raw const& tx, bool wire) {
    auto const ins = [wire](size_t size, input const& input) {
        return size + input.serialized_size(wire);
    };

    auto const outs = [wire](size_t size, const output& output) {
        return size + output.serialized_size(wire);
    };

    return (wire ? sizeof(tx.version()) : message::variable_uint_size(tx.version()))
        + (wire ? sizeof(tx.locktime()) : message::variable_uint_size(tx.locktime()))
        + message::variable_uint_size(tx.inputs().size())
        + message::variable_uint_size(tx.outputs().size())
        + std::accumulate(tx.inputs().begin(), tx.inputs().end(), size_t{0}, ins)
        + std::accumulate(tx.outputs().begin(), tx.outputs().end(), size_t{0}, outs);
}

// Serialization.
//-----------------------------------------------------------------------------

data_chunk to_data(transaction_raw const& tx, bool wire) {
    data_chunk data;
    auto const size = serialized_size(tx, wire);

    // Reserve an extra byte to prevent full reallocation in the case of
    // generate_signature_hash extension by addition of the sighash_type.
    data.reserve(size + sizeof(uint8_t));

    data_sink ostream(data);
    to_data(tx, ostream, wire);
    ostream.flush();
    BITCOIN_ASSERT(data.size() == size);
    return data;
}

void to_data(transaction_raw const& tx, std::ostream& stream, bool wire) {
    ostream_writer sink(stream);
    to_data(tx, sink, wire);
}

void to_data(transaction_raw const& tx, writer& sink, bool wire) {
    if (wire) {
        // Wire (satoshi protocol) serialization.
        sink.write_4_bytes_little_endian(tx.version());
        write(sink, tx.inputs(), wire);
        write(sink, tx.outputs(), wire);
        sink.write_4_bytes_little_endian(tx.locktime());
    } else {
        // Database (outputs forward) serialization.
        write(sink, tx.outputs(), wire);
        write(sink, tx.inputs(), wire);
        sink.write_variable_little_endian(tx.locktime());
        sink.write_variable_little_endian(tx.version());
    }
}

// Computations.
//-----------------------------------------------------------------------------

inline
hash_digest hash_compute(transaction_raw const& tx) {
    return bitcoin_hash(to_data(tx, true));
}

hash_digest hash_compute(transaction_raw const& tx, uint32_t sighash_type) {
    auto serialized = to_data(tx, true);
    extend_data(serialized, to_little_endian(sighash_type));
    return bitcoin_hash(serialized);
}

// Validation helpers.
//-----------------------------------------------------------------------------

bool is_coinbase(transaction_raw const& tx) {
    return tx.inputs().size() == 1 && tx.inputs().front().previous_output().is_null();
}

// True if coinbase and has invalid input[0] script size.
bool is_oversized_coinbase(transaction_raw const& tx) {
    if ( ! is_coinbase(tx)) return false;

    auto const script_size = tx.inputs().front().script().serialized_size(false);
    return script_size < min_coinbase_size || script_size > max_coinbase_size;
}

// True if not coinbase but has null previous_output(s).
bool is_null_non_coinbase(transaction_raw const& tx) {
    if (is_coinbase(tx)) return false;

    auto const invalid = [](input const& input) {
        return input.previous_output().is_null();
    };

    return std::any_of(tx.inputs().begin(), tx.inputs().end(), invalid);
}

// private
bool all_inputs_final(transaction_raw const& tx) {
    auto const finalized = [](input const& input) {
        return input.is_final();
    };

    return std::all_of(tx.inputs().begin(), tx.inputs().end(), finalized);
}

bool is_final(transaction_raw const& tx, size_t block_height, uint32_t block_time) {
    auto const max_locktime = [=]() {
        return tx.locktime() < locktime_threshold ? safe_unsigned<uint32_t>(block_height) : block_time;
    };

    return tx.locktime() == 0 || tx.locktime() < max_locktime() || all_inputs_final(tx);
}

bool is_locked(transaction_raw const& tx, size_t block_height, uint32_t median_time_past) {
    if (tx.version() < relative_locktime_min_version || is_coinbase(tx))
        return false;

    auto const locked = [block_height, median_time_past](input const& input) {
        return input.is_locked(block_height, median_time_past);
    };

    // If any input is relative time locked the transaction_raw is as well.
    return std::any_of(tx.inputs().begin(), tx.inputs().end(), locked);
}

// This is not a consensus rule, just detection of an irrational use.
bool is_locktime_conflict(transaction_raw const& tx) {
    return tx.locktime() != 0 && all_inputs_final(tx);
}

// Returns max_uint64 in case of overflow.
uint64_t total_input_value(transaction_raw const& tx) {
    ////static_assert(max_money() < max_uint64, "overflow sentinel invalid");
    auto const sum = [](uint64_t total, input const& input) {
        auto const& prevout = input.previous_output().validation.cache;
        auto const missing = !prevout.is_valid();

        // Treat missing previous outputs as zero-valued, no math on sentinel.
        return ceiling_add(total, missing ? 0 : prevout.value());
    };

    return std::accumulate(tx.inputs().begin(), tx.inputs().end(), uint64_t(0), sum);
}

// Returns max_uint64 in case of overflow.
uint64_t total_output_value(transaction_raw const& tx) {
    ////static_assert(max_money() < max_uint64, "overflow sentinel invalid");
    auto const sum = [](uint64_t total, const output& output) {
        return ceiling_add(total, output.value());
    };

    return std::accumulate(tx.outputs().begin(), tx.outputs().end(), uint64_t(0), sum);
}

uint64_t fees(transaction_raw const& tx) {
    return floor_subtract(total_input_value(tx), total_output_value(tx));
}

bool is_overspent(transaction_raw const& tx) {
    return !is_coinbase(tx) && total_output_value(tx) > total_input_value(tx);
}

// Returns max_size_t in case of overflow.
inline
size_t signature_operations(transaction_raw const& tx, chain_state::ptr const& state) {
    return state ? signature_operations(tx, state->is_enabled(rule_fork::bip16_rule)) : max_size_t;
}

// Returns max_size_t in case of overflow.
size_t signature_operations(transaction_raw const& tx, bool bip16_active) {
    auto const in = [bip16_active](size_t total, input const& input) {
        // This includes BIP16 p2sh additional sigops if prevout is cached.
        return ceiling_add(total, input.signature_operations(bip16_active));
    };

    auto const out = [](size_t total, const output& output) {
        return ceiling_add(total, output.signature_operations());
    };

    return std::accumulate(tx.inputs().begin(), tx.inputs().end(), size_t{0}, in) +
        std::accumulate(tx.outputs().begin(), tx.outputs().end(), size_t{0}, out);
}

bool is_missing_previous_outputs(transaction_raw const& tx) {
    auto const missing = [](input const& input) {
        auto const& prevout = input.previous_output();
        auto const coinbase = prevout.is_null();
        auto const missing = ! prevout.validation.cache.is_valid();
        return missing && !coinbase;
    };

    // This is an optimization of !missing_inputs().empty();
    return std::any_of(tx.inputs().begin(), tx.inputs().end(), missing);
}

point::list previous_outputs(transaction_raw const& tx) {
    point::list prevouts(tx.inputs().size());

    auto const pointer = [](input const& input) {
        return input.previous_output();
    };

    auto const& ins = tx.inputs();
    std::transform(ins.begin(), ins.end(), prevouts.begin(), pointer);
    return prevouts;
}

point::list missing_previous_outputs(transaction_raw const& tx) {
    point::list prevouts;

    for (auto& input: tx.inputs()) {
        auto const& prevout = input.previous_output();
        auto const missing = ! prevout.validation.cache.is_valid();

        if (missing && !prevout.is_null()) {
            prevouts.push_back(prevout);
        }
    }

    return prevouts;
}

hash_list missing_previous_transaction_raws(transaction_raw const& tx) {
    auto const points = missing_previous_outputs(tx);
    hash_list hashes(points.size());

//    auto const hasher = [](output_point const& point) { return point.hash(); };
    auto const hasher = [](point const& point) { return point.hash(); };

    std::transform(points.begin(), points.end(), hashes.begin(), hasher);
    return distinct(hashes);
}

bool is_internal_double_spend(transaction_raw const& tx) {
    auto prevouts = previous_outputs(tx);
    std::sort(prevouts.begin(), prevouts.end());
    auto const distinct_end = std::unique(prevouts.begin(), prevouts.end());
    auto const distinct = (distinct_end == prevouts.end());
    return ! distinct;
}

bool is_double_spend(transaction_raw const& tx, bool include_unconfirmed) {
    auto const spent = [include_unconfirmed](input const& input) {
        auto const& prevout = input.previous_output().validation;
        return prevout.spent && (include_unconfirmed || prevout.confirmed);
    };

    return std::any_of(tx.inputs().begin(), tx.inputs().end(), spent);
}

bool is_dusty(transaction_raw const& tx, uint64_t minimum_output_value) {
    auto const dust = [minimum_output_value](const output& output) {
        return output.is_dust(minimum_output_value);
    };

    return std::any_of(tx.outputs().begin(), tx.outputs().end(), dust);
}

bool is_mature(transaction_raw const& tx, size_t height) {
    auto const mature = [height](input const& input) {
        return input.previous_output().is_mature(height);
    };

    return std::all_of(tx.inputs().begin(), tx.inputs().end(), mature);
}

// Coinbase transaction_raws return success, to simplify iteration.
code connect_input(transaction_raw const& tx, chain_state const& state, size_t input_index) {
    if (input_index >= tx.inputs().size())
        return error::operation_failed;

    if (is_coinbase(tx)) return error::success;

    auto const& prevout = tx.inputs()[input_index].previous_output().validation;

    // Verify that the previous output cache has been populated.
    if (!prevout.cache.is_valid())
        return error::missing_previous_output;

    auto const forks = state.enabled_forks();
    auto const index32 = static_cast<uint32_t>(input_index);

    // Verify the transaction_raw input script against the previous output.
    return script::verify(tx, index32, forks);
}

// Validation.
//-----------------------------------------------------------------------------

// These checks are self-contained; blockchain (and so version) independent.
code check(transaction_raw const& tx, bool transaction_pool) {
    if (tx.inputs().empty() || tx.outputs().empty())
        return error::empty_transaction;

    else if (is_null_non_coinbase(tx))
        return error::previous_output_null;

    else if (total_output_value(tx) > max_money())
        return error::spend_overflow;

    else if ( ! transaction_pool && is_oversized_coinbase(tx))
        return error::invalid_coinbase_script_size;

    else if (transaction_pool && is_coinbase(tx))
        return error::coinbase_transaction;

    else if (transaction_pool && is_internal_double_spend(tx))
        return error::transaction_internal_double_spend;

    else if (transaction_pool && serialized_size(tx, true) >= get_max_block_size(is_bitcoin_cash()))
        return error::transaction_size_limit;

    // We cannot know if bip16 is enabled at this point so we disable it.
    // This will not make a difference unless prevouts are populated, in which
    // case they are ignored. This means that p2sh sigops are not counted here.
    // This is a preliminary check, the final count must come from accept().
    // Reenable once sigop caching is implemented, otherwise is deoptimization.
    ////else if (transaction_pool && signature_operations(false) > get_max_block_sigops(is_bitcoin_cash()))
    ////    return error::transaction_raw_legacy_sigop_limit;

    return error::success;
}

// code accept(transaction_raw const& tx, bool transaction_pool) {
//     auto const state = validation.state;
//     return state ? accept(*state, transaction_pool) : error::operation_failed;
// }

// These checks assume that prevout caching is completed on all tx.inputs.
code accept(transaction_raw const& tx, chain_state const& state, bool duplicate, bool transaction_pool) {
    auto const bip16 = state.is_enabled(rule_fork::bip16_rule);
    auto const bip30 = state.is_enabled(rule_fork::bip30_rule);
    auto const bip68 = state.is_enabled(rule_fork::bip68_rule);

    // We don't need to allow tx pool acceptance of an unspent duplicate
    // because tx pool inclusion cannot be required by consensus.
    auto const duplicates = state.is_enabled(rule_fork::allow_collisions) &&
        !transaction_pool;

    if (transaction_pool && state.is_under_checkpoint())
        return error::premature_validation;

    if (transaction_pool && ! chain::is_final(tx, state.height(), state.median_time_past()))
        return error::transaction_non_final;

    //*************************************************************************
    // CONSENSUS:
    // A transaction_raw hash that exists in the chain is not acceptable even if
    // the original is spent in the new block. This is not necessary nor is it
    // described by BIP30, but it is in the code referenced by BIP30. As such
    // the tx pool need only test against the chain, skipping the pool.
    //*************************************************************************
    // else if (!duplicates && bip30 && validation.duplicate)
    else if (!duplicates && bip30 && duplicate)
        return error::unspent_duplicate;

    else if (is_missing_previous_outputs(tx))
        return error::missing_previous_output;

    else if (is_double_spend(tx, transaction_pool))
        return error::double_spend;

    // This relates height to maturity of spent coinbase. Since reorg is the
    // only way to decrease height and reorg invalidates, this is cache safe.
    else if (!is_mature(tx, state.height()))
        return error::coinbase_maturity;

    else if (is_overspent(tx))
        return error::spend_exceeds_value;

    else if (bip68 && is_locked(tx, state.height(), state.median_time_past()))
        return error::sequence_locked;

    // This recomputes sigops to include p2sh from prevouts if bip16 is true.
    else if (transaction_pool && signature_operations(tx, bip16) > get_max_block_sigops(is_bitcoin_cash()))
        return error::transaction_embedded_sigop_limit;

    return error::success;
}

// code connect(transaction_raw const& tx) {
//     auto const state = validation.state;
//     return state ? connect(*state) : error::operation_failed;
// }

code connect(transaction_raw const& tx, chain_state const& state) {
    code ec;

    for (size_t input = 0; input < tx.inputs().size(); ++input) {
        if ((ec = connect_input(tx, state, input))) return ec;
    }

    return error::success;
}

}} // namespace libbitcoin::chain
