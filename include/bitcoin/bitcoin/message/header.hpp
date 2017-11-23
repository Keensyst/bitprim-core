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
#ifndef LIBBITCOIN_MESSAGE_HEADER_MESSAGE_HPP
#define LIBBITCOIN_MESSAGE_HEADER_MESSAGE_HPP

#include <cstddef>
#include <cstdint>
#include <istream>
#include <memory>
#include <bitcoin/bitcoin/chain/header.hpp>
#include <bitcoin/bitcoin/define.hpp>
#include <bitcoin/bitcoin/message/version.hpp>
#include <bitcoin/bitcoin/utility/data.hpp>
#include <bitcoin/bitcoin/utility/reader.hpp>

namespace libbitcoin {
namespace message {

//Note(fernando): I don't know what is the purpose of this class
class BC_API header
    : public chain::header
{
public:
    using list = std::vector<header>;
    using ptr = std::shared_ptr<header>;
    using const_ptr = std::shared_ptr<const header>;
    using ptr_list = std::vector<ptr>;
    using const_ptr_list = std::vector<const_ptr>;

    header();
    header(uint32_t version, hash_digest const& previous_block_hash,
        hash_digest const& merkle, uint32_t timestamp, uint32_t bits,
        uint32_t nonce);

    // header(uint32_t version, hash_digest&& previous_block_hash,
    //     hash_digest&& merkle, uint32_t timestamp, uint32_t bits,
    //     uint32_t nonce);

    header(header const& other);
    // header(header&& other);
    header(chain::header const& other);
    // header(chain::header&& other);

   
    /// This class is move assignable but not copy assignable. //Note(fernando): ????
    header& operator=(header const&) /*= delete*/;
    // header& operator=(header&& other);

    header& operator=(chain::header const& other);
    // header& operator=(chain::header&& other);


    // bool operator==(const chain::header& other) const;
    // bool operator!=(const chain::header& other) const;
    // bool operator==(const header& other) const;
    // bool operator!=(const header& other) const;

    friend
    bool operator==(header const& x, chain::header const& y);
    friend
    bool operator!=(header const& x, chain::header const& y);
    friend
    bool operator==(header const& x, header const& y);
    friend
    bool operator!=(chain::header const& x, chain::header const& y);


    bool from_data(uint32_t version, data_chunk const& data);
    bool from_data(uint32_t version, std::istream& stream);
    bool from_data(uint32_t version, reader& source);
    data_chunk to_data(uint32_t version) const;
    void to_data(uint32_t version, std::ostream& stream) const;
    void to_data(uint32_t version, writer& sink) const;
    void reset();
    size_t serialized_size(uint32_t version) const;


    static 
    header factory_from_data(uint32_t version, data_chunk const& data);
    
    static 
    header factory_from_data(uint32_t version, std::istream& stream);
    
    static 
    header factory_from_data(uint32_t version, reader& source);
    
    static 
    size_t satoshi_fixed_size(uint32_t version);




    static 
    std::string const command;
    
    static 
    uint32_t const version_minimum;
    
    static 
    uint32_t const version_maximum;
};

} // namespace message
} // namespace libbitcoin

#endif
