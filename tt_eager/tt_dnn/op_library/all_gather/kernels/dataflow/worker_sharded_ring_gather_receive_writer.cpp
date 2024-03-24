// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    DPRINT << "RW: START\n";
    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));

    ShardAddrGen<shard_type> output_tensor_shard_writer;
    uint32_t arg_index = 0;
    uint32_t const remote_sender_worker_x = get_arg_val<uint32_t>(arg_index++);
    uint32_t const remote_sender_worker_y = get_arg_val<uint32_t>(arg_index++);
    uint32_t const remote_sender_reader_semaphore_addres = get_arg_val<uint32_t>(arg_index++);
    uint32_t const max_shards_per_eth_buffer = get_arg_val<uint32_t>(arg_index++);
    uint32_t const num_transfers = get_arg_val<uint32_t>(arg_index++);
    ShardAddrGen<shard_type>::build_with_placement_new(&output_tensor_shard_writer, arg_index);
    arg_index += output_tensor_shard_writer.get_num_args_consumed();

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    // Each worker receiver writer matches with a specific worker sender reader
    // Used to signal that data has been committed to memory and can be read
    const uint64_t worker_send_reader_semaphore_noc_addr =
        get_noc_addr(remote_sender_worker_x, remote_sender_worker_y, remote_sender_reader_semaphore_addres);

    uint32_t total_num_shards = output_tensor_shard_writer.get_num_dest_cores() *
                             (output_tensor_shard_writer.get_chunks_per_core_before_advance() - 1);
    DPRINT << "RW: num_transfers: " << num_transfers << "\n";
    DPRINT << "RW: total_num_shards: " << total_num_shards << "\n";
    DPRINT << "RW: max_shards_per_eth_buffer: " << max_shards_per_eth_buffer << "\n";

    for (uint32_t d = 0; d < num_transfers; d++) {
        uint32_t num_shards_to_send = std::min(max_shards_per_eth_buffer, total_num_shards);

        DPRINT << "RW: Transfer " << d << "th transfer" << ENDL();
        DPRINT << "RW: \toutput_tensor_shard_writer.curr_worker_index " << output_tensor_shard_writer.curr_worker_index << "\n";
        DPRINT << "RW: \toutput_tensor_shard_writer.num_dest_cores " << output_tensor_shard_writer.num_dest_cores << "\n";
        write_chunk_sharded(cb_id_in0, output_tensor_shard_writer, num_shards_to_send);  // 1 shard = 1 page?
        // write_chunk_sharded(cb_id_in0, output_tensor_shard_writer, 1);  // 1 shard = 1 page?
        // Call above finished by we never see ths prints below
        DPRINT << "RW: \tget_noc_addr\n";
        DPRINT << "RW: Semaphore increment (y|x) " << (uint32_t)((remote_sender_worker_y << 16) | remote_sender_worker_x) << " " << (uint32_t)remote_sender_reader_semaphore_addres << ENDL();
        DPRINT << "RW: \tnoc_semaphore_inc\n";

        // noc_semaphore_inc(worker_send_reader_semaphore_noc_addr, 1);
        noc_semaphore_inc(worker_send_reader_semaphore_noc_addr, num_shards_to_send);
        DPRINT << "RW: \tdone noc_semaphore_inc\n";
        total_num_shards -= num_shards_to_send;

        if (total_num_shards == 0) {
            DPRINT << "RW: total_num_shards=" << total_num_shards << "\n";
            DPRINT << "RW: d=" << d << "\n";
            DPRINT << "RW: num_transfers=" << num_transfers << "\n";
        }
        ASSERT(total_num_shards > 0 || d == num_transfers - 1); // If we are out of shards, make sure we are on the last transfer
    }

    DEBUG_STATUS('R', 'W', 'D', 'D');
    DPRINT << "RW: EVERTHING DONE \n" << ENDL();


    DPRINT << "RW: END\n";
    DPRINT << "RW: END1\n";
    DPRINT << "RW: END2\n";
    DPRINT << "RW: END3\n";
}
