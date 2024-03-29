// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dtx.hpp"
#include "dtx_passes.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

using namespace std;

vector<vector<int>> dim_order_counting(vector<int> shape, vector<int> dim_order, vector<int> block_shape) {
    bool DEBUG = false;
    if (DEBUG) cout << "\n\nHelper function: dim order counting" << endl;

    int rank = shape.size();
    vector<int> shape_reordered;
    for (int d=0; d<rank; d++) {
        shape_reordered.push_back(shape[dim_order[d]]);
    }
    int count_size = vector_product(shape);

    if (DEBUG) {
        cout << "shape           = " << v2s(shape) << endl;
        cout << "dim order       = " << v2s(dim_order) << endl;
        cout << "shape reordered = " << v2s(shape_reordered) << endl;
        cout << "count size      = " << count_size << endl;
        cout << "block shape      = " << v2s(block_shape) << endl;
    }

    vector<vector<int>> list_of_counted_dims;
    vector<int> counter = zeros(rank);
    for (int i=0; i<count_size; i++){
        vector<int> counter_reordered;

        for (int d=0; d<rank; d++) {
            counter_reordered.push_back(counter[dim_order[d]]);
        }

        vector<int> str = vector_multiplication(counter_reordered, block_shape);
        vector<int> end = vector_addition(str, block_shape);
        end[rank-1]--;
        end[rank-2]--;
        if(DEBUG)
            cout << s(3) << "counter = " << v2s(counter) << ", reorderd = " << v2s(counter_reordered) << ";   " << v2s(str) << " => " << v2s(end) << endl;
        list_of_counted_dims.push_back(counter_reordered);

        counter.back()++;
        for (int d=rank-1; d>0; d--) {
            if (counter[d] == shape_reordered[d]) {
                counter[d-1]++;
                counter[d] = 0;
            }
        }
    }

    return list_of_counted_dims;
}


bool block_2d_matrix(DataTransformations * dtx, vector<int> dim_order, vector<int> block_shape_yx) {
    bool DEBUG = false;
    assert(dim_order.size() == 3);
    assert(block_shape_yx.size() == 2);
    assert(block_shape_yx[0] > 0);
    assert(block_shape_yx[1] > 0);
    if (DEBUG) cout << "\n\nPASS: Block 2d matrix" << endl;

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("block_2d_matrix", producer->groups.size());  // TODO: generalize for groups>1
    dtx->transformations.push_back(consumer);

    for (int group_idx=0; group_idx<producer->groups.size(); group_idx++) {
        if (DEBUG) cout << "\n\n" << s(2) << "Group = " << group_idx << endl;

        TensorPairGroup * consumer_group = consumer->groups[group_idx];
        TensorPairGroup * producer_group = producer->groups[group_idx];
        inherit_group_attributes_from_producer(producer_group, consumer_group);

        vector<int> shape = producer_group->shape;
        int rank = producer_group->shape.size();
        assert(rank <= 3);

        vector<int> block_shape = vector_pad_on_left(block_shape_yx, rank-2, 1);
        if (DEBUG) cout << s(4) << "block shape      = " << v2s(block_shape) << endl;
        if (DEBUG) cout << s(4) << "shape      = " << v2s(shape) << endl;
        vector<int> shape_blocked = vector_division(shape, block_shape);
        vector<vector<int>> list_of_counted_dims = dim_order_counting(shape_blocked,   dim_order, block_shape);

        vector<int> consumer_str = zeros(rank);
        vector<int> consumer_end = vector_addition(consumer_str, block_shape, -1);


        if (shape.size() != dim_order.size()) throw std::runtime_error("shape and dim_order dont have the same rank!");

        int block_size = block_shape[Y(rank)] * block_shape[X(rank)];
        int num_blocks = (shape[Y(rank)] / block_shape[Y(rank)]) * (shape[X(rank)] / block_shape[X(rank)]);
        assert(list_of_counted_dims.size() == num_blocks);
        int consumer_shape_z = Z(rank) >= 0 ? shape[Z(rank)] * num_blocks : num_blocks;
        assert(consumer_shape_z == num_blocks); // TODO: generalize for z > 1
        consumer_group->shape = {consumer_shape_z, block_shape[Y(rank)], block_shape[X(rank)]};

        if (DEBUG) cout << s(4) << "Tensor Pairs: " << num_blocks << endl;

        for (int i=0; i< list_of_counted_dims.size(); i++) {
            for(int j = 0; j < list_of_counted_dims[i].size(); j++) {
                if (DEBUG) std::cout << "dim " << list_of_counted_dims[i][j] << std::endl;
            }
            if (DEBUG) std::cout <<  std::endl;
            // Source Tensor: within the ND tensor from producer
            vector<int> str;
            vector<int> end;
            str = vector_multiplication(list_of_counted_dims[i], block_shape);
            end = vector_addition(str, block_shape, -1);
            vector<int> consumer_str = {i, 0, 0};
            vector<int> consumer_end = {i, block_shape[Y(rank)]-1, block_shape[X(rank)]-1};

            TensorPair * tp = new TensorPair(new DTXTensor({str}, {end}),
                                            group_idx,
                                            new DTXTensor({consumer_str}, {consumer_end}));
            consumer_group->tensor_pairs.push_back(tp);

            if (DEBUG) cout << s(6) << i << ".  " << tp->get_string() << endl;
        }
    }
    return true;
}
