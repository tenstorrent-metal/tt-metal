#include "util.hpp"
#include "dtx.hpp"



// ========================================================
//                      DTX HELPERS
// ========================================================

bool compare_two_vectors_of_ints(vector<int> a, vector<int> b) {
    bool pass = true;

    if (a.size() != b.size()) return false;
    for (int d=0; d<a.size(); d++) {
        if (a[d] != b[d]) return false;
    }
    return true;
}

bool compare_two_tensors(Tensor * a, Tensor * b) {
    if (compare_two_vectors_of_ints(a->str, b->str) && compare_two_vectors_of_ints(a->end, b->end)) return true;
    else return false;
}

bool compare_two_tensor_pairs(TensorPair * a, TensorPair * b) {
    if (compare_two_tensors(a->src_tensor, b->src_tensor) && compare_two_tensors(a->dst_tensor, b->dst_tensor) && a->src_group==b->src_group) return true;
    else return false;
}

bool compare_two_groups(TensorPairGroup * a, TensorPairGroup * b) {
    if (!compare_two_vectors_of_ints(a->shape, b->shape)) return false;

    for (int tp=0; tp<a->tensor_pairs.size(); tp++){
        if (!compare_two_tensor_pairs(a->tensor_pairs[tp], b->tensor_pairs[tp])) return false;
    }
    return true;
}


// ========================================================
//                      TENSOR OVERLAP
// ========================================================

bool has_overlap(Tensor * overlap) {
    bool has_overlap = true;
    for (int i=0; i<overlap->str.size(); i++){
        if (overlap->str[i] == -1) has_overlap = false;
    }
    return has_overlap;
}

// Given four points, calculate and return the line segment overlap
vector<int> calculate_line_segment_overlap_in_1d(int l1_str, int l1_end, int l2_str, int l2_end) {
    vector<int> overlap = {-2, -2};

    // No Overlap
    if ((l1_str > l2_end) || (l2_str > l1_end)) {
        return {-1, -1};}

    // Full overlap
    else if (l1_str >= l2_str and l1_end <= l2_end) {
        return {l1_str, l1_end};}
    else if (l2_str >= l1_str and l2_end <= l1_end) {
        return {l2_str, l2_end};}

    // Partial overlap
    else if (l1_str <= l2_str) {
        return {l2_str, l1_end};}
    else {
        return {l1_str, l2_end};}
}

Tensor * calculate_tensor_overlap_in_nd(Tensor * t0, Tensor * t1) {
    bool DEBUG = true;
    // Tensors must be of the same rank
    int rank = t1->rank;

    Tensor * overlap_nd = new Tensor();
    bool overlap_nd_exists = true;


    for (int d=0; d<rank; d++) {
        vector<int> overlap_1d = calculate_line_segment_overlap_in_1d(t0->str[d], t0->end[d], t1->str[d], t1->end[d]);
        //if (DEBUG) cout << "dim = " << d << ", overlap_1d = " << v2s(overlap_1d) << endl;

        if (overlap_1d[0] == -1 && overlap_1d[1] == -1) {
            overlap_nd_exists = false;
        }
        overlap_nd->str.push_back(overlap_1d[0]);
        overlap_nd->end.push_back(overlap_1d[1]);
        overlap_nd->rank++;
    }

    //if (DEBUG) cout << "nd overlap exists = " << overlap_nd_exists << endl;
    //if (DEBUG) cout << "nd overlap exists = " << has_overlap(overlap_nd) << endl;

    if (DEBUG) cout << "Calculating overlap between: " << t0->get_string() << " && " << t1->get_string() << "  ==  " << overlap_nd->get_string() << "    (" << has_overlap(overlap_nd) << ")" << endl;

    return overlap_nd;

}

pair<vector<int>, vector<int>> get_chunk_within_tensor(Tensor * t, int start_offset, int chunk_size) {
    pair<vector<int>, vector<int>> chunk_coordinates;
    int tensor_volume = t->volume();
    int rank = t->rank;
    assert(start_offset < tensor_volume);
    assert(chunk_size <= tensor_volume);
    auto chunk_start = t->str;
    int dim_to_increment = rank - 1;
    int count = 0;
    std::cout << "start_offset " << start_offset << std::endl;
    while (count < start_offset-1) {
        if(chunk_start[dim_to_increment]+1 > t->end[dim_to_increment]) {
            dim_to_increment--;
            assert(dim_to_increment >= 0);
        }
        chunk_start[dim_to_increment]++;
        count++;
    }
    auto chunk_end = chunk_start;
    count = 0;
    dim_to_increment = rank - 1;
    std::cout << "chunk_size " << chunk_size << std::endl;
    while(count < chunk_size-1) {
        if(chunk_end[dim_to_increment]+1 > t->end[dim_to_increment]) {
            dim_to_increment--;
            assert(dim_to_increment >= 0);
        }
        chunk_end[dim_to_increment]++;
        count++;
    }
    chunk_coordinates.first = chunk_start;
    chunk_coordinates.second = chunk_end;
    return chunk_coordinates;
}
