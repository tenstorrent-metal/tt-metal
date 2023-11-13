/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <string>
#include <set>
#include <optional>
#include <algorithm>
#include <limits>
#include "common/assert.hpp"
#include "third_party/umd/device/tt_xy_pair.h"

using std::pair;

using CoreCoord = tt_xy_pair;

constexpr inline bool operator<=(const CoreCoord &a, const CoreCoord &b) {
    return (a < b) or (a == b);
}

struct RelativeCoreCoord {
  long x = 0;
  long y = 0;

  std::string str() const { return "(x=" + std::to_string(x) + ",y=" + std::to_string(y) + ")"; }

};

constexpr inline bool operator==(const RelativeCoreCoord &a, const RelativeCoreCoord &b) { return a.x == b.x && a.y == b.y; }

constexpr inline bool operator!=(const RelativeCoreCoord &a, const RelativeCoreCoord &b) { return !(a == b); }

inline CoreCoord get_core_coord_from_relative(const RelativeCoreCoord& in, const CoreCoord& grid_size) {
  CoreCoord coord;
  coord.x = in.x + ((in.x < 0)? grid_size.x : 0);
  coord.y = in.y + ((in.y < 0)? grid_size.y : 0);
  return coord;
}

struct CoreRange {
    CoreCoord start;
    CoreCoord end;
    CoreRange (const CoreCoord &point) {
      this->start = point;
      this->end = point;
    }

    CoreRange(const CoreCoord & start, const CoreCoord & end) {
        tt::log_assert(
            end.x >= start.x and end.y >= start.y,
            "Invalid core range for start: {}, end: {}", start.str(), end.str());

        this->start = start;
        this->end = end;
    }

    CoreRange(const CoreRange &other) = default;
    CoreRange& operator=(const CoreRange &other) = default;
    CoreRange(CoreRange &&other) = default;
    CoreRange& operator=(CoreRange &&other) = default;

    std::optional<CoreRange> intersects ( const CoreRange & other ) const
    {
        std::size_t x1 = std::max(this->start.x, other.start.x);
        std::size_t y1 = std::max(this->start.y, other.start.y);
        std::size_t x2 = std::min(this->end.x, other.end.x);
        std::size_t y2 = std::min(this->end.y, other.end.y);
        if (x1 <= x2 and y1 <= y2)
            return CoreRange( {x1, y1}, {x2, y2} );

        return {};
    }

    bool adjacent ( const CoreRange & other ) const
    {
        std::size_t x1 = std::max(this->start.x, other.start.x);
        std::size_t y1 = std::max(this->start.y, other.start.y);
        std::size_t x2 = std::min(this->end.x, other.end.x);
        std::size_t y2 = std::min(this->end.y, other.end.y);
        return ( (x2 + 1 == x1 && y1 <= y2) || (y2 + 1 == y1 && x1 <= x2) );
    }

    bool contains ( const CoreRange & other ) const
    {
        return (other.start.x >= this->start.x ) &&
               (other.end.x <= this->end.x) &&
               (other.start.y >= this->start.y)  &&
               (other.end.y <= this->end.y);
    }

    // Merge lined-up (in x or y dimension) intersecting/adjacent rectangles
    std::optional<CoreRange> merge ( const CoreRange & cr) const
    {
        if ( this->intersects(cr) || this->adjacent(cr) ) {
            if ( this->start.x == cr.start.x && this->end.x == cr.end.x )
                return CoreRange ( {this->start.x, std::min(this->start.y, cr.start.y)} , { this->end.x, std::max( this->end.y, cr.end.y) } );

            else if ( this->start.y == cr.start.y && this->end.y == cr.end.y )
                return CoreRange ( { std::min( this->start.x, cr.start.x ), this->start.y}, { std::max( this->end.x, cr.end.x) , this->end.y });
        }
        return std::nullopt;
    }

    std::set<CoreRange> diff ( const CoreRange & cr) const
    {
        auto irect = this->intersects(cr);
        if (!irect.has_value())
            return {*this};

        std::set<size_t> xs = {this->start.x, this->end.x};
        std::set<size_t> ys = {this->start.y, this->end.y};

        if ( this->start.x < cr.start.x < this->end.x ) xs.insert(cr.start.x);
        if ( this->start.x < cr.end.x < this->end.x) xs.insert(cr.end.x);
        if ( this->start.y < cr.start.y < this->end.y ) ys.insert(cr.start.y);
        if ( this->start.y < cr.end.y < this->end.y ) ys.insert(cr.end.y);

        std::vector<size_t> vxs(xs.begin(), xs.end());
        std::vector<size_t> vys(ys.begin(), ys.end());
        std::set<CoreRange> ret;
        for (unsigned i = 0; i < vxs.size()-1; i++){
            for (unsigned j = 0; j < vys.size()-1; j++){
                CoreRange r( {vxs[i],vys[i]}, {vxs[i+1], vys[i+1]});
                if (r.start != irect.value().start || r.end != irect.value().end )
                  ret.insert(r);
            }
        }
        return ret;
    }

    std::string str() const { return "[" + start.str() + " - " + end.str() + "]"; }

    size_t size() const { return (this->end.x - this->start.x + 1) * (this->end.y - this->start.y + 1); }
};

constexpr inline bool operator==(const CoreRange &a, const CoreRange &b) { return a.start == b.start && a.end == b.end; }

constexpr inline bool operator!=(const CoreRange &a, const CoreRange &b) { return !(a == b); }

constexpr inline bool operator<(const CoreRange &left, const CoreRange &right) {
  return (left.start < right.start || (left.start == right.start && left.end < right.end));
}

struct CoresInCoreRangeGenerator {
    CoreCoord current;
    CoreCoord end;
    int num_worker_cores_x;
    int num_worker_cores_y;

    CoresInCoreRangeGenerator(const CoreRange& core_range, const CoreCoord& worker_grid_size) {
        this->current = core_range.start;
        this->end = core_range.end;

        this->num_worker_cores_x = worker_grid_size.x;
        this->num_worker_cores_y = worker_grid_size.y;
    }

    pair<CoreCoord, bool> operator() () {
        CoreCoord coord = this->current;
        CoreCoord new_coord;

        new_coord.x = (coord.x + 1) % this->num_worker_cores_x;
        new_coord.y = coord.y + (new_coord.x == 0); // It means we moved to next row

        this->current = new_coord;

        bool terminate = this->end == coord;

        return {coord, terminate};
    }
};

class CoreRangeSet {
  public:
    CoreRangeSet(const std::set<CoreRange> &core_ranges) : ranges_(core_ranges) {
      for (auto outer_it = this->ranges_.begin(); outer_it != this->ranges_.end(); outer_it++) {
        for (auto inner_it = this->ranges_.begin(); inner_it != this->ranges_.end(); inner_it++) {
          if (outer_it == inner_it) {
            continue;
          }
          CoreRange first_core_range = *outer_it;
          CoreRange second_core_range = *inner_it;
          bool first_core_left_of_second = first_core_range.end.x < second_core_range.start.x;
          bool first_core_right_of_second = first_core_range.start.x > second_core_range.end.x;
          bool first_core_above_second = first_core_range.end.y < second_core_range.start.y;
          bool first_core_below_second = first_core_range.start.y > second_core_range.end.y;
          auto no_overlap = first_core_left_of_second or first_core_right_of_second or first_core_above_second or first_core_below_second;
          if (not no_overlap) {
            TT_THROW("Cannot create CoreRangeSet with specified core ranges because core ranges " + first_core_range.str() + " and " + second_core_range.str() + " overlap!");
          }
        }
      }
    }

    CoreRangeSet(const CoreRangeSet &other) = default;
    CoreRangeSet& operator=(const CoreRangeSet &other) = default;

    CoreRangeSet(CoreRangeSet &&other) = default;
    CoreRangeSet& operator=(CoreRangeSet &&other) = default;

    CoreRangeSet merge ( const std::set<CoreRange> & other) const
    {
      size_t min_x = std::numeric_limits<size_t>::max(), max_x = 0, min_y = std::numeric_limits<size_t>::max(), max_y = 0;
      std::set<CoreRange> crs = this->ranges_;
      crs.insert(other.begin(), other.end());

      for (const auto & cr : crs )
      {
        // std::cout << "merging " << cr.str() << std::endl;
        min_x = std::min ( min_x, cr.start.x);
        max_x = std::max ( max_x, cr.end.x);
        min_y = std::min ( min_y, cr.start.y);
        max_y = std::max ( max_y , cr.end.y );
      }

      bool grid[max_y+1][max_x+1] = {};

      for (const auto & cr : crs )
        for (unsigned y = cr.start.y; y <= cr.end.y; y++)
          for (unsigned x = cr.start.x; x <= cr.end.x; x++)
            grid[y][x] = true;

      crs.clear();
      for (unsigned y = min_y; y <= max_y; y++){
        std::set<CoreRange> filter_set, tmp, new_crs;
        std::vector < CoreRange > ranges;
        std::optional<unsigned> x_start;
        for (unsigned x = min_x; x <= max_x; x++){
          if (grid[y][x] && !x_start.has_value()){
            // std::cout << "Setting x_start " << x << "," << y << std::endl;
            x_start = x;
          }
          else if ( !grid[y][x] && x_start.has_value()){
            ranges.push_back( CoreRange( {x_start.value(), y}, {x-1, y} ));
            // std::cout << "added CR " << ranges.back().str() << std::endl;
            x_start = std::nullopt;
          }
        }
        if (x_start.has_value()){
          ranges.push_back( CoreRange( {x_start.value(), y}, {max_x, y} ) );
          // std::cout << "added CR " << ranges.back().str() << std::endl;
        }
        for (const auto & cr : ranges){
          for ( const auto & prev_cr : crs ){
              if ( auto merged = cr.merge(prev_cr) ){
                // std::cout << "merging " << cr.str() << " and " << prev_cr.str() << " with " << merged.value().str() << std::endl;
                new_crs.insert ( merged.value());
                filter_set.insert(prev_cr);
                filter_set.insert(cr);
              }
          }
          crs.insert ( cr );
        }
        // Set(A) = Set(A) - Set(B)
        std::set_difference( std::make_move_iterator( crs.begin() ),
                            std::make_move_iterator( crs.end() ),
                            filter_set.begin(), filter_set.end(),
            std::inserter(tmp, tmp.end()));
        crs.swap(tmp);
        crs.insert(new_crs.begin(), new_crs.end());
      }

      // for ( const auto & cr : crs ){
      //   std::cout << " final merged CR:" << cr.str() << std::endl;
      // }
      return CoreRangeSet(crs);
    }

    CoreRangeSet merge ( const CoreRangeSet & s ) const
    {
      return this->merge (s.ranges());
    }

    bool core_coord_in_core_ranges(const CoreCoord &core_coord) const {
      for (auto core_range : this->ranges_) {
        bool in_x_range = (core_coord.x >= core_range.start.x) and (core_coord.x <= core_range.end.x);
        bool in_y_range = (core_coord.y >= core_range.start.y) and (core_coord.y <= core_range.end.y);
        if (in_x_range and in_y_range) {
          return true;
        }
      }
      return false;
    }

    bool intersects ( const CoreRange & cr) const{
      for (auto local_cr : this->ranges_) {
        if ( local_cr.intersects(cr) ) return true;
      }
      return false;
    }

    const std::set<CoreRange>& ranges() const { return this->ranges_; }

    std::string str() const {
      if (this->ranges().size() > 0) {
        std::string core_range_set_str = "{";
        for (auto core_range : this->ranges_) {
          core_range_set_str += core_range.str() + ", ";
        }
        core_range_set_str[core_range_set_str.length() - 2] = '}';
        core_range_set_str.pop_back();
        return core_range_set_str;
      } else {
        return "{}";
      }
    }

    const uint32_t num_cores() const {
      uint32_t num_cores = 0;
      for (const auto& core_range : this->ranges()) {
        num_cores += core_range.size();
      }
      return num_cores;
    }

  private:
    std::set<CoreRange> ranges_;
};
const inline bool operator==(const CoreRangeSet &a, const CoreRangeSet &b) {
  if (a.ranges().size() == b.ranges().size()) {
    auto range_a = a.ranges();
    auto range_b = b.ranges();
    for (auto it_a = range_a.begin(), it_b = range_b.begin(); it_a != range_a.end(); it_a++, it_b++) {
      if (*it_a != *it_b) {
        return false;
      }
    }
    return true;
  }
  return false;
}

const inline bool operator!=(const CoreRangeSet &a, const CoreRangeSet &b) { return !(a == b); }

inline std::vector<CoreCoord> grid_to_cores(uint32_t num_cores, uint32_t grid_size_x, uint32_t grid_size_y, bool row_wise=false) {
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    TT_ASSERT(num_cores <= grid_size_x * grid_size_y);
    if (row_wise) {
        for(uint32_t i = 0; i < num_cores; ++i) {
            cores.push_back({i % grid_size_x, i / grid_size_x});
        }
    } else {
        for(uint32_t i = 0; i < num_cores; ++i) {
            cores.push_back({i / grid_size_y, i % grid_size_y});
        }
    }

    return cores;
}


namespace std {
template <>
struct hash<RelativeCoreCoord> {
  std::size_t operator()(RelativeCoreCoord const &o) const {
    std::size_t seed = 0;
    seed = std::hash<std::size_t>()(o.x) ^ std::hash<std::size_t>()(o.y) << 1;
    return seed;
  }
};
}
