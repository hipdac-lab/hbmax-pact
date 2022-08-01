//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2019, Battelle Memorial Institute
// 
// Battelle Memorial Institute (hereinafter Battelle) hereby grants permission
// to any person or entity lawfully obtaining a copy of this software and
// associated documentation files (hereinafter “the Software”) to redistribute
// and use the Software in source and binary forms, with or without
// modification.  Such person or entity may use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and may permit
// others to do so, subject to the following conditions:
// 
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// 3. Other than as used herein, neither the name Battelle Memorial Institute or
//    Battelle may be used in any form whatsoever without the express written
//    consent of Battelle.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_GENERATE_RRR_SETS_H
#define RIPPLES_GENERATE_RRR_SETS_H

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>
#include <iostream>
#include "omp.h"

#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/imm_execution_record.h"
#include "ripples/utility.h"
#include "ripples/streaming_rrr_generator.h"
#include "ripples/huffman.h"

#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#ifdef ENABLE_MEMKIND
#include "memkind_allocator.h"
#endif

namespace ripples {

#ifdef ENABLE_MEMKIND
template<typename vertex_type>
using RRRsetAllocator = libmemkind::static_kind::allocator<vertex_type>;
#else
template <typename vertex_type>
using RRRsetAllocator = std::allocator<vertex_type>;
#endif

//! \brief The Random Reverse Reachability Sets type
template <typename GraphTy>
using RRRset = std::vector<typename GraphTy::vertex_type, RRRsetAllocator<typename GraphTy::vertex_type>>;
template <typename GraphTy>
using RRRsets = std::vector<RRRset<GraphTy>>;

//! \brief Execute a randomize BFS to generate a Random RR Set.
//!
//! \tparam GraphTy The type of the graph.
//! \tparam PRNGGeneratorTy The type of pseudo the random number generator.
//! \tparam diff_model_tag The policy for the diffusion model.
//!
//! \param G The graph instance.
//! \param r The starting point for the exploration.
//! \param generator The pseudo random number generator.
//! \param result The RRR set
//! \param tag The diffusion model tag.
template <typename GraphTy, typename RRRset>
size_t CheckRRRSize(const GraphTy &G, std::vector<RRRset> &RRRsets) {
  using vertex_type = typename GraphTy::vertex_type;
  auto in_begin=RRRsets.begin();
  auto in_end=RRRsets.end();
  size_t s1 = RRRsets.size();
  size_t total_rrr_size = 0;
  size_t mem_usage = 0;
  for (; in_begin != in_end; ++in_begin) {
    int s2=std::distance(in_begin->begin(),in_begin->end());
    total_rrr_size+=s2;
  }
  mem_usage = total_rrr_size * sizeof(vertex_type) / (1024 * 1024); // in MB
  return mem_usage;
}

template <typename GraphTy, typename RRRset>
void DumpRRRSets(const GraphTy &G, std::vector<RRRset> &RRRsets, size_t offset) {
  using vertex_type = typename GraphTy::vertex_type;
  auto in_begin=RRRsets.begin()+offset;
  auto in_end=RRRsets.end();
  size_t s1 = RRRsets.size()-offset;
  size_t total_rrr_size = 0;
  std::ofstream FILE("rrr_out.bin", std::ios::out | std::ofstream::binary);
  FILE.write(reinterpret_cast<const char *>(&s1), sizeof(s1));
  for (; in_begin != in_end; ++in_begin) {
    int s2=std::distance(in_begin->begin(),in_begin->end());
    total_rrr_size+=s2;
    FILE.write(reinterpret_cast<const char *>(&s2), sizeof(s2));
    FILE.write(reinterpret_cast<const char *> (&(*in_begin->begin())), s2*sizeof(vertex_type));
  }
  std::cout<<"write-external-rrr="<<s1<<", total-vtx-size="<<total_rrr_size;
  std::cout<<", total-bytes:"<<(total_rrr_size * sizeof(vertex_type))/(1024*1024)<<"MB."<<std::endl;
}

template <typename GraphTy>
void ReadRRRSets(const GraphTy &G) {
  using vertex_type = typename GraphTy::vertex_type;
  // auto in_begin=RRRsets.begin();
  // auto in_end=RRRsets.end();
  // size_t s1 = RRRsets.size();
  size_t s1 = 0;
  size_t total_rrr_size = 0;
  std::vector<vertex_type> tmp_A;
  vertex_type tmp_a;
  std::ifstream FILE("rrr_out.bin", std::ios::in | std::ofstream::binary);
  FILE.read(reinterpret_cast<char *>(&s1), sizeof(s1));
  for (int i=0; i<s1; i++) {
    int s2=0;
    FILE.read(reinterpret_cast<char *>(&s2), sizeof(s2));
    total_rrr_size+=s2;
    tmp_A.resize(s2);
    for(int j=0;j<s2;j++){
        FILE.read(reinterpret_cast<char *> (&tmp_a), sizeof(vertex_type));
        tmp_A.push_back(tmp_a);
    }
  }
  std::cout<<"read-external-rrr="<<s1<<", total-vtx-size="<<total_rrr_size;
  std::cout<<", total-bytes:"<<(total_rrr_size * sizeof(vertex_type))/(1024*1024)<<"Mb."<<std::endl;
  // return tmp_A;
}

template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
void AddRRRSet(const GraphTy &G, typename GraphTy::vertex_type r,
               PRNGeneratorTy &generator, RRRset<GraphTy> &result,
               diff_model_tag &&tag) {
  using vertex_type = typename GraphTy::vertex_type;

  trng::uniform01_dist<float> value;

  std::queue<vertex_type> queue;
  std::vector<bool> visited(G.num_nodes(), false);

  queue.push(r);
  visited[r] = true;
  result.push_back(r);

  while (!queue.empty()) {
    vertex_type v = queue.front();
    queue.pop();

    if (std::is_same<diff_model_tag, ripples::independent_cascade_tag>::value) {
      for (auto u : G.neighbors(v)) {
        if (!visited[u.vertex] && value(generator) <= u.weight) {
          queue.push(u.vertex);
          visited[u.vertex] = true;
          result.push_back(u.vertex);
        }
      }
    } else if (std::is_same<diff_model_tag,
                            ripples::linear_threshold_tag>::value) {
      float threshold = value(generator);
      for (auto u : G.neighbors(v)) {
        threshold -= u.weight;

        if (threshold > 0) continue;

        if (!visited[u.vertex]) {
          queue.push(u.vertex);
          visited[u.vertex] = true;
          result.push_back(u.vertex);
        }
        break;
      }
    } else {
      throw;
    }
  }
  std::stable_sort(result.begin(), result.end());
}

template <typename GraphTy, typename PRNGeneratorTy, typename diff_model_tag>
void AddRRRSet2(const GraphTy &G, typename GraphTy::vertex_type r,
               PRNGeneratorTy &generator, RRRset<GraphTy> &result,
               diff_model_tag &&tag) {
  using vertex_type = typename GraphTy::vertex_type;

  trng::uniform01_dist<float> value;

  std::deque<vertex_type> queue;
  std::vector<bool> visited(G.num_nodes(), false);
  double vm1,vm2;
  queue.push_front(r);
  visited[r] = true;
  result.push_back(r);
  while (!queue.empty()) {
    vertex_type v = queue.front();
    queue.pop_front();

    if (std::is_same<diff_model_tag, ripples::independent_cascade_tag>::value) {
      for (auto u : G.neighbors(v)) {
        if (!visited[u.vertex] && value(generator) <= u.weight) {
          queue.push_front(u.vertex);
          visited[u.vertex] = true;
          result.push_back(u.vertex);
        }
      }
    } else if (std::is_same<diff_model_tag,
                            ripples::linear_threshold_tag>::value) {
      float threshold = value(generator);
      for (auto u : G.neighbors(v)) {
        threshold -= u.weight;

        if (threshold > 0) continue;

        if (!visited[u.vertex]) {
          queue.push_front(u.vertex);
          visited[u.vertex] = true;
          result.push_back(u.vertex);
        }
        break;
      }
    } else {
      throw;
    }
  }
  result.shrink_to_fit(); // try collect memory 
  std::deque<vertex_type>().swap(queue);
  std::vector<bool>().swap(visited);
}

//! \brief Generate Random Reverse Reachability Sets - sequential.
//!
//! \tparam GraphTy The type of the garph.
//! \tparam PRNGeneratorty The type of the random number generator.
//! \tparam ItrTy A random access iterator type.
//! \tparam ExecRecordTy The type of the execution record
//! \tparam diff_model_tag The policy for the diffusion model.
//!
//! \param G The original graph.
//! \param generator The random numeber generator.
//! \param begin The start of the sequence where to store RRR sets.
//! \param end The end of the sequence where to store RRR sets.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename PRNGeneratorTy,
          typename ItrTy, typename ExecRecordTy,
          typename diff_model_tag>
void GenerateRRRSets(GraphTy &G, PRNGeneratorTy &generator,
                     ItrTy begin, ItrTy end,
                     ExecRecordTy &,
                     diff_model_tag &&model_tag,
                     sequential_tag &&ex_tag) {
  trng::uniform_int_dist start(0, G.num_nodes());

  for (auto itr = begin; itr < end; ++itr) {
    typename GraphTy::vertex_type r = start(generator[0]);
    AddRRRSet(G, r, generator[0], *itr,
              std::forward<diff_model_tag>(model_tag));
  }
}

template <typename GraphTy, typename PRNGeneratorTy,
          typename ItrTy, typename ExecRecordTy,
          typename diff_model_tag>
void GenerateRRRSets2(GraphTy &G, PRNGeneratorTy &generator,
                     ItrTy begin, ItrTy end,
                     ExecRecordTy &,
                     diff_model_tag &&model_tag,
                     sequential_tag &&ex_tag) {
  trng::uniform_int_dist start(0, G.num_nodes());
  
  for (auto itr = begin; itr < end; ++itr) {
    typename GraphTy::vertex_type r = start(generator[0]);
    AddRRRSet2(G, r, generator[0], *itr,
              std::forward<diff_model_tag>(model_tag));
  }
}

//! \brief Generate Random Reverse Reachability Sets - CUDA.
//!
//! \tparam GraphTy The type of the garph.
//! \tparam PRNGeneratorty The type of the random number generator.
//! \tparam ItrTy A random access iterator type.
//! \tparam ExecRecordTy The type of the execution record
//! \tparam diff_model_tag The policy for the diffusion model.
//!
//! \param G The original graph.
//! \param generator The random numeber generator.
//! \param begin The start of the sequence where to store RRR sets.
//! \param end The end of the sequence where to store RRR sets.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename PRNGeneratorTy,
          typename ItrTy, typename ExecRecordTy,
          typename diff_model_tag>
void GenerateRRRSets(const GraphTy &G,
                     StreamingRRRGenerator<GraphTy, PRNGeneratorTy, ItrTy, diff_model_tag> &se,
                     ItrTy begin, ItrTy end,
                     ExecRecordTy &,
                     diff_model_tag &&,
                     omp_parallel_tag &&) {
  se.generate(begin, end);
}

template <typename GraphTy, typename PRNGeneratorTy,
          typename ItrTy, typename ExecRecordTy,
          typename diff_model_tag, typename vertex_type>
void GenerateRRRSets2(const GraphTy &G,
                     StreamingRRRGenerator<GraphTy, PRNGeneratorTy, ItrTy, diff_model_tag> &se,
                     ItrTy begin, ItrTy end,
                     ExecRecordTy &,
                     diff_model_tag &&,
                     omp_parallel_tag &&,
                     std::vector<uint32_t> &globalcnt, vertex_type* maxvtx) {
  se.generate2(begin, end, globalcnt, maxvtx);
}

}  // namespace ripples

#endif  // RIPPLES_GENERATE_RRR_SETS_H
