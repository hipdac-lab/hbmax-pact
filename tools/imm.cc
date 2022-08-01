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

#include <iostream>
#include <sstream>
#include <string>

#include "ripples/configuration.h"
#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/imm.h"
#include "ripples/loaders.h"
#include "ripples/utility.h"

#include "omp.h"

#include "CLI/CLI.hpp"
#include "nlohmann/json.hpp"

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace ripples {

auto GetWalkIterationRecord(
    const typename IMMExecutionRecord::walk_iteration_prof &iter) {
  nlohmann::json res{{"NumSets", iter.NumSets}, {"Total", iter.Total}};
  for (size_t wi = 0; wi < iter.CPUWalks.size(); ++wi) {
    std::stringstream wname;
    wname << "CPU Worker " << wi;
    res[wname.str()] = nlohmann::json{{"NumSets", iter.CPUWalks[wi].NumSets},
                                      {"Total", iter.CPUWalks[wi].Total}};
  }
  for (size_t wi = 0; wi < iter.GPUWalks.size(); ++wi) {
    std::stringstream wname;
    wname << "GPU Worker " << wi;
    res[wname.str()] = nlohmann::json{{"NumSets", iter.GPUWalks[wi].NumSets},
                                      {"Total", iter.GPUWalks[wi].Total},
                                      {"Kernel", iter.GPUWalks[wi].Kernel},
                                      {"D2H", iter.GPUWalks[wi].D2H},
                                      {"Post", iter.GPUWalks[wi].Post}};
  }
  return res;
}

template <typename SeedSet>
auto GetExperimentRecord(const ToolConfiguration<IMMConfiguration> &CFG,
                         const IMMExecutionRecord &R, const SeedSet &seeds) {
  using ex_time_ms = std::chrono::duration<double, std::milli>;
  
  std::stringstream ss;
  for(size_t i = 0; i < seeds.size(); i++)
  {
    if(i != 0)
      ss << ",";
    ss << seeds[i];
  }
  std::string sseeds = ss.str();

  ex_time_ms thetasampling;
  for(size_t i = 0; i < R.ThetaEstimationGenerateRRR.size(); i++)
  {
    thetasampling += R.ThetaEstimationGenerateRRR[i];
  }
  
  ex_time_ms thetaselect;
  for(size_t i = 0; i < R.ThetaEstimationMostInfluential.size(); i++)
  {
    thetaselect += R.ThetaEstimationMostInfluential[i];
  }
  nlohmann::json experiment{
      {"Algorithm", "IMM"},
      {"VCounting", R.Counting},
      {"DiffusionModel", CFG.diffusionModel},
      {"Epsilon", CFG.epsilon},
      {"WGenerateRRRSets", R.GenerateRRRSets},
      {"Input", CFG.IFileName},
      {"K", CFG.k},
      {"L", 1},
      {"NumThreads", R.NumThreads},
      {"SampleThreads", CFG.streaming_workers},
      {"SelectThreads", CFG.seed_select_max_workers}, 
      {"Pivoting", R.Pivoting},
      {"RRRSetSizeBytes", R.RRRSetSize},
      {"Seeds", seeds},
      {"Theta", R.Theta},
      {"ThetaEstimation", R.ThetaEstimationTotal},
      {"WThetaGenerateRRR", R.ThetaEstimationGenerateRRR},
      {"WThetaEstimationGenerateRRR", thetasampling},
      {"ThetaEstimationMostInfluential", R.ThetaEstimationMostInfluential},
      {"Total", R.Total}};
      
  for (auto &ri : R.WalkIterations) {
    experiment["Iterations"].push_back(GetWalkIterationRecord(ri));
  }
  return experiment;
}

ToolConfiguration<ripples::IMMConfiguration> CFG;

void parse_command_line(int argc, char **argv) {
  CFG.ParseCmdOptions(argc, argv);
#pragma omp single
  CFG.streaming_workers = omp_get_max_threads();

  if (CFG.seed_select_max_workers == 0)
    CFG.seed_select_max_workers = CFG.streaming_workers;
  if (CFG.seed_select_max_gpu_workers == std::numeric_limits<size_t>::max())
    CFG.seed_select_max_gpu_workers = CFG.streaming_gpu_workers;
}

ToolConfiguration<ripples::IMMConfiguration> configuration() { return CFG; }

}  // namespace ripples

void process_mem_usage(double& rss_usage)
{
    double vm_usage     = 0.0;
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> vsize >> rss;
    }
    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
    vm_usage = vsize / (1024.0*1024.0);
    rss_usage = rss * page_size_kb/1024;
}

int main(int argc, char **argv) {
  auto console = spdlog::stdout_color_st("console");
  double vm1,vm2;
  // process command line
  ripples::parse_command_line(argc, argv);
  auto CFG = ripples::configuration();
  if (CFG.parallel) {
    if (ripples::streaming_command_line(
            CFG.worker_to_gpu, CFG.streaming_workers, CFG.streaming_gpu_workers,
            CFG.gpu_mapping_string) != 0) {
      console->error("invalid command line");
      return -1;
    }
  }
  spdlog::set_level(spdlog::level::info);

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  using dest_type = ripples::WeightedDestination<uint32_t, float>;
  using GraphFwd =
      ripples::Graph<uint32_t, dest_type, ripples::ForwardDirection<uint32_t>>;
  using GraphBwd =
      ripples::Graph<uint32_t, dest_type, ripples::BackwardDirection<uint32_t>>;
  
  process_mem_usage(vm1);
  console->info("Loading... {}",vm1);
  GraphFwd Gf = ripples::loadGraph<GraphFwd>(CFG, weightGen);
  GraphBwd G = Gf.get_transpose();
  process_mem_usage(vm2);
  console->info("Loading Done! {}",vm2);
  console->info("Number of Nodes : {},  Edges : {}", G.num_nodes(), G.num_edges());
  console->info("G.size : {} MB", vm2-vm1);
  
  nlohmann::json executionLog;

  std::vector<typename GraphBwd::vertex_type> seeds;
  ripples::IMMExecutionRecord R;

  trng::lcg64 generator;
  generator.seed(42UL);
  generator.split(2, 1);
  std::ofstream perf(CFG.OutputFile);

  if (CFG.parallel) {
    auto workers = CFG.streaming_workers;
    auto gpu_workers = CFG.streaming_gpu_workers;
    decltype(R.Total) real_total;
    if (CFG.diffusionModel == "IC") {
      ripples::StreamingRRRGenerator<
          decltype(G), decltype(generator),
          typename ripples::RRRsets<decltype(G)>::iterator,
          ripples::independent_cascade_tag>
          se(G, generator, R, workers - gpu_workers, gpu_workers,
             CFG.worker_to_gpu);
      auto start = std::chrono::high_resolution_clock::now();
      seeds = IMM3(G, CFG, 1, se, ripples::independent_cascade_tag{},
                  ripples::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start - R.Total;
      real_total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      ripples::StreamingRRRGenerator<
          decltype(G), decltype(generator),
          typename ripples::RRRsets<decltype(G)>::iterator,
          ripples::linear_threshold_tag>
          se(G, generator, R, workers - gpu_workers, gpu_workers,
             CFG.worker_to_gpu);
      auto start = std::chrono::high_resolution_clock::now();
      seeds = IMM3(G, CFG, 1, se, ripples::linear_threshold_tag{},
                  ripples::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start - R.Total;
      real_total = end - start;
    }

    console->info("IMM Parallel : {}ms", R.Total.count());
    console->info("IMM Parallel Real Total : {}ms", real_total.count());

    size_t num_threads;
#pragma omp single
    num_threads = omp_get_max_threads();
    R.NumThreads = num_threads;

    G.convertID(seeds.begin(), seeds.end(), seeds.begin());
    auto experiment = GetExperimentRecord(CFG, R, seeds);
    executionLog.push_back(experiment);
    perf << executionLog.dump(2);
  } else {
    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();
      seeds = IMM(G, CFG, 1, generator, R, ripples::independent_cascade_tag{},
                  ripples::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      seeds = IMM(G, CFG, 1, generator, R, ripples::linear_threshold_tag{},
                  ripples::sequential_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R.Total = end - start;
    }
    console->info("IMM squential : {}ms", R.Total.count());

    size_t num_threads;
#pragma omp single
    num_threads = omp_get_max_threads();
    R.NumThreads = num_threads;

    G.convertID(seeds.begin(), seeds.end(), seeds.begin());
    auto experiment = GetExperimentRecord(CFG, R, seeds);
    executionLog.push_back(experiment);
    perf << executionLog.dump(2);
  }

  return EXIT_SUCCESS;
}
