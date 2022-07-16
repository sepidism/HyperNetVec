/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "netvec.h"
#include "galois/AtomicHelpers.h"
#include <set>
#include <iostream>
#include <fstream>

namespace {
const int DIM = 128;
// This is only used on the terminal graph (find graph)
// Should workd for hmetis

void projectPart(MetisGraph* Graph) {
  GGraph* fineGraph   = Graph->getFinerGraph()->getGraph();
  GGraph* coarseGraph = Graph->getGraph();
  galois::do_all(
      galois::iterate(fineGraph->hedges, fineGraph->size()),
      [&](GNode n) {
        auto parent   = fineGraph->getData(n).getParent();
        auto& cn      = coarseGraph->getData(parent);
        for (int i = 0; i < DIM; i++)
          fineGraph->getData(n).EMvec[i] = cn.EMvec[i];
      },
      galois::loopname("project"));
}

// refine
void parallel_findMean(GGraph& g) {
  float w = 0.5;

  galois::do_all(
      galois::iterate(size_t{0},g.size()),
      [&](GNode n) {
          bool flag = false;
          std::vector<double> tmp(DIM,0.0);
              
          int size = std::distance(g.edges(n).begin(), g.edges(n).end());

          for (auto a : g.edges(n)) {
            auto node = g.getEdgeDst(a);
            for (int i = 0; i < DIM; i++) {
              tmp[i] += (g.getData(node).EMvec[i]);
            }
          }

          for (auto i = 0; i < DIM; i++) {
            g.getData(n).tmpvec[i] = ((w*tmp[i]) / size) +( (1-w) *g.getData(n).EMvec[i]);
          }

      },
      galois::loopname("find mean for each hedge"));


      for (auto n = 0; n < g.size(); n++) 
          g.getData(n).EMvec = g.getData(n).tmpvec;
}


} // namespace


void refine(MetisGraph* coarseGraph, unsigned rsize, std::string initEmb) {

 std::ifstream f(initEmb.c_str());
  
  std::string line;
 // std::set<unsigned> order;
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    uint64_t node;
    double val;
    ss >> node;
    unsigned iter = 0;
    while (ss >> val) {
      coarseGraph->getGraph()->getData(node).EMvec[iter] = val;
      iter++;
    }
  }
  do {
    MetisGraph* fineGraph = coarseGraph->getFinerGraph();
    auto gg               = coarseGraph->getGraph();
    
    //for (int i = 0; i < rsize; i++)
    bool notConverge = true;
    unsigned iter = 0;
    int maxIter = rsize;
    while (iter < maxIter) {
        parallel_findMean(*gg);
      iter++;
    }

    bool do_pro = true;
    if (fineGraph && do_pro) {
      projectPart(coarseGraph);
    }
  } while ((coarseGraph = coarseGraph->getFinerGraph()));
}
