/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef BIPART_H_
#define BIPART_H_

#include "galois/graphs/LC_CSR_Hypergraph.h"
#include "galois/AtomicWrapper.h"

class MetisNode;
typedef uint32_t EdgeTy;
const int dim = 1433;
using GGraph   = 
    galois::graphs::LC_CSR_Hypergraph<MetisNode, EdgeTy>::with_no_lockable<true>::type::with_numa_alloc<true>::type;
using GNode    = GGraph::GraphNode;
using GNodeBag = galois::InsertBag<GNode>;

constexpr galois::MethodFlag flag_no_lock = galois::MethodFlag::UNPROTECTED;
// algorithms

enum scheduleMode {PLD, WD, RI, PP, MRI, MWD, DEG, MDEG, HIS, RAND};

class MetisNode {

  struct coarsenData {
    int matched : 1;
    GNode parent;
  };
  struct refineData {
    unsigned partition;
    unsigned oldPartition;
    bool maybeBoundary;
  };
  struct partitionData {
    bool locked;
  };

  partitionData pd;

  void initCoarsen() {
    data.cd.matched     = false;
    data.cd.parent      = NULL;
    netval = 0;
  }

public:
  //bool flag;
  int nodeid;
  galois::CopyableAtomic<int> netnum;
  galois::CopyableAtomic<int> netvals;
  double netval;
  galois::CopyableAtomic<int> netrand;
  std::vector<double> EMvec = std::vector<double>(dim,0);
  std::vector<double> tmpvec = std::vector<double>(dim,0);

  // int num;
  explicit MetisNode(int weight) : _weight(weight) {
    initCoarsen();
  }
  MetisNode(unsigned weight, GNode child0, GNode child1 = NULL)
      : _weight(weight) {
    initCoarsen();
  }

  MetisNode() : _weight(1) {
    initCoarsen();
    data.cd.matched = false;
  }

  // call to switch data to refining
  int getWeight() const { return _weight; }
  void setWeight(int weight) { _weight = weight; }


  void setParent(GNode p) { data.cd.parent = p; }
  GNode getParent() const {
    assert(data.cd.parent);
    return data.cd.parent;
  }

  void setMatched() { data.cd.matched = true; }
  void notMatched() { data.cd.matched = false; }
  bool isMatched() const { return data.cd.matched; }

  unsigned getPart() const { return data.rd.partition; }
  void setPart(unsigned val) { data.rd.partition = val; }

private:
  union {
    coarsenData cd;
    refineData rd;
  } data;

  unsigned _weight;
};

// Structure to keep track of graph hirarchy
class MetisGraph {
  MetisGraph* coarser;
  MetisGraph* finer;

  GGraph graph;

public:
  MetisGraph() : coarser(0), finer(0) {}

  explicit MetisGraph(MetisGraph* finerGraph) : coarser(0), finer(finerGraph) {
    finer->coarser = this;
  }

  const GGraph* getGraph() const { return &graph; }
  GGraph* getGraph() { return &graph; }
  MetisGraph* getFinerGraph() const { return finer; }
  MetisGraph* getCoarserGraph() const { return coarser; }

  //unsigned getNumNodes() { return std::distance(graph.cellList().begin(), graph.cellList().end()); }

  unsigned getTotalWeight() {
    MetisGraph* f = this;
    while (f->finer)
      f = f->finer;
    //return std::distance(f->graph.cellList().begin(), f->graph.cellList().end());
  }
};

// Metrics
unsigned graphStat(GGraph& graph);
// Coarsening
MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo);
MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo,
                    scheduleMode sMode);

// Partitioning
//void partition(MetisGraph* coarseMetisGraph, unsigned K);
// Refinement
 void refine(MetisGraph* coarseGraph, unsigned K, std::string initEmb);
 //void imp(MetisGraph* coarseGraph);

#endif
