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

#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <array>
#include <unordered_set>
#include <math.h>

#include "netvec.h"
#include "galois/graphs/Util.h"
#include "galois/Timer.h"
//#include "GraphReader.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/graphs/FileGraph.h"
#include "galois/LargeArray.h"

namespace cll = llvm::cl;

static const char* name = "netvec";
static const char* desc =
    "Hypergraph embedding system";
static const char* url = "netvec";

static cll::opt<bool> nodefeature(cll::Positional,cll::desc("node feature?"),
                               cll::init(true));
static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> features(cll::Positional,
                                      cll::desc("<feature file>"), cll::Required);
static cll::opt<std::string> initEmb(cll::Positional,
                                     cll::desc("initial embedding file name"), cll::Required);
static cll::opt<unsigned> csize(cll::Positional,
                                   cll::desc("<size of coarsest graph>"),
                                   cll::init(2));

static cll::opt<unsigned> task(cll::Positional,
                                   cll::desc("<ML task: node classification (1) or hyperedge prediction (2)>"),
                                   cll::init(1));
static cll::opt<scheduleMode> schedulingMode(
    cll::desc("Choose a inital scheduling mode:"),
    cll::values(clEnumVal(PLD, "PLD"), clEnumVal(PP, "PP"), clEnumVal(WD, "WD"),
                clEnumVal(RAND, "random")), cll::init(PP));

void Partition(MetisGraph* metisGraph, unsigned coarsenTo, unsigned K) {
  galois::StatTimer TM;
  TM.start();

  galois::StatTimer T("CoarsenSEP");
  galois::StatTimer T3("Refine");
  if (task == 1) {
  T.start();
  MetisGraph* mcg = coarsen(metisGraph, coarsenTo); 
  T.stop();
  GGraph* g = mcg->getGraph();
  std::ofstream out("edgelist.txt");

  T3.start();
  refine(mcg, 100, initEmb);
  T3.stop();
  std::cout << "Coarsening:," << T.get() << "\n";

  std::cout << "Refinement:," << T3.get() << "\n";
  }
  else {
  T.start();
  MetisGraph* mcg = coarsen(metisGraph, coarsenTo, schedulingMode); 
  T.stop();
  GGraph* g = mcg->getGraph();
  std::ofstream out("edgelist.txt");

  T3.start();
  refine(mcg, 8, initEmb);  
  T3.stop();
  std::cout << "Coarsening:," << T.get() << "\n";

  std::cout << "Refinement:," << T3.get() << "\n";
  }
  return;
}

int hash(unsigned val) {
  unsigned long int seed = val * 1103515245 + 12345;
  return((unsigned)(seed/65536) % 32768);
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

 // srand(-1);
  MetisGraph metisGraph;
  GGraph& graph = *metisGraph.getGraph();
  std::ifstream f(filename.c_str());
  //GGraph graph;// = *metisGraph.getGraph();
  std::string line;
  std::getline(f, line);
  std::stringstream ss(line);
  uint32_t i1;
  uint64_t i2;
  ss >> i1 >> i2;
  const int hedges = i1, nodes = i2;
  printf("hedges: %d\n", hedges);
  printf("nodes: %d\n\n", nodes);

  galois::StatTimer T("buildingG");
  T.start();
  // read rest of input and initialize hedges (build hgraph)
  galois::gstl::Vector<galois::PODResizeableArray<uint32_t> > edges_id(hedges+nodes);
  std::vector<std::vector<EdgeTy> > edges_data(hedges+nodes);
  std::vector<uint64_t> prefix_edges(nodes+hedges);
  int cnt = 0, edges = 0;
  while (std::getline(f, line)) {
    if (cnt >= hedges) {printf("ERROR: too many lines in input file\n"); exit(-1);}
    std::stringstream ss(line);
    int val;
    while (ss >> val) {
      if ((val < 0) || (val > nodes)) {printf("ERROR: node value %d out of bounds\n", val); exit(-1);}
      unsigned newval = hedges + (val);
      edges_id[cnt].push_back(newval);
      edges_id[newval].push_back(cnt);
      edges+=2;
    }
    cnt++;
  }

  f.close();
  graph.hedges = hedges;
  graph.hnodes = nodes;
  std::cout<<"number of edges "<<edges<<"\n";
  uint32_t sizes = hedges+nodes;
  galois::do_all(galois::iterate((uint32_t)0, sizes),
                [&](uint32_t c){
                  prefix_edges[c] = edges_id[c].size();
                });
  
  for (uint32_t c = 1; c < nodes+hedges; ++c) {
    prefix_edges[c] += prefix_edges[c - 1];
  }
  // edges = #edges, hedgecount = how many edges each node has, edges_id: for each node, which ndoes it is connected to
  // edges_data: data for each edge = 1
  graph.constructFrom(nodes+hedges, edges, prefix_edges, edges_id);
  galois::do_all(galois::iterate(graph),
                  [&](GNode n) {
                    if (n < hedges)
                      graph.getData(n).netnum = n+1;
                    else
                      graph.getData(n).netnum = INT_MAX;
                    graph.getData(n).netrand = INT_MAX;
                    graph.getData(n).netval = -2.0;
                    graph.getData(n).nodeid = n+1;
  
  });
  T.stop();
  std::cout<<"time to build a graph "<<T.get()<<"\n";
  graphStat(graph);
  std::cout<<"\n";
  galois::preAlloc(galois::runtime::numPagePoolAllocTotal() * 5);
  galois::reportPageAlloc("MeminfoPre");


  //std::ifstream in("new_feat.txt");
  if (nodefeature) {
  std::ifstream in(features.c_str());
  
  std::string line1;
  unsigned iter = 0;
  GNode c = 0;
  while(std::getline(in, line1)) {
    std::stringstream ss(line1);
    unsigned node;
    //ss >> node;
    node = c + graph.hedges;
    double val;
    iter = 0;
    while(ss >> val) {
      graph.getData(node).EMvec[iter] = val;
      iter++;
    }
    c++;

  }
  galois::do_all(
      galois::iterate((uint64_t)0, graph.hedges),
      [&](GNode item) {
        unsigned size = 0;
        std::vector<double> tmp(dim,0.0);
        for (auto n : graph.edges(item)) {
          size++;
          auto node = graph.getEdgeDst(n);
          for (int i = 0; i < dim; i++) 
            tmp[i] += graph.getData(node).EMvec[i];
        }
          for (int i = 0; i < dim; i++) 
            graph.getData(item).EMvec[i] = tmp[i] / size;
      },
      galois::loopname("initPart"));
  
  }
  Partition(&metisGraph, csize, 2);
  std::ofstream out("embedding.txt");
  for (auto n = graph.hedges; n < graph.size(); n++) {
    for (int i = 0; i < 128; i++)
      if (isnormal(graph.getData(n).EMvec[i]))
        out<<graph.getData(n).EMvec[i]<<"\t";
      else
        out<<0.0<<"\t";
    out<<"\n";
  }

  return 0;
}

