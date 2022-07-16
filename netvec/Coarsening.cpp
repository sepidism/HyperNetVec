#include "bipart.h"
#include "galois/Galois.h"
#include "galois/AtomicHelpers.h"
#include "galois/Reduction.h"
#include "galois/runtime/Profile.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/gstl.h"

#include <iostream>
#include <unordered_set>

int TOTALW;
int LIMIT;
bool FLAG = false;
namespace {

typedef galois::GAccumulator<unsigned> Pcounter;

double computeScore(GGraph& g, GNode n, GNode m) { //compute cosine sim
  double score = 0.0;
  double normN = 0.0;
  double normM = 0.0;
  for (int i = 0; i < dim; i++) {
    double val = g.getData(n).EMvec[i] * g.getData(m).EMvec[i];
    score += val;
    normN += pow(g.getData(n).EMvec[i], 2);
    normM += pow(g.getData(m).EMvec[i], 2);
  }
  
  return (score / (sqrt(normN) * sqrt(normM)));
}

std::vector<double> mergeEmd(GGraph& g, std::vector<GNode> vec) {
  std::vector<double> tmp(dim, 0.0);
  for (auto v : vec) {
    for (int j = 0; j < dim; j++)
      tmp[j] += g.getData(v).EMvec[j];
  }
  for (int i = 0; i < dim; i++)
    tmp[i] /= vec.size();
  
 return tmp;
}

int hash(unsigned val) {
  unsigned long int seed = val * 1103515245 + 12345;
  return((unsigned)(seed/65536) % 32768);
}

void parallelRand(MetisGraph* graph, int iter) {

  GGraph* fineGGraph  = graph->getFinerGraph()->getGraph();
  galois::do_all(
      galois::iterate((uint64_t)0, fineGGraph->hedges),
      [&](GNode item) {
          fineGGraph->getData(item).netrand = hash(fineGGraph->getData(item).netnum);
    },
    galois::loopname("rand"));
}




// hyper edge matching
void parallelHMatchAndCreateNodes(MetisGraph* graph,
                                 int iter, GNodeBag& bag, std::vector<bool>& hedges, std::vector<unsigned>& weight) {
  //parallelPrioRand<matcher>(graph, iter);
  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  parallelRand(graph, iter); 
  //galois::do_all(
   //   galois::iterate((uint64_t)0,fineGGraph->hedges),
    //  [&](GNode item) {
     for (uint64_t item = 0; item < fineGGraph->hedges; item++) {
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          double score = computeScore(*fineGGraph, item, dst);
          if (fineGGraph->getData(dst).netval < score) {
            fineGGraph->getData(dst).netnum.store(fineGGraph->getData(item).netnum);
            fineGGraph->getData(dst).netval = score;
          }
        }
     }
     // },
  //galois::steal(),  galois::loopname("atomicScore"));
  galois::do_all(
      galois::iterate((uint64_t)0, fineGGraph->hedges),
      [&](GNode item) {
            for (auto c : fineGGraph->edges(item)) {
                auto dst = fineGGraph->getEdgeDst(c);
                if (fineGGraph->getData(dst).netval == fineGGraph->getData(item).netval)
                galois::atomicMin(fineGGraph->getData(dst).netrand, fineGGraph->getData(item).netrand.load());
            }  
     },
      galois::steal(),  galois::loopname("secondMin2"));
  galois::do_all(
      galois::iterate((uint64_t)0, fineGGraph->hedges),
      [&](GNode item) {
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          if (fineGGraph->getData(dst).netrand == fineGGraph->getData(item).netrand)
            galois::atomicMin(fineGGraph->getData(dst).netnum, fineGGraph->getData(item).netnum.load());
        }
      },
      galois::steal(),  galois::loopname("secondMin"));
  typedef std::vector<GNode> VecTy;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  std::string name = "phaseI";
  // hyperedge coarsening 
  galois::do_all(
      galois::iterate((uint64_t)0,fineGGraph->hedges),
      [&](GNode item) {
        unsigned id = fineGGraph->getData(item).netnum;
        bool flag = false;
        unsigned nodeid = INT_MAX;
        auto& edges = *edgesThreadLocal.getLocal();
        edges.clear();
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          auto& data = fineGGraph->getData(dst);
          unsigned nid = data.nodeid;
          if (data.isMatched()) {
            flag = true;
            continue;
          }
          if (data.netnum == fineGGraph->getData(item).netnum) {
            edges.push_back(dst);
            nodeid = std::min(nodeid, dst);
          }
          else { 
             flag = true;
          }
        }

        if (!edges.empty()) {
          if (flag && edges.size() == 1) return; 
          fineGGraph->getData(item).setMatched();
          if (flag) hedges[item] = true;
          bag.push(nodeid);
          unsigned ww = 0;
          for (auto pp : edges) {
            ww += fineGGraph->getData(pp).getWeight();
            fineGGraph->getData(pp).setMatched();
            fineGGraph->getData(pp).setParent(nodeid);
            fineGGraph->getData(pp).netnum = fineGGraph->getData(item).netnum;
          }
          fineGGraph->getData(nodeid).EMvec = mergeEmd(*fineGGraph, edges);
          weight[nodeid-fineGGraph->hedges] = ww;
        }
      },
      galois::loopname("phaseI"));
}

void moreCoarse(MetisGraph* graph, int iter, std::vector<unsigned>& weight) {
  
  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  typedef std::set<int> SecTy;
  typedef std::vector<GNode> VecTy;
  GNodeBag bag;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  galois::do_all(
      galois::iterate((uint64_t)0, fineGGraph->hedges),
      [&](GNode item) {
        if (fineGGraph->getData(item).isMatched()) return;
          for (auto c : fineGGraph->edges(item)) {
            auto dst = fineGGraph->getEdgeDst(c);
            if (fineGGraph->getData(dst).isMatched()) 
                fineGGraph->getData(dst).netval = -2.0;
          }
      },
      galois::steal(),  galois::loopname("atomicMin2"));
  galois::do_all( 
      galois::iterate((uint64_t)0, fineGGraph->hedges),
      [&](GNode item) {
          if (fineGGraph->getData(item).isMatched()) return;
          auto& cells = *edgesThreadLocal.getLocal();
          cells.clear();
          int best = INT_MAX;
          GNode b;
          //int w = 0;
          for (auto edge : fineGGraph->edges(item)) {
	        auto e = fineGGraph->getEdgeDst(edge);
              auto& data = fineGGraph->getData(e);
              if (!fineGGraph->getData(e).isMatched()) {
                  if (data.netnum == fineGGraph->getData(item).netnum) {
                      cells.push_back(e);
                  }
              }
              else if (fineGGraph->getData(e).netval == -2.0) {
                  auto nn = fineGGraph->getData(e).getParent();
                  if (fineGGraph->getData(e).getWeight() < best) {
                    best = fineGGraph->getData(e).getWeight();
                    b = e;
                  }
                  else if (fineGGraph->getData(e).getWeight() == best) {
                    if (e < b)
                      b = e;
                  }
              }

          }
          if (cells.size() > 0) {
              if (best < INT_MAX) {
                  auto nn = fineGGraph->getData(b).getParent();
                  for (auto e : cells) {
	            bag.push(e);
                    fineGGraph->getData(e).setMatched();
                    fineGGraph->getData(e).setParent(nn);
                    fineGGraph->getData(e).netnum = fineGGraph->getData(b).netnum;
                    for (int i = 0; i < dim; i++)
                       fineGGraph->getData(nn).EMvec[i] = 
                         (fineGGraph->getData(nn).EMvec[i] + fineGGraph->getData(e).EMvec[i])/2;
                  }
              }
                             
          }        
      },
         galois::steal(),galois::loopname("moreCoarse"));
      for (auto c : bag) {
        auto nn = fineGGraph->getData(c).getParent();
        int ww = weight[nn-fineGGraph->hedges];
        ww += fineGGraph->getData(c).getWeight();
        weight[nn-fineGGraph->hedges] = ww;
      }
}

// Coarsening phaseII
void coarsePhaseII(MetisGraph* graph,
                    int iter, std::vector<bool>& hedges, std::vector<unsigned> & weight) {

  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  typedef std::set<int> SecTy;
  typedef std::vector<GNode> VecTy;
  typedef galois::substrate::PerThreadStorage<SecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalDataV;
  ThreadLocalDataV edgesThreadLocalV;
  std::string name = "CoarseningPhaseII";
  galois::GAccumulator<int> hhedges;
  galois::GAccumulator<int> hnode;
  //moreCoarse(graph, iter, weight);

  galois::do_all( 
      galois::iterate((uint64_t)0, fineGGraph->hedges),
      [&](GNode item) {
        if (fineGGraph->getData(item).isMatched()) return;
        unsigned id = fineGGraph->getData(item).netnum;
        unsigned ids;
        int count = 0;
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          auto& data = fineGGraph->getData(dst);
          if (data.isMatched()) {
            if (count == 0) {
              ids = data.getParent();
              count++;
            }
            else if (ids != data.getParent()) {
              count++;
              break;
            }
          }
          else { 
              count = 0;
              break;
          }
        }
        if (count == 1) {
            fineGGraph->getData(item).setMatched();
          
        }
        else {
           hedges[item] = true;
           fineGGraph->getData(item).setMatched();
         
        }

      },
      galois::loopname("count # Hyperedges"));

   //hedges += hhedges.reduce();
}

void parallelCreateEdges(MetisGraph* graph, GNodeBag& bag, std::vector<bool> hedges, std::vector<unsigned> weight) {

  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  assert(fineGGraph != coarseGGraph);
  galois::GAccumulator<unsigned> hg;
  galois::do_all(
      galois::iterate((uint64_t)0, fineGGraph->hedges),
      [&](GNode n) {
          if (hedges[n])
              hg += 1;
      },
      galois::steal(), galois::loopname("number of hyperedges loop"));
  galois::do_all(
      galois::iterate((uint64_t)fineGGraph->hedges, fineGGraph->size()),
      [&](GNode ii) {
            if (!fineGGraph->getData(ii).isMatched()) { 
              //auto s = std::distance(fineGGraph->edge_begin(ii), fineGGraph->edge_end(ii));
              //if (s < 1 ) return;
              bag.push(ii);
              fineGGraph->getData(ii).setMatched();
              fineGGraph->getData(ii).setParent(ii);
              //fineGGraph->getData(ii).netnum = INT_MAX;
              weight[ii-fineGGraph->hedges] = fineGGraph->getData(ii).getWeight();
            }
          
      },
      galois::steal(), galois::loopname("noedgebag match"));
  unsigned nodes = std::distance(bag.begin(), bag.end());// + numnodes;
  int hnum = hg.reduce();
  unsigned newval = hnum;
  std::vector<unsigned> idmap(fineGGraph->hnodes);
  std::vector<unsigned> newrand(nodes);
  std::vector<unsigned> newWeight(nodes);
  std::set<unsigned> myset;
  galois::StatTimer Tloop("for loop");
  Tloop.start();
  std::vector<unsigned> v;
  for (auto n : bag) v.push_back(n);
  std::sort(v.begin(), v.end());
  for (auto n : v) {
    newrand[newval-hnum] = n;
    idmap[n-fineGGraph->hedges] = newval++;
    newWeight[idmap[n-fineGGraph->hedges]-hnum] = weight[n-fineGGraph->hedges];
  }
  galois::do_all(
      galois::iterate((uint64_t)fineGGraph->hedges, fineGGraph->size()),
      [&](GNode n) {
        unsigned id = fineGGraph->getData(n).getParent();
        fineGGraph->getData(n).setParent(idmap[id-fineGGraph->hedges]);
      },
      galois::steal(), galois::loopname("first loop"));
  Tloop.stop();
  //std::cout<<"total first loop "<<Tloop.get()<<"\n";

  uint32_t num_nodes_next = nodes + hnum;
  uint64_t num_edges_next; 
  galois::gstl::Vector<galois::PODResizeableArray<uint32_t>> edges_id(num_nodes_next);
  std::vector<unsigned> old_id(hnum);
//  std::vector<unsigned> old_label(hnum);
  unsigned h_id = 0;
  //galois::StatTimer sloop("for loop II");
  //sloop.start();
  for (GNode n = 0; n < fineGGraph->hedges; n++) {
    if (hedges[n]) {
       old_id[h_id] = fineGGraph->getData(n).netnum;
  //     old_label[h_id] = fineGGraph->getData(n).label;
       fineGGraph->getData(n).nodeid = h_id++;
    }
  }
  galois::do_all(galois::iterate((uint64_t)0, fineGGraph->hedges),
                [&](GNode n) {
                    if (!hedges[n]) return;
                        auto data = fineGGraph->getData(n, flag_no_lock);
                        unsigned id =  fineGGraph->getData(n).nodeid;
           
                    for (auto ii : fineGGraph->edges(n)) { 
                        GNode dst = fineGGraph->getEdgeDst(ii);
                        auto dst_data = fineGGraph->getData(dst, flag_no_lock);
                          unsigned pid = dst_data.getParent();
                          auto f = std::find(edges_id[id].begin(), edges_id[id].end(), pid);
                          if (f == edges_id[id].end()) {
                            edges_id[id].push_back(pid);
                          }
                    } // End edge loop
                }, galois::steal(),
                   galois::loopname("BuildGrah: Find edges"));
  for (auto n = 0; n < hnum; n++) {
      auto& tmp = edges_id[n];
      for (auto m : tmp) { 
          edges_id[m].push_back(n);
      }
  }

  std::vector<uint64_t> prefix_edges(num_nodes_next);
  galois::GAccumulator<uint64_t> num_edges_acc;
  galois::do_all(galois::iterate((uint32_t)0, num_nodes_next),
                [&](uint32_t c){
                  prefix_edges[c] = edges_id[c].size();
                  num_edges_acc += prefix_edges[c];
                }, galois::steal(),
                   galois::loopname("BuildGrah: Prefix sum"));

  num_edges_next = num_edges_acc.reduce();
  for (uint32_t c = 1; c < num_nodes_next; ++c) {
    prefix_edges[c] += prefix_edges[c - 1];
  }
  //galois::StatTimer TimerConstructFrom("Timer_Construct_From");
  //TimerConstructFrom.start();
  coarseGGraph->constructFrom(num_nodes_next, num_edges_next, prefix_edges, edges_id);
  coarseGGraph->hedges = hnum;
  coarseGGraph->hnodes = nodes;
  galois::do_all(
      galois::iterate(*coarseGGraph),
      [&](GNode ii) {
        if (ii < hnum) {
          coarseGGraph->getData(ii).netnum = old_id[ii]; 

        } 
        else {
            coarseGGraph->getData(ii).netval = -2.0;
            coarseGGraph->getData(ii).netnum = INT_MAX;
            coarseGGraph->getData(ii).netrand = INT_MAX;
            coarseGGraph->getData(ii).nodeid = ii;//fineGGraph->getData(id).nodeid;
            coarseGGraph->getData(ii).setWeight(newWeight[ii-coarseGGraph->hedges]);
            coarseGGraph->getData(ii).EMvec = fineGGraph->getData(newrand[ii-hnum]).EMvec;
        }
      },
      galois::steal(), galois::loopname("noedgebag match"));
}


void findMatching(MetisGraph* coarseMetisGraph,
                      
                       int iter) {
  MetisGraph* fineMetisGraph = coarseMetisGraph->getFinerGraph();
  GNodeBag nodes;
  int sz = coarseMetisGraph->getFinerGraph()->getGraph()->hedges;
  std::vector<bool> hedges(sz, false);
  //for (int i = 0; i < sz; i++) hedges[i] = false; 
  std::vector<unsigned> weight(fineMetisGraph->getGraph()->hnodes);
  
        parallelHMatchAndCreateNodes(coarseMetisGraph,
                                            iter, nodes, hedges, weight);
       coarsePhaseII(coarseMetisGraph, iter, hedges, weight);
       parallelCreateEdges(coarseMetisGraph, nodes, hedges, weight);
}

MetisGraph* coarsenOnce(MetisGraph* fineMetisGraph, 
                         int iter) {
  MetisGraph* coarseMetisGraph = new MetisGraph(fineMetisGraph);
  findMatching(coarseMetisGraph, iter);
  return coarseMetisGraph;
}

} // namespace

MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo) {

  MetisGraph* coarseGraph = fineMetisGraph;
  unsigned size           = fineMetisGraph->getGraph()->hnodes;//, fineMetisGraph->getGraph()->cellList().end());
  unsigned hedgeSize = 0;
  const float ratio = 55.0 / 45.0;  // change if needed
  const float tol = std::max(ratio, 1 - ratio) - 1;
  const int hi = (1 + tol) * size / (2 + tol);
  const int lo = size - hi;
  LIMIT = hi / 4;
  int totw = 0;
  
  //std::cout<<"inital weight is "<<totw<<"\n";
  unsigned Size = size;
  unsigned iterNum        = 0;
  unsigned newSize = size;
  while (Size > 1000 && iterNum < coarsenTo) { 
    if (iterNum >= coarsenTo) break;
    if (Size - newSize <= 20 && iterNum > 2) break; //final
     newSize = coarseGraph->getGraph()->hnodes;
     coarseGraph      = coarsenOnce(coarseGraph, iterNum);
     Size = coarseGraph->getGraph()->hnodes;
     hedgeSize = coarseGraph->getGraph()->hedges; 
     std::cout<<"nodes "<<coarseGraph->getGraph()->hnodes<<" and hyperedges "<<hedgeSize<<"\n";
     if (hedgeSize < 1000) return coarseGraph->getFinerGraph();
     
    ++iterNum;
    
  }
  return coarseGraph;
}
