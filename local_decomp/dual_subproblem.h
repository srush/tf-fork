#ifndef DUALSUB_H_
#define DUALSUB_H_

#include "Bigram.h"
#include "Graph.h"
#include "LMCache.h"
#include "GraphDecompose.h"
class Subproblem {
 public: 
  Bigram cur_best[NUMSTATES];
  float cur_best_score[NUMSTATES];


  int cur_best_bigram[NUMSTATES];
  int cur_best_path_bigram[NUMSTATES];
  float cur_best_score_bigram[NUMSTATES];

  Subproblem(Graph *g, LMCache * lm_in, GraphDecompose * gd_in);
  void update_weights(int u_pos[], float u_values[], int len);
  void solve();
  void solve_bigram();
  vector <int> get_best_nodes_between(int w1, int w2);
  float get_best_bigram_weight(int w1, int w2 );

 private:

  void recompute_bigram_weights();

  void cache_paths(int, int);
  void setup_problems();
  void reconstruct_path(int n1, int n2, int best_split[NUMSTATES][NUMSTATES], vector <int > & array );
  void find_shortest(int n1, int n2,
                               int best_split[NUMSTATES][NUMSTATES], 
                                 float best_split_score[NUMSTATES][NUMSTATES]);


  // Weight management
  bitset <NUMSTATES> update_filter;
  
  float current_weights[NUMSTATES];


  float update_values[NUMSTATES];
  int update_position[NUMSTATES];
  int update_len;
  // PROBLEMS
  
  // The lagragian score associated with a bigram 
  //vector<float> bigram_weights[NUMSTATES][NUMSTATES];

  // current best weight associated with a 
  float bigram_weights[NUMSTATES][NUMSTATES];
  vector <int> bigram_path[NUMSTATES][NUMSTATES];

  vector <vector <bitset <NUMSTATES> > > bigram_cache;
  bool need_to_recompute[NUMSTATES][NUMSTATES];
  
  //Bigram valid_bigrams[NUMSTATES*NUMSTATES];
  
  Graph * graph;
  LMCache * lm;
  GraphDecompose * gd;
};

Subproblem * initialize_subproblem(const char* graph_file, const char* word_file, const char * lm_file );

#endif
