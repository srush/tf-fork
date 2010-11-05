#ifndef BIGRAMRESCORE_H_
#define BIGRAMRESCORE_H_

#include "Graph.h"
#include "LMCache.h"
#include "GraphDecompose.h"

class BigramRescore {

 public:   

  BigramRescore(Graph * graph_in, GraphDecompose * gd_in);
  void update_weights(int u_pos[], float u_values[], int len);
  float bigram_weights[NUMSTATES][NUMSTATES];

  float current_weights[NUMSTATES];
  vector <int> bigram_path[NUMSTATES][NUMSTATES];

  //float update_values[NUMSTATES];
  int update_position[NUMSTATES];
  int update_len;
  bitset <NUMSTATES> update_filter;
  void recompute_bigram_weights(bool init);
  int move_direction[NUMSTATES][NUMSTATES]; 

 private:
  bool need_to_recompute[NUMSTATES][NUMSTATES];
  vector <Bigram> for_updates[NUMSTATES];
  
  int best_split[NUMSTATES][NUMSTATES]; 
  float best_split_score[NUMSTATES][NUMSTATES]; 
  
  
  void cache_paths(int, int);
  void cache_forward();
  vector<int> forward_paths[NUMSTATES];
  vector<int> backward_paths[NUMSTATES];
  
  void setup_problems();
  void reconstruct_path(int n1, int n2, int best_split[NUMSTATES][NUMSTATES], vector <int > & array );
  void find_shortest(int n1, int n2,
                               int best_split[NUMSTATES][NUMSTATES], 
                                 float best_split_score[NUMSTATES][NUMSTATES]);

  GraphDecompose * gd;  
  Graph * graph;  
  vector <vector <bitset <NUMSTATES> > > bigram_cache;

};
#endif
