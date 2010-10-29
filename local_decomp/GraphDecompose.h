#ifndef GRAPHDECOMPOSE_H_
#define GRAPHDECOMPOSE_H_


#include <vector> 
#include <bitset> 
#include "Bigram.h"
#include "Graph.h"

using namespace std; 

class GraphDecompose {
 public: 
  vector<int> all_pairs_path[NUMSTATES][NUMSTATES];
  //vector<int> all_pairs_path_length[NUMSTATES][NUMSTATES];

  vector <Bigram> valid_bigrams;
  vector <int> forward_bigrams[NUMSTATES];
  vector <bitset <NUMSTATES> > bigram_bitset[NUMSTATES][NUMSTATES];

  vector <vector <int> >  bigram_pairs[NUMSTATES][NUMSTATES];
  GraphDecompose();
  void decompose( Graph * g);

 private:
    // DP chart - points to the next node on the path
  
  Graph * g;

 private:
  void compute_bigrams();
  void graph_to_all_pairs();
  void reconstruct_path(int n, int n2, vector <vector <int> > & array);
  void all_pairs_to_bigram();
};

#endif
