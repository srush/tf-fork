#include "Graph.h"
#include "Bigram.h"
#include <vector>
#include <bitset>
#include <assert.h>
using namespace std;

#define INF 1000000

class GraphDecompose {
  // DP chart - points to the next node on the path
  int all_pairs_path[NUMSTATES][NUMSTATES];
  int all_pairs_path_length[NUMSTATES][NUMSTATES];
  
  vector <int> bigram_pairs[NUMSTATES][NUMSTATES];
  
  //int bigram_pairs_length[NUMSTATES][NUMSTATES];

  vector <Bigram> valid_bigrams;
  vector <int> forward_bigrams[NUMSTATES];
  bitset <NUMSTATES> bigram_bitset[NUMSTATES][NUMSTATES];
  
  Graph * g;

  GraphDecompose() {
    
  }
  
  
  void compute_bigrams() {
    for (int n; n < g->num_nodes; n++ ) {
      if (g->word_node[n] != -1) continue; 
      for (int n2; n2 < g->num_nodes; n2++ ) {
        if (g->word_node[n2] != -1) continue; 
        if (all_pairs_path_length[n][n2] != INF) { 
          valid_bigrams.push_back(Bigram(n,n2));
          forward_bigrams[n].push_back(n2);
          for (int i=0; i < bigram_pairs[n][n2].size() ; i ++) {
            bigram_bitset[n][n2][bigram_pairs[n][n2][i]] = 1;
          }
        }
      }   
    }
  }

  void decompose(const Graph & g) {
    graph_to_all_pairs();
    all_pairs_to_bigram();
    compute_bigrams();
  }

  void graph_to_all_pairs() {

    // INITIALIZE DP CHART
    for (int n=0;n < g->num_nodes;n++) {

      // Initialize all INF
      for (int n2=0;n2< g->num_nodes;n2++) {
        all_pairs_path_length[n][n2] = INF;
      }

      // If path exists set it to 1
      for (int j=0;j < g->node_edges[n]; j++) {
        int n2 = g->graph[n][j];
        all_pairs_path_length[n][n2] = 1;
        // back pointer (split path)
        all_pairs_path[n][n2] = n;
      }
    }
    
    

    // RUN Modified FLOYD WARSHALL
    for (int k=0;k < g->num_nodes; k++) {
      
      // lex nodes can't be in the middle of a path
      if (g->word_node[k]!= -1) continue;
      
      for (int n=0; n < g->num_nodes; n++) {
        if (all_pairs_path_length[n][k] == INF) continue;
      
        for (int n2=0; n2 < g->num_nodes; n2++) {
          if (all_pairs_path_length[n][k] + all_pairs_path_length[k][n2] < all_pairs_path_length[n][n2]) {
            
            assert(all_pairs_path_length[n][k] != INF);
            assert(all_pairs_path_length[k][n2] != INF);
            
            all_pairs_path_length[n][n2] = all_pairs_path_length[n][k] + all_pairs_path_length[k][n2];

            assert(k != n);

            all_pairs_path[n][n2] = k;
          }
        }
      }
    }
  }
  
  
  void reconstruct_path(int n, int n2, vector <int> & array) {
    // find the path between nodes n and n2, and fill in array
    int k = all_pairs_path[n][n2];
    assert(k != n2);
    array.push_back(k);
  
    if (k != n){
      reconstruct_path(n, k, array);
      reconstruct_path(k, n2, array);
    }
  }
  
  void all_pairs_to_bigram() {

    for (int n=0; n < g->num_nodes; n++) {
      if (g->word_node[n]==-1) continue;
      for (int n2=0; n2 < g->num_nodes; n2++) {
        if (g->word_node[n2]==-1) continue;

        if (all_pairs_path_length[n][n2] == INF) continue;
        
        int node;       

        reconstruct_path(n, n2, bigram_pairs[n][n2]);
      }
    }
  }
};
