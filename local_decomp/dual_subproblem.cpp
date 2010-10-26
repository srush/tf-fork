#include <bitset>
using namespace std;





class Subproblem {

  // Weight management
  bitset <NUMSTATES> update_filter;
  float update_values[NUMSTATES];
  float current_weights[NUMSTATES];


  // PROBLEMS
  
  // The lagragian score associated with a bigram 
  float bigram_weights[NUMSTATES][NUMSTATES];

  Bigram valid_bigrams[NUMSTATES*NUMSTATES];

  // The best forward bigram at a state
  Bigram cur_best[NUMSTATES];
  float cur_best_score[NUMSTATES];
  
  
  Graph * graph;

  Subproblem(Graph * g, LMCache lm) {
    update_filter.reset();
    for (int i=0; i< NUMSTATES; i++) {
      update_filter[i] = 0;
      current_weights[i] = 0;
    }
    graph = g;
  }

  void update_weights(float u_values[NUMSTATES]) {
    update_filter.reset();
    for (int i=0; i< NUMSTATES; i++) {
      if (u_values[i] != 0) {
        update_values[i] = u_values[i];
        current_weights[i] += u_values[i];
        update_filter[i] = true; 
      }
    }
    recompute_bigram_weights();
  }

  void recompute_bigram_weights() {
    for (int i=0; i< gd.valid_bigrams.length() ;i++) {
      Bigram b = gd.valid_bigrams[i];
      
      // updates are sparse, so first try intersection
      Bitset <NUMSTATES> inter = gd.bigrams_bitset[b.w1][b.w2] & update_filter;      
      
      
      if (inter.any()) {
        for (int j=0; j < gd.bigram_pairs[b.w1][b.w2].length(); j++) {
          bigram_weights[b.w1][b.w2] += update_values[j];
        }
      }
    }
  }

  void setup_problems() {
    GraphDecompose gd;

    gd.decompose(graph);    


  }

  void solve() {
    for (int i =0; i< g.valid_bigrams.length(); i++ ) {
      Bigram b = gd.valid_bigrams[i];
      int word1 = g.word_node[b.w1];
      int word2 = g.word_node[b.w2];

      float score1 = bigram_weights[b.w1][b.w2]; 
      for (int j =0; j < gd.forward_bigrams[b.w2]; j++) {
        int w3 = gd.forward_bigrams[b.w2][j];

        int word3 = g.word_node[w3];
        float score2 = bigram_weights[b.w2][w3];
        
        float score = score1 + score2 + lm.all_score[word1][word2][word3];
        
        if (score > cur_best_score[b.w1]) {
          cur_best_score[b.w1] = score;
          cur_best[b.w1] = Bigram(b.w2, w3);
        }
      }     
    }
  }
}
