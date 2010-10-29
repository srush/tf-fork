#include <cpptest.h>
#include <string>
#include <cpptest-suite.h>
#include "LMCache.h"
#include "WordHolder.h"
#include "Graph.h"
#include "GraphDecompose.h"
#include "dual_subproblem.h"
#include <sstream>
#include <iostream>
#include <algorithm>
using namespace Test;
using namespace std;

class LocalTestSuite : public Test::Suite
{
public:
  
  LocalTestSuite() {
    TEST_ADD(LocalTestSuite::subproblem_test);    
    TEST_ADD(LocalTestSuite::wordholder_test);
    TEST_ADD(LocalTestSuite::lm_cache_test);
    TEST_ADD(LocalTestSuite::graph_test);
    TEST_ADD(LocalTestSuite::graph_decomp_test);
    //TEST_ADD(LocalTestSuite::load_subproblem_test);

    lm = new LMCache("test_data/lm.3.sri");
    wh = new WordHolder("test_data/words");
    g = new Graph("test_data/graph");
    gd = new GraphDecompose();
    gd->decompose(g);
  
    s = new Subproblem(g, lm, gd);
    

  }
  
private:
  LMCache * lm;
  WordHolder * wh;
  Graph * g;
  GraphDecompose * gd;

  Subproblem *s;

  void wordholder_test() {
    stringstream ss;
    ss << wh->num_words;
    TEST_ASSERT_MSG(wh->num_words == 49, ss.str().c_str())
    return;

  }

  void lm_cache_test() {
    lm->cache_sentence_probs(*wh);
  }

  void graph_test() {
    stringstream ss;
    ss << g->num_nodes;
    TEST_ASSERT_MSG(g->num_nodes == 112, ss.str().c_str());
      
    int total =0, word=0, node =0;
    for (int i = 0; i < NUMSTATES;i++) {
      if (i != 110) {
        TEST_ASSERT(!g->final[i]);
      } else {
        TEST_ASSERT(g->final[i]);
      }

      // Has to be either a word or a otherwise
      if (g->node_edges[i] > 0) { 
        if (g->word_node[i] == -1) { 
          TEST_ASSERT(g->edge_node[i] != -1);
          node++;
        } else {
          TEST_ASSERT(g->word_node[i] != -1);
          word++;
        }
        total++;
      }
    }
    ss.str("");
    ss << total;
    //TEST_ASSERT_MSG(total == g->num_nodes, ss.str().c_str() );
    TEST_ASSERT(word > wh->num_words);
    return;
  }

  void graph_decomp_test() {
    TEST_ASSERT(gd->valid_bigrams.size()>10);
    for (unsigned int i=0; i < gd->valid_bigrams.size();i++) {
      Bigram b = gd->valid_bigrams[i];
      TEST_ASSERT(g->word_node[b.w1] != -1);
      TEST_ASSERT(g->word_node[b.w1] < wh->num_words);
      
      TEST_ASSERT(g->word_node[b.w2] != -1);
      TEST_ASSERT(g->word_node[b.w2] < wh->num_words);
      
      //TEST_ASSERT(gd->all_pairs_path_length[b.w1][b.w2] < 100);
      
      TEST_ASSERT(find(gd->forward_bigrams[b.w1].begin(), gd->forward_bigrams[b.w1].end(), b.w2) != 
                  gd->forward_bigrams[b.w1].end());
      
      for (int path =0; path <  gd->bigram_bitset[b.w1][b.w2].size(); path++ ){ 
        TEST_ASSERT(gd->bigram_bitset[b.w1][b.w2][path].count()>0); 
    
        //cout <<gd->bigram_bitset[b.w1][b.w2].count() << " " <<gd->bigram_pairs[b.w1][b.w2].size() << endl;
        TEST_ASSERT(gd->bigram_bitset[b.w1][b.w2][path].count() == gd->bigram_pairs[b.w1][b.w2][path].size());
      }
    }
  }

  void subproblem_test() {
    s->solve_bigram();
    int u_pos[1];
    float u_values[1];
    
    //s->update_weights(u_pos, u_values, 0)

    for (unsigned int i =0; i< g->num_nodes; i++ ) {
      if (g->word_node[i]!= -1) {
        TEST_ASSERT(s->cur_best_bigram[i] != -1);
        TEST_ASSERT(g->word_node[s->cur_best_bigram[i]] != -1);
      }
    }

  }
  
  void load_subproblem_test() {
    //Subproblem * s = initialize_subproblem("/tmp/graph", "/tmp/words", "test_data/lm.3.sri");
  }
  
};


int main(int argc, const char * argv[]) {
  Test::TextOutput output(Test::TextOutput::Verbose);
  LocalTestSuite ets;
  return ets.run(output) ;
}
