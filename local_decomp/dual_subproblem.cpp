#include <time.h>
#include "LMCache.h"
#include "WordHolder.h"
#include "Graph.h"
#include "GraphDecompose.h"
#include "dual_subproblem.h"
#define INF 1000000

#include <bitset>

using namespace std;


double diffclock(clock_t clock1,clock_t clock2)
{
  double diffticks=clock1-clock2;
  double diffms=(diffticks*1000)/CLOCKS_PER_SEC;
  return diffms;
}

Subproblem::Subproblem(Graph * g, LMCache *lm_in, GraphDecompose * gd_in) {
  update_filter.reset();
  for (int i=0; i < NUMSTATES; i++) {
    current_weights[i] = 0;
  }
  graph = g;
  lm = lm_in;
  gd = gd_in;

  bigram_cache.resize(graph->num_nodes);
  for (unsigned int i=0; i < graph->num_nodes; i++) {
    bigram_cache[i].resize(NUMSTATES);
  }
  for (unsigned int i=0; i< gd->valid_bigrams.size() ;i++) {
    Bigram b = gd->valid_bigrams[i];
    //assert(gd->bigram_pairs[b.w1][b.w2].size() > 0);
    //bigram_weights[b.w1][b.w2].resize(gd->bigram_pairs[b.w1][b.w2].size());
    //for (unsigned int path=0; path < gd->bigram_pairs[b.w1][b.w2].size(); path++) {
    //bigram_weights[b.w1][b.w2][path] = 0; 
    //}
    cache_paths(b.w1, b.w2);
  }

  recompute_bigram_weights();
}

void Subproblem::update_weights(int u_pos[NUMSTATES], float u_values[NUMSTATES], int len) {
  clock_t begin=clock();
  //update_filter.reset();
  for (int i=0; i< NUMSTATES; i++) {
    //update_values[i] = 0.0;
  }
  update_len = len;
  for (int i=0; i< len; i++) {
    int pos = u_pos[i];
    update_values[i] = u_values[i];
    update_position[i] = u_pos[i];
    current_weights[pos] += u_values[i];
    //update_filter[pos] = true; 
  }
  clock_t end=clock();
  cout << "Weight Update: " << double(diffclock(end,begin)) << " ms"<< endl;
  recompute_bigram_weights();

}



void Subproblem::cache_paths(int n1, int n2) {  
  assert(!bigram_cache[n1][n2].any());
  int s = gd->all_pairs_path[n1][n2].size();
  for (int split=0; split < s; split++) {
    int k = gd->all_pairs_path[n1][n2][split];

    if (k == n2) {
      bigram_cache[n1][n2][n2] = true;
    } else {

      if (!bigram_cache[n1][k].any()) {
        cache_paths(n1, k);
      } 

      if (!bigram_cache[k][n2].any()) {
        cache_paths(k, n2);
      } 
      bigram_cache[n1][n2] = bigram_cache[n1][k] | bigram_cache[k][n2]; 
    }
  }
  assert(bigram_cache[n1][n2].any());
}

void Subproblem::reconstruct_path(int n1, int n2, int best_split[NUMSTATES][NUMSTATES], vector <int > & array ) {
  int k = best_split[n1][n2];
  if (k != n2) {
    
    reconstruct_path(n1, k, best_split, array);
    reconstruct_path(k, n2, best_split, array);
    
  } else {
    array.push_back(k);
  }

}

void Subproblem::find_shortest(int n1, int n2,
                               int best_split[NUMSTATES][NUMSTATES], 
                               float best_split_score[NUMSTATES][NUMSTATES]) {
  assert (need_to_recompute[n1][n2]);
  bool has_update = false;
  for (int i=0; i < update_len; i++) {
    break;
    if (bigram_cache[n1][n2][update_position[i]]) {
      has_update = true;
      break;
    }
  }
  
  need_to_recompute[n1][n2] =0;

  if (update_len != 0 && !has_update ) {    
    //cout << "skip" << endl;
    //return;
  }

  //assert(best_split_score[n1][n2] == 0 || update_len != 0);
  best_split_score[n1][n2] = INF;
  int s = gd->all_pairs_path[n1][n2].size();
  for (int split=0; split < s; split++) {
    int k = gd->all_pairs_path[n1][n2][split];
    
    if (k == n2) {
      //base case
      float val =  current_weights[n2];
      if (val < best_split_score[n1][n2]) {
        best_split[n1][n2] = k;
        best_split_score[n1][n2] = val;
      }
    } else {
      float a, b;
      if (need_to_recompute[n1][k]) {
        find_shortest(n1, k, best_split, best_split_score);      
      } 
      a = best_split_score[n1][k];
      
      if (need_to_recompute[k][n2]) {
        find_shortest(k, n2, best_split, best_split_score);      
      } 
      b = best_split_score[k][n2];
      
      if (a + b < best_split_score[n1][n2]) {
        best_split[n1][n2] = k;
        best_split_score[n1][n2] = a+b;
        //array.clear();
        //array.insert(array.end(), one.begin(), one.end());
        //array.insert(array.end(), two.begin(), two.end());
      }
    }
  }
}

int best_split[NUMSTATES][NUMSTATES]; 
float best_split_score[NUMSTATES][NUMSTATES]; 

void Subproblem::recompute_bigram_weights() {
  
  clock_t begin=clock();

  for (int i=0;i < graph->num_nodes; i++) {
    for (int j=0;j<graph->num_nodes; j++) {
      //best_split_score[i][j] =0;
      need_to_recompute[i][j] = 1;
    }
  }
  for (unsigned int i=0; i< gd->valid_bigrams.size() ;i++) {
    Bigram b = gd->valid_bigrams[i];
    int w1 = b.w1;
    int w2 = b.w2;
    
    // Basically we need to find the best path between w1 and w2
    // using only gd->all_pairs_path
    
    
    // first do it with recursion
    find_shortest(w1, w2, best_split, best_split_score);
    bigram_weights[w1][w2] = best_split_score[w1][w2]; 
    bigram_path[w1][w2].clear();
    reconstruct_path(w1, w2, best_split, bigram_path[w1][w2]);
    /*
    int l = gd->bigram_pairs[w1][w2].size();
    for (unsigned int path=0; path < l; path++) {
      // updates are sparse, so first try intersection
      //bitset <NUMSTATES> inter = gd->bigram_bitset[b.w1][b.w2][path] & update_filter;      
      
      //if (inter.any()) {
      int l2 = gd->bigram_pairs[w1][w2][path].size();
      float total = 0;
      for (unsigned int k=0; k < l2; k++) {
         total += update_values[gd->bigram_pairs[w1][w2][path][k]];
      }
      bigram_weights[w1][w2][path] += total;
        //}
    }
    */
  }

  clock_t end=clock();
  cout << "Weight Update: " << double(diffclock(end,begin)) << " ms"<< endl;
  
}

void Subproblem::setup_problems() {
  //gd->decompose(graph);   
}


vector <int> Subproblem::get_best_nodes_between(int w1, int w2) {
  //assert(path < gd->bigram_pairs[w1][w2].size());
  return bigram_path[w1][w2];
}

float Subproblem::get_best_bigram_weight(int w1, int w2) {
  return bigram_weights[w1][w2];
}

void Subproblem::solve_bigram() {
  assert(graph->num_nodes > 10);
  for (unsigned int i =0; i< graph->num_nodes; i++ ) {
    cur_best_score_bigram[i] = INF;
    cur_best_bigram[i] = -1;
    //cur_best_path_bigram[i] = -1;
  }

  clock_t begin=clock();
  assert(gd->valid_bigrams.size() > 0);
  for (unsigned int i =0; i< gd->valid_bigrams.size(); i++ ) {
    Bigram b = gd->valid_bigrams[i];

    int word1 = graph->word_node[b.w1];
    int word2 = graph->word_node[b.w2];

    float lm_score = lm->all_score_bi[word1][word2];
    
    


    //cout << b.w1 << " " << b.w2 << " " << lm->all_score_bi[word1][word2] <<endl;    
    float score1 = bigram_weights[b.w1][b.w2]; 
    
    float score = score1 + (-0.141221) * lm_score;  
      
    if (cur_best_score_bigram[b.w1] == INF || score < cur_best_score_bigram[b.w1]) {
      cur_best_score_bigram[b.w1] = score;
      cur_best_bigram[b.w1] = b.w2;
      //cur_best_path_bigram[b.w1] = path;
    }
  }
  
  clock_t end=clock();
  cout << "Time elapsed: " << double(diffclock(end,begin)) << " ms"<< endl;

  
  for (unsigned int i =0; i< graph->num_nodes; i++ ) {
    if (graph->word_node[i]!= -1 && graph->final[i]!= 1) {
      //cout << i << endl;
      assert(cur_best_score_bigram[i] != INF);
      assert(cur_best_bigram[i] != -1);
      //assert(cur_best_path_bigram[i] != -1);
      assert(graph->word_node[cur_best_bigram[i]] != -1);
    }
  }
  
}

void Subproblem::solve() {
  // for (unsigned int i =0; i< gd->valid_bigrams.size(); i++ ) {
  //   Bigram b = gd->valid_bigrams[i];
  //   int word1 = graph->word_node[b.w1];
  //   int word2 = graph->word_node[b.w2];
    
  //   float score1 = bigram_weights[b.w1][b.w2]; 
  //   for (unsigned int j =0; j < gd->forward_bigrams[b.w2].size(); j++) {
  //     int w3 = gd->forward_bigrams[b.w2][j];
      
  //     int word3 = graph->word_node[w3];
  //     float score2 = bigram_weights[b.w2][w3];
      
  //     float score = score1 + score2 + lm->all_score[word1][word2][word3];
      
  //     if (score > cur_best_score[b.w1]) {
  //       cur_best_score[b.w1] = score;
  //         cur_best[b.w1] = Bigram(b.w2, w3);
  //     }
  //   }     
  // }
}



Subproblem * initialize_subproblem(const char* graph_file, const char* word_file, const char * lm_file ) {
  LMCache * lm = new LMCache(lm_file);
  WordHolder * wd = new WordHolder(word_file);
  Graph * g = new Graph(graph_file);
  lm->cache_sentence_probs(*wd);
  GraphDecompose * gd = new GraphDecompose();
  gd->decompose(g);
  
  Subproblem * s = new Subproblem(g, lm, gd);
  return s;
}
