#include <Ngram.h>
#include <Vocab.h>
#include <Prob.h>

class LMCache {

  Vocab all;
  Ngram * lm;
  
  // LOG PROB cache of words
  float all_score[NUMWORDS][NUMWORDS][NUMWORDS];

  LMCache(const char * lm_file ) {
    lm = new LM(all, 3);
    
    File file(lm_file, "r", 0);
    lm.read(file);
  
  }
  
  
  void cache_sentence_probs(WordHolder words) {
    for (int i = 0; i < words.num_words; i++) {
      for (int j = 0; j < words.num_words; j++) {
        VocabIndex context [] = {words.word_map[j], words.word_map[i], Vocab_None};

        for (int k = 0; k < words.num_words; k++) {
          if (k ==0) {
            all_score[i][j][k] = lm.wordProb(words.word_map[k], context); 
          } else {
            all_score[i][j][k] = lm.wordProbRecompute(words.word_map[k], context); 
          }
        }
      }
    }
  }
  

}
