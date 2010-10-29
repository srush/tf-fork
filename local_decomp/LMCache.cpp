#include <Ngram.h>
#include <Vocab.h>
#include <Prob.h>

#include "WordHolder.h"
#include "LMCache.h"


LMCache::LMCache(const char * lm_file ) {
  lm = new Ngram(all, 3);
  
  File file(lm_file, "r", 0);
  lm->read(file);
    
}
  
  
void LMCache::cache_sentence_probs(const WordHolder & words) {
  for (int i = 0; i < words.num_words; i++) {
    for (int j = 0; j < words.num_words; j++) {
      VocabIndex context [] = {words.word_map[j], words.word_map[i], Vocab_None};
      VocabIndex context2 [] = {words.word_map[j], Vocab_None};
      
      for (int k = 0; k < words.num_words; k++) {
        if (k ==0) {
          all_score[i][j][k] = lm->wordProb(words.word_map[k], context); 
          all_score_bi[j][k] = lm->wordProb(words.word_map[k], context2);

        } else {
          all_score[i][j][k] = lm->wordProbRecompute(words.word_map[k], context); 
          all_score_bi[j][k] = lm->wordProbRecompute(words.word_map[k], context2);
        }
      }
    }
  }
}


