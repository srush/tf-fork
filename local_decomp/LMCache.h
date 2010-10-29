
#include "WordHolder.h"

#include <Ngram.h>
#include <Vocab.h>
#include <Prob.h>

#ifndef LMCACHE_H_
#define LMCACHE_H_


class LMCache {
 private:
  Vocab all;
  Ngram * lm;

 public:
    float all_score[NUMWORDS][NUMWORDS][NUMWORDS];
    float all_score_bi[NUMWORDS][NUMWORDS];
    LMCache(const char * lm_file);
    void cache_sentence_probs(const WordHolder & words);
};


#endif
