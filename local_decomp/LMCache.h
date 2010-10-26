
#include "WordHolder.h"

class LMCache {
 public:
    float all_score[NUMWORDS][NUMWORDS][NUMWORDS];
    LMCache(const char * lm_file);
    
    cache_sentence_probs(WordHolder words);
}
