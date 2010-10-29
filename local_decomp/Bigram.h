#ifndef BIGRAM_H_
#define BIGRAM_H_

struct Bigram{ 
  int w1;
  int w2;
  Bigram(int word1,int word2): w1(word1), w2(word2) {}
  Bigram(){}
};

#endif
