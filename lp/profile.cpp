#include <Ngram.h>
#include <Vocab.h>
#include <Prob.h>

#include <iostream>
using namespace std;
int main(int argc, char ** argv) {
  Vocab all;
  Ngram lm(all, 3);

  File file(argv[1] , "r", 0);
  lm.read(file);
  VocabIndex context [] = {2,3};
  for (int i =0; i < 200 * 200 * 200; i++) {
    
    lm.wordProb(1, context); 
  }
}
