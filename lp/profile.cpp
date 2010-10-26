#include <Ngram.h>
#include <Vocab.h>
#include <Prob.h>

#include <iostream>
#include <iomanip>
using namespace std;
int main(int argc, char ** argv) {
  Vocab all;
  Ngram lm(all, 3);

  File file(argv[1] , "r", 0);
  lm.read(file);
  
  cout << "Checking " << endl;
  double best = 0.0;
  for (int i =0; i < 300; i++) {
    VocabIndex word = i;
    for (int j =0; j < 300; j++) {
      for (int k =0; k < 300; k++) {
        VocabIndex context [] = {j, k, Vocab_None};
        double m = lm.wordProb(word, context); 
        if (m > best) {
          best = m;
        }
      }
    }
  }
  cout << "Done " << endl;
}
