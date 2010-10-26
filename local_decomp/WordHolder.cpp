#include <fstream>
#include <iostream>
using namespace std;

class WordHolder {
    int num_words = 0;
    int word_map[NUMWORDS];
    
    WordHolder(const char * f_name) {
      read_word_map(f_name);
    }

    void read_word_map(char * f_name) {
      ifstream fin(f_name, ios::in);
      while (!fin.eof() ) {
        int word_ind, vocab_ind;
        fin >> word_ind >> vocab_ind;
        word_map[word_ind] = vocab_ind;
        num_words++;
      }
      fin.close();
    }
    

  }
}
