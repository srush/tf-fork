#include <fstream>
#include <iostream>

#include "WordHolder.h"
using namespace std;
#define NUMWORDS 300

WordHolder::WordHolder(const char * f_name) {
  num_words =0;
  read_word_map(f_name);

}

void WordHolder::read_word_map(const char * f_name) {
    ifstream fin(f_name, ios::in);
    while (!fin.eof() ) {
      int word_ind, vocab_ind;
      fin >> word_ind >> vocab_ind;
      word_map[word_ind] = vocab_ind;
      if (!fin.eof())
        num_words++;
    }
    fin.close();
}
    


