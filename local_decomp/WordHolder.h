#ifndef WORDHOLDER_H_
#define WORDHOLDER_H_

#define NUMWORDS 300

class WordHolder {
 private:
  void read_word_map(const char * f_name);
 public:     

  int num_words;

  int word_map[NUMWORDS];

  WordHolder(const char * f_name);
};

#endif
