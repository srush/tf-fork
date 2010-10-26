#include <iostream>
#include <fstream>
using namespace std;
#define NUMSTATES 2000

enum NodeType { LEX_NODE, NT_NODE};


class Graph {

  int num_nodes;
  int graph[NUMSTATES][NUMSTATES];
  int node_edges[NUMSTATES];
  
  int word_node[NUMSTATES];
  int edge_node[NUMSTATES];
  
  int final[NUMSTATES];

  Graph(const char * f_name ) {
    read_graph(f_name);
  }


  void read_graph(const char * f_name) {
    ifstream fin(f_name ,ios::in);
    int tree_pos = 0;
    int node_type;
    while (!fin.eof()) {
      fin >> tree_pos;
      fin >> final[tree_pos];
      fin >> node_edges[tree_pos] ;  
      fin >> node_type;
      if (node_type == LEX_NODE) {
        fin >> word_node[tree_pos];
        edge_node[tree_pos]= -1;
      } else {
        fin >> edge_node[tree_pos];
        word_node[tree_pos]= -1;
      }
      for (int j=0; j < node_edges[tree_pos]; j++) {
        fin >> graph[tree_pos][j];
      }
      num_nodes++;
    }
    fin.close();
  }
};
