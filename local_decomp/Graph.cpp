#include <iostream>
#include <fstream>

#include "Graph.h"
using namespace std;


enum NodeType { LEX_NODE, NT_NODE};


Graph::Graph(const char * f_name ) {
  for (int i=0; i< NUMSTATES; i++) {
    node_edges[i] = 0;
    final[i] =0;
    word_node[i]=-1;
    edge_node[i]=-1;
  }

  read_graph(f_name);
}


void Graph::read_graph(const char * f_name) {
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

    num_nodes = max(tree_pos, num_nodes)+1;

  }
  fin.close();
  cout << num_nodes;
}
