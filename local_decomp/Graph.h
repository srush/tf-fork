#ifndef GRAPH_H_
#define GRAPH_H_


#define NUMSTATES 2000

class Graph {
 public:
  int num_nodes;
  int graph[NUMSTATES][NUMSTATES];
  int node_edges[NUMSTATES];
  
  int word_node[NUMSTATES];
  int edge_node[NUMSTATES];
  
  int final[NUMSTATES];

  Graph(const char * f_name );
 private:
  void read_graph(const char * f_name);
};


#endif
