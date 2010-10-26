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
};
