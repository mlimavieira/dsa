# Graph Algorithm Templates

This document contains essential graph problem-solving templates using **DFS**, **BFS**, and **Disjoint Set Union (DSU)**. Each template includes a description of when to use it and the key logic structure.

---

## ✅ DFS Templates

### 1. Basic DFS Traversal  
**When to use:** Traversing graphs, connected components, simple path existence.  
**LeetCode Example:** [Number of Provinces - LeetCode #547](https://leetcode.com/problems/number-of-provinces/)
**When to use:** Traversing graphs, connected components, simple path existence.

```java
void dfs(int node, Set<Integer> visited, Map<Integer, List<Integer>> graph) {
    visited.add(node);
    for (int neighbor : graph.getOrDefault(node, List.of())) {
        if (!visited.contains(neighbor)) {
            dfs(neighbor, visited, graph);
        }
    }
}
```

---

### 2. DFS with Cycle Detection / Backtracking  
**When to use:** Cycle detection, all paths problems, leads-to-destination validation.  
**LeetCode Example:** [All Paths From Source to Target - LeetCode #797](https://leetcode.com/problems/all-paths-from-source-to-target/)
**When to use:** Cycle detection, all paths problems, leads-to-destination validation.

```java
boolean dfs(int node, Set<Integer> visiting, Set<Integer> visited, Map<Integer, List<Integer>> graph) {
    if (!graph.containsKey(node)) return true; // or false depending on leaf logic

    if (visiting.contains(node)) return false; // cycle detected
    if (visited.contains(node)) return true;   // memoized safe

    visiting.add(node);
    for (int neighbor : graph.get(node)) {
        if (!dfs(neighbor, visiting, visited, graph)) return false;
    }
    visiting.remove(node);
    visited.add(node);

    return true;
}
```

---

### 3. DFS for Topological Sorting (Post-order)  
**When to use:** Topological sorting in DAGs.  
**LeetCode Example:** [Course Schedule II - LeetCode #210](https://leetcode.com/problems/course-schedule-ii/)
**When to use:** Topological sorting in DAGs.

```java
void dfsTopo(int node, Set<Integer> visited, Stack<Integer> stack, Map<Integer, List<Integer>> graph) {
    visited.add(node);
    for (int neighbor : graph.getOrDefault(node, List.of())) {
        if (!visited.contains(neighbor)) {
            dfsTopo(neighbor, visited, stack, graph);
        }
    }
    stack.push(node);
}
```

---

## 🔁 BFS Templates

### 1. BFS Traversal (Shortest Path in Unweighted Graph)  
**When to use:** Shortest path problems, level-order traversal, multi-source problems.  
**LeetCode Example:** [Shortest Path in Binary Matrix - LeetCode #1091](https://leetcode.com/problems/shortest-path-in-binary-matrix/)
**When to use:** Shortest path problems, level-order traversal, multi-source problems.

```java
int bfs(int start, int target, Map<Integer, List<Integer>> graph) {
    Queue<Integer> queue = new LinkedList<>();
    Set<Integer> visited = new HashSet<>();
    queue.offer(start);
    visited.add(start);
    int steps = 0;

    while (!queue.isEmpty()) {
        int size = queue.size();
        for (int i = 0; i < size; i++) {
            int node = queue.poll();
            if (node == target) return steps;
            for (int neighbor : graph.getOrDefault(node, List.of())) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }
        steps++;
    }
    return -1;
}
```

---

### 2. BFS Topological Sort (Kahn's Algorithm)  
**When to use:** Scheduling tasks, topological sort in DAGs, cycle detection.  
**LeetCode Example:** [Course Schedule II - LeetCode #210](https://leetcode.com/problems/course-schedule-ii/)
**When to use:** Scheduling tasks, topological sort in DAGs, cycle detection.

```java
List<Integer> topoSort(int n, Map<Integer, List<Integer>> graph) {
    int[] indegree = new int[n];
    for (List<Integer> neighbors : graph.values()) {
        for (int neighbor : neighbors) {
            indegree[neighbor]++;
        }
    }

    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < n; i++) {
        if (indegree[i] == 0) queue.offer(i);
    }

    List<Integer> result = new ArrayList<>();
    while (!queue.isEmpty()) {
        int node = queue.poll();
        result.add(node);
        for (int neighbor : graph.getOrDefault(node, List.of())) {
            if (--indegree[neighbor] == 0) {
                queue.offer(neighbor);
            }
        }
    }

    return result.size() == n ? result : new ArrayList<>(); // empty = cycle exists
}
```

---

## 🧩 Disjoint Set Union (DSU) / Union-Find

### 1. Union-Find with Path Compression  
**When to use:** Detect cycles in undirected graphs, Kruskal’s MST, group merging problems.  
**LeetCode Example:** [Graph Valid Tree - LeetCode #261](https://leetcode.com/problems/graph-valid-tree/)
**When to use:** Detect cycles in undirected graphs, Kruskal’s MST, group merging problems.

```java
class UnionFind {
    int[] parent;

    UnionFind(int n) {
        parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    boolean union(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false; // cycle
        parent[px] = py;
        return true;
    }
}
```

---

## ⚖️ Weighted Graph Templates

### 1. Dijkstra’s Algorithm (Shortest Path - Weighted Graph)  
**When to use:** Shortest path from one node in graphs with positive edge weights.  
**LeetCode Example:** [Network Delay Time - LeetCode #743](https://leetcode.com/problems/network-delay-time/)
**When to use:** Shortest path from one node in graphs with positive edge weights.

```java
int[] dijkstra(int n, int start, Map<Integer, List<int[]>> graph) {
    int[] dist = new int[n];
    Arrays.fill(dist, Integer.MAX_VALUE);
    dist[start] = 0;
    PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
    pq.offer(new int[]{start, 0});

    while (!pq.isEmpty()) {
        int[] curr = pq.poll();
        int node = curr[0], cost = curr[1];
        if (cost > dist[node]) continue;

        for (int[] edge : graph.getOrDefault(node, List.of())) {
            int neighbor = edge[0], weight = edge[1];
            if (dist[node] + weight < dist[neighbor]) {
                dist[neighbor] = dist[node] + weight;
                pq.offer(new int[]{neighbor, dist[neighbor]});
            }
        }
    }
    return dist;
}
```

---

### 2. Floyd-Warshall (All-Pairs Shortest Path)  
**When to use:** Finding shortest paths between all pairs of nodes in a dense graph.  
**LeetCode Example:** [Find the City With the Smallest Number of Neighbors at a Threshold Distance - LeetCode #1334](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/)
**When to use:** Finding shortest paths between all pairs of nodes in a dense graph.

```java
int[][] floydWarshall(int n, int[][] graph) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (graph[i][k] != Integer.MAX_VALUE && graph[k][j] != Integer.MAX_VALUE) {
                    graph[i][j] = Math.min(graph[i][j], graph[i][k] + graph[k][j]);
                }
            }
        }
    }
    return graph;
}
```

---

### 3. Grid as Graph (BFS / DFS in 2D Grid)  
**When to use:** Maze problems, island counting, shortest path in grids.  
**LeetCode Example:** [Number of Islands - LeetCode #200](https://leetcode.com/problems/number-of-islands/)
**When to use:** Maze problems, island counting, shortest path in grids.

```java
int[][] directions = {{0,1},{1,0},{0,-1},{-1,0}};

void bfsGrid(int[][] grid, int startX, int startY) {
    Queue<int[]> queue = new LinkedList<>();
    queue.offer(new int[]{startX, startY});
    boolean[][] visited = new boolean[grid.length][grid[0].length];
    visited[startX][startY] = true;

    while (!queue.isEmpty()) {
        int[] cell = queue.poll();
        for (int[] dir : directions) {
            int newX = cell[0] + dir[0];
            int newY = cell[1] + dir[1];
            if (newX >= 0 && newY >= 0 && newX < grid.length && newY < grid[0].length && !visited[newX][newY]) {
                visited[newX][newY] = true;
                queue.offer(new int[]{newX, newY});
            }
        }
    }
}
```

---

## 🧠 Advanced Graph Algorithms

### 0. Cycle Detection in Directed Graph
**When to use:** To detect cycles in course schedules, task dependencies, etc.  
**LeetCode Example:** [Course Schedule - LeetCode #207](https://leetcode.com/problems/course-schedule/)

```java
boolean hasCycle(int node, Set<Integer> visiting, Set<Integer> visited, Map<Integer, List<Integer>> graph) {
    if (visiting.contains(node)) return true;
    if (visited.contains(node)) return false;

    visiting.add(node);
    for (int neighbor : graph.getOrDefault(node, List.of())) {
        if (hasCycle(neighbor, visiting, visited, graph)) return true;
    }
    visiting.remove(node);
    visited.add(node);
    return false;
}
```

### 1. Tarjan’s Algorithm (Strongly Connected Components - SCC)

### 1. Tarjan’s Algorithm (Strongly Connected Components - SCC)  
**When to use:** Find SCCs in a directed graph.  
**LeetCode Example:** [Critical Connections in a Network - LeetCode #1192](https://leetcode.com/problems/critical-connections-in-a-network/)
**When to use:** Find SCCs in a directed graph.

```java
void tarjan(int u, int[] low, int[] disc, boolean[] inStack, Stack<Integer> stack, List<List<Integer>> sccs, Map<Integer, List<Integer>> graph, int time) {
    low[u] = disc[u] = time++;
    stack.push(u);
    inStack[u] = true;

    for (int v : graph.getOrDefault(u, List.of())) {
        if (disc[v] == -1) {
            tarjan(v, low, disc, inStack, stack, sccs, graph, time);
            low[u] = Math.min(low[u], low[v]);
        } else if (inStack[v]) {
            low[u] = Math.min(low[u], disc[v]);
        }
    }

    if (low[u] == disc[u]) {
        List<Integer> scc = new ArrayList<>();
        while (true) {
            int v = stack.pop();
            inStack[v] = false;
            scc.add(v);
            if (v == u) break;
        }
        sccs.add(scc);
    }
}
```

### 2. Prim’s Algorithm (Minimum Spanning Tree)  
**When to use:** Minimum spanning tree in dense graphs.  
**LeetCode Example:** [Optimize Water Distribution in a Village - LeetCode #1168](https://leetcode.com/problems/optimize-water-distribution-in-a-village/)
**When to use:** Minimum spanning tree in dense graphs.

```java
int primMST(int n, Map<Integer, List<int[]>> graph) {
    boolean[] visited = new boolean[n];
    PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
    pq.offer(new int[]{0, 0});
    int totalWeight = 0;

    while (!pq.isEmpty()) {
        int[] curr = pq.poll();
        int node = curr[0], weight = curr[1];
        if (visited[node]) continue;
        visited[node] = true;
        totalWeight += weight;

        for (int[] edge : graph.getOrDefault(node, List.of())) {
            int neighbor = edge[0], w = edge[1];
            if (!visited[neighbor]) {
                pq.offer(new int[]{neighbor, w});
            }
        }
    }
    return totalWeight;
}
```

### 3. Eulerian Path and Circuit Detection  
**When to use:** Check if a graph has an Eulerian path or circuit.  
**LeetCode Example:** [Reconstruct Itinerary - LeetCode #332](https://leetcode.com/problems/reconstruct-itinerary/)

```java
boolean hasEulerianCircuit(int[][] graph) {
    int[] degree = new int[graph.length];
    for (int i = 0; i < graph.length; i++) {
        for (int j = 0; j < graph[i].length; j++) {
            if (graph[i][j] > 0) {
                degree[i]++;
                degree[j]++;
            }
        }
    }

    for (int deg : degree) {
        if (deg % 2 != 0) return false; // Eulerian circuit requires all even degrees
    }
    return true;
}
```

### 4. Bipartite Graph Check  
**When to use:** Graph coloring problems, odd cycle detection, two-group partitioning.  
**LeetCode Example:** [Is Graph Bipartite? - LeetCode #785](https://leetcode.com/problems/is-graph-bipartite/)

```java
boolean isBipartite(int[][] graph) {
    int[] color = new int[graph.length]; // 0: unvisited, 1: red, -1: blue

    for (int i = 0; i < graph.length; i++) {
        if (color[i] != 0) continue;
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(i);
        color[i] = 1;

        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int neighbor : graph[node]) {
                if (color[neighbor] == 0) {
                    color[neighbor] = -color[node];
                    queue.offer(neighbor);
                } else if (color[neighbor] == color[node]) {
                    return false;
                }
            }
        }
    }
    return true;
}

### 5. Bellman-Ford Algorithm  
**When to use:** Shortest path with negative weights.  
**LeetCode Example:** [Cheapest Flights Within K Stops - LeetCode #787](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

```java
int bellmanFord(int n, int[][] edges, int src, int dst, int K) {
    int[] dist = new int[n];
    Arrays.fill(dist, Integer.MAX_VALUE);
    dist[src] = 0;

    for (int i = 0; i <= K; i++) {
        int[] tmp = Arrays.copyOf(dist, n);
        for (int[] edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            if (dist[u] != Integer.MAX_VALUE && dist[u] + w < tmp[v]) {
                tmp[v] = dist[u] + w;
            }
        }
        dist = tmp;
    }
    return dist[dst] == Integer.MAX_VALUE ? -1 : dist[dst];
}
```

### 6. Connected Components in Undirected Graph  
**When to use:** Finding the number of connected components using Union-Find.  
**LeetCode Example:** [Number of Connected Components in an Undirected Graph - LeetCode #323](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

```java
int countComponents(int n, int[][] edges) {
    int[] parent = new int[n];
    for (int i = 0; i < n; i++) parent[i] = i;

    int components = n;
    for (int[] edge : edges) {
        int x = find(edge[0], parent), y = find(edge[1], parent);
        if (x != y) {
            parent[x] = y;
            components--;
        }
    }
    return components;
}

int find(int x, int[] parent) {
    if (x != parent[x]) parent[x] = find(parent[x], parent);
    return parent[x];
}
```
  
**When to use:** Check if a graph has an Eulerian path or circuit.  
**LeetCode Example:** [Reconstruct Itinerary - LeetCode #332](https://leetcode.com/problems/reconstruct-itinerary/)
**When to use:** Check if a graph has an Eulerian path or circuit.

```java
boolean hasEulerianCircuit(int[][] graph) {
    int[] degree = new int[graph.length];
    for (int i = 0; i < graph.length; i++) {
        for (int j = 0; j < graph[i].length; j++) {
            if (graph[i][j] > 0) {
                degree[i]++;
                degree[j]++;
            }
        }
    }

    for (int deg : degree) {
        if (deg % 2 != 0) return false; // Eulerian circuit requires all even degrees
    }
    return true;
}
```

