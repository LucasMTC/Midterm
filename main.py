import heapq
from collections import deque

INFINITY = float('inf')

class Edge:
    def __init__(self, to: str, weight: int):
        self.to = to
        self.weight = weight

    def __repr__(self):
        return f"Edge(to={self.to}, weight={self.weight})"

class Graph:
    def __init__(self):
        self.vertices = 0
        self.adj = {}
        for i in range(self.vertices):
            self.adj[chr(i + 65)] = []


    def add_edge(self, u: str, v: str, weight: float):
        if u in self.adj:
            self.adj[u].append(Edge(v, weight))
        else:
            self.adj[u] = [Edge(v, weight)]
            self.vertices += 1
        if v not in self.adj:
            self.adj[v] = []
            self.vertices += 1

    def __repr__(self):
        edge_count = sum(len(e) for v, e in self.adj.items())
        return f"Graph(vertices={self.vertices}, edges={edge_count})"


# DATA STRUCTURE RETREIVED FROM AN ONLINE REPO TO HELP WITH THE BMSSP ALGORITHM IMPLEMENTATION
class EfficientDataStructure:
    def __init__(self, block_size: int, bound: float):
        self.batch_blocks = deque()
        self.sorted_blocks = []
        self.block_size = block_size
        self.bound = bound

    def insert(self, vertex: int, distance: float):
        if distance < self.bound:
            if not self.sorted_blocks or len(self.sorted_blocks[-1]) >= self.block_size:
                self.sorted_blocks.append([])
            self.sorted_blocks[-1].append((vertex, distance))

    def batch_prepend(self, items: list):
        if items:
            self.batch_blocks.appendleft(list(items))

    def pull(self) -> tuple:
        block_to_process = None
        if self.batch_blocks:
            block_to_process = self.batch_blocks.popleft()
        elif self.sorted_blocks:
            min_dist_in_blocks = [
                min(distance for _, distance in block) if block else float('inf')
                for block in self.sorted_blocks
            ]
            min_block_idx = min(range(len(min_dist_in_blocks)), key=min_dist_in_blocks.__getitem__)
            block_to_process = self.sorted_blocks.pop(min_block_idx)

        if block_to_process:
            block_to_process.sort(key=lambda x: x[1])
            vertices = [vertex for vertex, _ in block_to_process]
            next_bound = self.peek_min()
            return next_bound, vertices

        return self.bound, []

    def peek_min(self) -> float:
        min_val = self.bound
        all_blocks = list(self.batch_blocks) + self.sorted_blocks
        for block in all_blocks:
            if block:
                block_min = min(distance for _, distance in block)
                min_val = min(min_val, block_min)
        return min_val

    def is_empty(self) -> bool:
        return not any(self.batch_blocks) and not any(self.sorted_blocks)

# DATA STRUCTURE RETREIVED FROM AN ONLINE REPO TO HELP WITH THE BMSSP ALGORITHM IMPLEMENTATION
class BucketQueue:
    def __init__(self, delta: float, initial_buckets=64):
        self.delta = delta
        self.min_idx = 0
        self.buckets = [deque() for _ in range(initial_buckets)]
        self.max_idx = -1

    def clear(self):
        for i in range(self.min_idx, self.max_idx + 1):
            self.buckets[i].clear()
        self.min_idx = 0
        self.max_idx = -1

    def insert(self, v: int, dist: float):
        idx = int(dist / self.delta)
        if idx >= len(self.buckets):
            num_new = idx - len(self.buckets) + 1
            self.buckets.extend(deque() for _ in range(num_new))
        self.buckets[idx].append(v)
        if idx > self.max_idx:
            self.max_idx = idx

    def extract_min(self) -> tuple:
        min_idx = self.min_idx
        while min_idx <= self.max_idx:
            if self.buckets[min_idx]:
                v = self.buckets[min_idx].popleft()
                self.min_idx = min_idx
                return v, True
            min_idx += 1
        return None, False


def dijkstra(graph: Graph, source: str, goal: str):
    dist = {v: INFINITY for v in graph.adj}
    prev = {v: None for v in graph.adj}
    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == goal:
            break
        for edge in graph.adj[u]:
            nd = dist[u] + edge.weight
            if nd < dist[edge.to]:
                dist[edge.to] = nd
                prev[edge.to] = u
                heapq.heappush(pq, (nd, edge.to))

    if dist[goal] == INFINITY:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        if cur == source:
            break
        cur = prev[cur]

    if not path or path[-1] != source:
        return None

    return dist[goal], path[::-1]



def main():
    with open("input.txt") as file:
        raw_graph = file.readlines()
        graph = Graph()
        for line in raw_graph:
            node = line.split()
            for i in range(len(node)):
                if node[i].isnumeric():
                    node[i] = int(node[i])
            graph.add_edge(node[0], node[1], node[2])
            
    print(dijkstra(graph, "A", "E"))

if __name__ == "__main__":
    main()