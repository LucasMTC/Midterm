import heapq
import math
import time
import random
import csv
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


def bmssp(graph, source, goal):
    n = graph.vertices
    logn = math.log2(n) if n > 1 else 1
    k = max(3, int(logn**0.5))
    t = max(2, int(logn**(1/3)))

    k = max(k, 3)
    t = max(t, 2)

    distances = {v: INFINITY for v in graph.adj}
    predecessors = {v: None for v in graph.adj}
    complete = {v: False for v in graph.adj}
    best_goal = INFINITY
    distances[source] = 0.0

    def reconstruct(source, goal):
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            if cur == source:
                break
            cur = predecessors[cur]
        return path[::-1]

    def base_case(bound, frontier, goal):
        nonlocal best_goal
        pq = []
        for s in frontier:
            if not complete[s] and distances[s] < bound:
                heapq.heappush(pq, (distances[s], s))
        done = []
        while pq:
            dist, u = heapq.heappop(pq)
            if complete[u] or dist > distances[u]:
                continue
            complete[u] = True
            done.append(u)
            if u == goal:
                if dist < best_goal:
                    best_goal = dist
                break
            for edge in graph.adj[u]:
                v = edge.to
                nd = dist + edge.weight
                if not complete[v] and nd <= distances[v] and nd < bound and nd < best_goal:
                    distances[v] = nd
                    predecessors[v] = u
                    heapq.heappush(pq, (nd, v))
        return done

    def find_pivots(bound, frontier):
        nonlocal best_goal
        working = set(frontier)
        layer = {x for x in frontier if not complete[x]}
        for _ in range(k):
            next_layer = set()
            for u in layer:
                if distances[u] >= bound:
                    continue
                for edge in graph.adj[u]:
                    v = edge.to
                    if complete[v]:
                        continue
                    nd = distances[u] + edge.weight
                    if nd <= distances[v] and nd < bound and nd < best_goal:
                        distances[v] = nd
                        predecessors[v] = u
                        if v not in working:
                            next_layer.add(v)
            if not next_layer:
                break
            working.update(next_layer)
            layer = next_layer
            if len(working) > k * len(frontier):
                return frontier, list(working)
        children = {x: [] for x in working}
        for x in working:
            p = predecessors[x]
            if p in working:
                children.setdefault(p, []).append(x)
        sizes = {x: len(children.get(x, [])) for x in working}
        pivots = [r for r in frontier if sizes.get(r, 0) >= k]
        if not pivots:
            return frontier, list(working)
        return pivots, list(working)

    def edge_relax(completed_vertices, lower, upper, ds):
        nonlocal best_goal
        batch = []
        for u in completed_vertices:
            for edge in graph.adj[u]:
                v = edge.to
                if complete[v]:
                    continue
                nd = distances[u] + edge.weight
                if nd <= distances[v] and nd < best_goal:
                    distances[v] = nd
                    predecessors[v] = u
                    if nd < lower:
                        batch.append((v, nd))
                    elif nd < upper:
                        ds.insert(v, nd)
        if batch:
            ds.batch_prepend(batch)

    def recurse(level, bound, pivots, goal):
        if not pivots or (goal is not None and complete[goal]):
            return []
        if level == 0:
            return base_case(bound, pivots, goal)
        pivots, _ = find_pivots(bound, pivots)
        block = min(1024, 2**max(0, (level - 1) * t))
        ds = EfficientDataStructure(block, bound)
        for p in pivots:
            if not complete[p] and distances[p] < bound:
                ds.insert(p, distances[p])
        res = []
        while not ds.is_empty():
            if goal is not None and complete[goal]:
                break
            subset_bound, subset = ds.pull()
            if not subset:
                continue
            sub = recurse(level - 1, subset_bound, subset, goal)
            res.extend(sub)
            edge_relax(sub, subset_bound, bound, ds)
        return res

    max_level = math.ceil(math.log2(n) / t) if n > 1 else 0
    recurse(max_level, INFINITY, [source], goal)

    if distances[goal] == INFINITY:
        return None

    return distances[goal], reconstruct(source, goal)


def main():

    def generate_random_graph(num_vertices, density=0.1, max_weight=20):
        g = Graph()
        vertices = [str(i) for i in range(num_vertices)]
        for v in vertices:
            g.adj.setdefault(v, [])
        for u in vertices:
            for v in vertices:
                if u != v and random.random() < density:
                    g.add_edge(u, v, random.randint(1, max_weight))
        return g, vertices[0], vertices[-1]

    def run_trial(num_vertices, density=0.1):
        g, source, goal = generate_random_graph(num_vertices, density)
        start = time.time()
        dijkstra(g, source, goal)
        t_dijkstra = time.time() - start

        start = time.time()
        bmssp(g, source, goal)
        t_bmssp = time.time() - start

        return t_dijkstra, t_bmssp

    def benchmark_suite(sizes=[10, 100, 1000, 5000], density=0.005, iterations=10):
        results = []
        for n in sizes:
            d_times, b_times = [], []
            for _ in range(iterations):
                dt, bt = run_trial(n, density)
                d_times.append(dt)
                b_times.append(bt)
            avg_d = sum(d_times)/len(d_times)
            avg_b = sum(b_times)/len(b_times)
            speedup = avg_d/avg_b if avg_b>0 else float('inf')
            results.append((n, avg_d, avg_b, speedup))
            print(f"Graph size: {n} | Dijkstra: {avg_d:.6f}s | BMSSP: {avg_b:.6f}s | Speedup: {speedup:.2f}x")
        return results

    def save_results_csv(results, filename="benchmark_results.csv"):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Vertices", "Dijkstra", "BMSSP", "Speedup"])
            for row in results:
                writer.writerow(row)
        print(f"Results saved to {filename}")

    results = benchmark_suite()
    save_results_csv(results)

if __name__ == "__main__":
    main()