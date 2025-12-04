import heapq

def originalDijkstra(edges, n, src):
    adj = {}
    for i in range(1, n + 1):
        adj[i] = []

    for s, d, w in edges:
        adj[s].append([d, w])

    shortest = {}
    minHeap = [[0, src]]
    while minHeap:
        w1, n1 = heapq.heappop(minHeap)
        if n1 in shortest:
            continue
        shortest[n1] = w1

        for n2, w2 in adj[n1]:
            if n2 not in shortest:
                heapq.heappush(minHeap, [w1 + w2, n2])
    return shortest


def main():
    with open("input.txt") as input:
        graph = []
        for line in input:
            node = line.split()
            for i in range(len(node)):
                if node[i].isnumeric():
                    node[i] = int(node[i])
            graph.append(node)

if __name__ == "__main__":
    main()