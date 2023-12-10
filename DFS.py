import numpy as np

# The DFS function
def DFS(graph, start, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []
    
    visited.add(start)
    path.append(start)
    
    for next_node in graph[start]:
        if next_node not in visited:
            DFS(graph, next_node, visited, path)
    
    return path

# The graph represented as a dictionary where each node points to a list of connected nodes
graph = {
    "1-body": ["11-body", "12-body", "14-face"],
    "11-body": ["1-body", "13-body"],
    "13-body": ["11-body", "0-right_hand"],
    "0-right_hand": ["13-body", "1-right_hand", "5-right_hand", "17-right_hand"],
    "1-right_hand": ["0-right_hand", "2-right_hand"],
    "2-right_hand": ["1-right_hand", "3-right_hand"],
    "3-right_hand": ["2-right_hand", "4-right_hand"],
    "4-right_hand": ["3-right_hand"],
    "5-right_hand": ["0-right_hand", "6-right_hand", "9-right_hand"],
    "6-right_hand": ["5-right_hand", "7-right_hand"],
    "7-right_hand": ["6-right_hand", "8-right_hand"],
    "8-right_hand": ["7-right_hand"],
    "9-right_hand": ["5-right_hand", "13-right_hand", "10-right_hand"],
    "10-right_hand": ["9-right_hand", "11-right_hand"],
    "11-right_hand": ["10-right_hand", "12-right_hand"],
    "12-right_hand": ["11-right_hand"],
    "13-right_hand": ["9-right_hand", "14-right_hand", "17-right_hand"],
    "14-right_hand": ["13-right_hand", "15-right_hand"],
    "15-right_hand": ["14-right_hand", "16-right_hand"],
    "16-right_hand": ["15-right_hand"],
    "17-right_hand": ["13-right_hand", "0-right_hand", "18-right_hand"],
    "18-right_hand": ["17-right_hand", "19-right_hand"],
    "19-right_hand": ["18-right_hand", "20-right_hand"],
    "20-right_hand": ["19-right_hand"],
    "12-body": ["1-body", "14-body"],
    "14-body": ["12-body", "0-left_hand"],
    "0-left_hand": ["14-body", "1-left_hand", "5-left_hand", "17-left_hand"],
    "1-left_hand": ["0-left_hand", "2-left_hand"],
    "2-left_hand": ["1-left_hand", "3-left_hand"],
    "3-left_hand": ["2-left_hand", "4-left_hand"],
    "4-left_hand": ["3-left_hand"],
    "5-left_hand": ["0-left_hand", "6-left_hand", "9-left_hand"],
    "6-left_hand": ["5-left_hand", "7-left_hand"],
    "7-left_hand": ["6-left_hand", "8-left_hand"],
    "8-left_hand": ["7-left_hand"],
    "9-left_hand": ["5-left_hand", "13-left_hand", "10-left_hand"],
    "10-left_hand": ["9-left_hand", "11-left_hand"],
    "11-left_hand": ["10-left_hand", "12-left_hand"],
    "12-left_hand": ["11-left_hand"],
    "13-left_hand": ["9-left_hand", "14-left_hand", "17-left_hand"],
    "14-left_hand": ["13-left_hand", "15-left_hand"],
    "15-left_hand": ["14-left_hand", "16-left_hand"],
    "16-left_hand": ["15-left_hand"],
    "17-left_hand": ["13-left_hand", "0-left_hand", "18-left_hand"],
    "18-left_hand": ["17-left_hand", "19-left_hand"],
    "19-left_hand": ["18-left_hand", "20-left_hand"],
    "20-left_hand": ["19-left_hand"],
    "14-face": ["1-body", "324-face", "13-face", "76-face"],
    "324-face": ["14-face", "13-face"],
    "76-face": ["14-face", "13-face"],
    "13-face": ["324-face", "76-face", "14-face", "0-face"],
    "0-face": ["13-face", "155-face", "65-face", "295-face", "382-face"],
    "155-face": ["0-face", "159-face", "145-face"],
    "145-face": ["155-face", "7-face"],
    "7-face": ["145-face", "159-face"],
    "159-face": ["7-face", "155-face"],
    "65-face": ["0-face", "53-face"],
    "53-face": ["65-face", "52-face"],
    "52-face": ["53-face", "46-face"],
    "46-face": ["52-face"],
    "295-face": ["0-face", "283-face"],
    "283-face": ["295-face", "282-face"],
    "282-face": ["283-face", "276-face"],
    "276-face": ["282-face"],
    "382-face": ["0-face", "385-face", "374-face"],
    "374-face": ["382-face", "249-face"],
    "249-face": ["374-face", "385-face"],
    "385-face": ["382-face", "249-face"],
}

# Perform DFS traversal starting from node -1
dfs_path = DFS(graph, "1-body")

# Convert the path to a numpy array
dfs_path_array = np.array(dfs_path)
print("DFS:", dfs_path_array)
print("DFS legth:", len(dfs_path_array))
