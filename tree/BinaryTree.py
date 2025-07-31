import json
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

# Binary Tree Node Definition
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

# Binary Tree Class
class BinaryTree:
    def __init__(self, root=None):
        self.root = root
        
    def insert(self, data):
        if not self.root:
            self.root = Node(data)
        else:
            self._insert_recursively(self.root, data)

    def _insert_recursively(self, node, data):
        if data < node.val:
            if node.left is None:
                node.left = Node(data)
            else:
                self._insert_recursively(node.left, data)
        elif data > node.val:
            if node.right is None:
                node.right = Node(data)
            else:
                self._insert_recursively(node.right, data)
        

    # Preorder Traversal (Root -> Left -> Right)
    def preorder(self, node):
        if node:
            print(node.val, end=" ")
            self.preorder(node.left)
            self.preorder(node.right)

    # Inorder Traversal (Left -> Root -> Right)
    def inorder(self, node):
        if node:
            self.inorder(node.left)
            print(node.val, end=" ")
            self.inorder(node.right)

    # Postorder Traversal (Left -> Right -> Root)
    def postorder(self, node):
        if node:
            self.postorder(node.left)
            self.postorder(node.right)
            print(node.val, end=" ")

    # Level Order Traversal (BFS)
    def level_order(self, node):
        if not node:
            return
        queue = deque([node])
        while queue:
            curr = queue.popleft()
            print(curr.val, end=" ")
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)

    # Compute Height of the Tree
    def height(self, node):
        if not node:
            return 0
        return 1 + max(self.height(node.left), self.height(node.right))

    # Check if Tree is Balanced
    def isBalanced(self, node):
        if not node:
            return True
        left_height = self.height(node.left)
        right_height = self.height(node.right)
        return abs(left_height - right_height) <= 1 and self.isBalanced(node.left) and self.isBalanced(node.right)

    # Lowest Common Ancestor (LCA)
    def LCA(self, root, p, q):
        if not root or root == p or root == q:
            return root
        left = self.LCA(root.left, p, q)
        right = self.LCA(root.right, p, q)
        return root if left and right else left or right

    # Serialize the Tree (Convert to String)
    def serialize(self, root):
        return json.dumps(self._serialize_helper(root))

    def _serialize_helper(self, node):
        if not node:
            return None
        return [node.val, self._serialize_helper(node.left), self._serialize_helper(node.right)]

    # Deserialize the Tree (Reconstruct from String)
    def deserialize(self, data):
        return self._deserialize_helper(json.loads(data))

    def _deserialize_helper(self, data):
        if not data:
            return None
        node = Node(data[0])
        node.left = self._deserialize_helper(data[1])
        node.right = self._deserialize_helper(data[2])
        return node

    # Construct Tree from Preorder & Inorder Traversal
    def build_tree(self, preorder, inorder):
        if not inorder:
            return None
        root_val = preorder.pop(0)
        root = Node(root_val)
        mid = inorder.index(root_val)
        root.left = self.build_tree(preorder, inorder[:mid])
        root.right = self.build_tree(preorder, inorder[mid+1:])
        return root

    # Tree Visualization using networkx and matplotlib
    def visualize_tree(self, root):
        if not root:
            print("Tree is empty!")
            return

        graph = nx.DiGraph()
        self._add_edges(graph, root)
        
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=12, font_weight="bold")
        plt.show()

    def _add_edges(self, graph, node):
        if node.left:
            graph.add_edge(node.val, node.left.val)
            self._add_edges(graph, node.left)
        if node.right:
            graph.add_edge(node.val, node.right.val)
            self._add_edges(graph, node.right)

# Example Usage
if __name__ == "__main__":
    # Constructing the Binary Tree
    tree = BinaryTree()
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    root.right.right = Node(6)
    tree.root = root

    print("Preorder Traversal:")
    tree.preorder(tree.root)
    print("\nInorder Traversal:")
    tree.inorder(tree.root)
    print("\nPostorder Traversal:")
    tree.postorder(tree.root)
    print("\nLevel Order Traversal:")
    tree.level_order(tree.root)

    print("\nHeight of Tree:", tree.height(tree.root))
    print("Is Tree Balanced?", tree.isBalanced(tree.root))

    # Finding LCA of nodes 4 and 5
    lca_node = tree.LCA(tree.root, root.left.left, root.left.right)
    print("Lowest Common Ancestor of 4 and 5:", lca_node.val if lca_node else "Not Found")

    # Serialization & Deserialization
    serialized_data = tree.serialize(tree.root)
    print("Serialized Tree:", serialized_data)
    
    new_root = tree.deserialize(serialized_data)
    print("Tree Deserialized. Preorder of new tree:")
    tree.preorder(new_root)

    # Visualize the Tree
    tree.visualize_tree(tree.root)
