# Definition of the Node class.
class Node:
    # Node constructor, initializes a new node with the given data.
    def __init__(self, data):
        self.data = data      # Store the provided data.
        self.next = None      # Pointer to the next node (initially None).
        self.prev = None      # Pointer to the previous node (for doubly linked list, initially None).

# Definition of the DoublyLinkedList class.
class DoublyLinkedList:
    # Constructor initializes an empty linked list with no head node.
    def __init__(self):
        self.head = None

    # Method to add a new node with the given data to the end of the linked list.
    def append(self, data):
        new_node = Node(data)  # Create a new node.
        
        # If the linked list is empty, set the new node as the head.
        if not self.head:
            self.head = new_node
            return
        
        # If the list is not empty, traverse to the end of the list.
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        
        # Once the last node is found, set its next pointer to the new node.
        last_node.next = new_node
        
        # For the new node, set its previous pointer to the last node.
        new_node.prev = last_node

    # Method to get the position/index of a given node in the linked list.
    def get_node_position(self, node):
        current_node = self.head  # Start traversal from the head node.
        position = 0  # Initialize the position counter.
        
        # Traverse the linked list.
        while current_node:
            # If the current node matches the given node, return its position.
            if current_node == node:
                return position
            
            # Move to the next node and increment the position counter.
            current_node = current_node.next
            position += 1
        
        # If node is not found in the list, return None.
        return None

    # Method to traverse and print the linked list in reverse order (from tail to head).
    def traverse_backward(self):
        # If the list is empty, exit the method.
        if not self.head:
            return None
        
        # Traverse to the end of the list.
        current_node = self.head
        while current_node.next:
            current_node = current_node.next
        
        # Now, traverse backward from the tail to the head, printing each node's data.
        while current_node:
            print(current_node.data)
            current_node = current_node.prev  # Move to the previous node.


class LinkedList:
    def __init__(self):
        # start of linked list 
        self.head = None

    def append(self, data):
        # Add new node by init node object
        new_node = Node(data)
        # Check if the node is empty
        if not self.head:
            # if its empty insert to start of node
            self.head = new_node
            return
        # Get the current node
        last_node = self.head
        # traverse through the linked list
        while last_node.next:
            last_node = last_node.next
        # once the last node is found , Add new node
        last_node.next = new_node

    def print_list(self):
        current_node = self.head
        while current_node:
            print(current_node.data)
            current_node = current_node.next
            
    def len(self):
        current_node = self.head
        count = 0 
        while current_node:
            count+=1
            current_node = current_node.next
        return count

    def reverse(self):
        prev = None
        current = self.head
        while current:
            next_node = current.next  # Temporary node to store the next node.
            current.next = prev  # Reverse the link
            prev = current  # Move prev to this node.
            current = next_node  # Move to the next node.
        self.head = prev  # Finally, update the head node.
