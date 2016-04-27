import random

class Node:
    def __init__(self, value):
        self.value = value

class Connection:
    def __init__(self, parent, child):
        self.weight = random.random()
        self.parent = parent
        self.child = child