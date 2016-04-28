import numpy
import random

from config import LEARNING_RATE
from formulas import sig, inv_sig, inv_err

curr_node_id = 0

class Node:
    def __init__(self, value):
        self.id = curr_node_id
        self.value = value
        self.bias = (random.random() * 2) - 1
        curr_node_id += 1

class Edge:
    def __init__(self, weight, from_node, to_node):
        self.weight = weight
        self.from_node = from_node
        self.to_node = to_node

class Layer:
    def __init__(self, num_nodes, input_vals, layer_num):
        self.num_nodes = num_nodes
        self.input_vals = input_vals
        self.layer_num = layer_num
        self.weight = [[random.random() for col in range(len(input_vals))] for row in range(num_nodes)]
        self.weight_delta = [[0 for col in range(len(input_vals))] for row in range(num_nodes)]
        self.layer_net = [0 for col in range(num_nodes)]
        self.layer_out = [0 for col in range(num_nodes)]
        self.bias = (random.random() * 2) - 1

    def eval(self):
        for x in range(self.num_nodes):
            self.layer_net[x] = numpy.dot(self.input_vals, numpy.transpose(self.weight[x])) + self.bias
            self.layer_out[x] = sig(self.layer_net[x])

    def backprop(self, other):
        for x in range(len(self.weight)):
            for y in range(len(self.weight[x])):
                if self.layer_num == 1:
                    self.weight[x][y] = self.weight[x][y] - (LEARNING_RATE * other.weight_delta[0][x] * self.input_vals[y] * other.weight[0][x] * inv_sig(self.layer_out[x]))
                elif self.layer_num == 2:
                    self.weight_delta[x][y] = inv_sig(self.layer_out[x]) * inv_err(self.layer_out[x], other)
                    self.weight[x][y] = self.weight[x][y] - (LEARNING_RATE * self.weight_delta[x][y] * self.input_vals[y])

class cfile(file):
    def __init__(self, name, mode = 'r'):
        self = file.__init__(self, name, mode)

    def w(self, string):
        self.writelines(str(string) + '\n')
        return None