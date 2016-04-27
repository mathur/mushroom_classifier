import numpy

from formulas import sig, inv_sig, err, inv_err
from models import Node, Connection

if __name__ == '__main__':
    f = open('mushroom-training.txt', 'r').readlines()

    attrs = []
    results = []

    for row in f:
        row = [x.strip() for x in row.split(',')]
        row = [int(num) for num in row]
        results.append(int(row[0]))
        attrs.append(row[1:])