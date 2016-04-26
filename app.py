import numpy

if __name__ == '__main__':
    file = open('mushroom-training.txt', 'r').readlines()

    attrs = []
    results = []

    for row in file:
        row = [x.strip() for x in row.split(',')]
        row = [int(num) for num in row]
        results.append(int(row[0]))
        attrs.append(row[1:])

    print results