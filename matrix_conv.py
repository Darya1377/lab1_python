import argparse

    
def count_convolution(matrix, kernel, column_bias, row_bias, kx, ky):
    """

    :param matrix: two-dimensional list
    :param kernel: two-dimensional list
    :param column_bias: int
    column bias in matrix

    :param row_bias: int
    row bias in matrix

    :param kx: int
    kernel size by x coordinate

    :param ky: int
    kernel size by y coordinate

    :return: int
    resultant convolution value
    """
    s = 0
    for i in range(kx):
        for j in range(ky):
            s += matrix[i + row_bias][j + column_bias] * kernel[i][j]
    return s


def convolution(matrix, kernel):
    """

    :param matrix: two-dimensional list
    :param kernel: two-dimensional list

    :return: two-dimensional list
    resultant convolution matrix
    """
    mx = len(matrix)
    my = len(matrix[0])
    kx = len(kernel)
    ky = len(kernel[0])
    res = [[0] * (my - ky + 1) for n in range(mx - kx + 1)]
    if mx >= kx and my >= ky:
        for i in range(mx - kx + 1):
            for j in range(my - ky + 1):
                res[i][j] = count_convolution(matrix, kernel, j, i, kx, ky)

    return res


def read_matrix(f):
    """

    :param f: input txt file

    :return: two-dimensional list
    reads matrix from an input file
    """
    matrix = []
    for line in f:
        if line.split():
            matrix.append([int(x) for x in line.split()])
        else:
            break
    return matrix


def write_matrix(f, matrix):
    """

    :param f: output txt file

    :param matrix: two-dimensional list
    writes resultant convolution matrix in output file
    """
    for row in matrix:
        f.write(' '.join(map(str, row)) + '\n')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('matrix',
                        type=str,
                        help='file of a matrix')
    parser.add_argument('result',
                        type=str,
                        help='file of a result ')
    args = parser.parse_args()
    f1 = open(args.matrix)
    matrix = read_matrix(f1)
    kernel = read_matrix(f1)
    for i in range(len(matrix)-1):
        if len(matrix[i]) != len(matrix[i + 1]):
            print("Incorrect matrix\n")
            return -1
    for i in range(len(kernel)-1):
        if len(kernel[i]) != len(kernel[i + 1]):
            print("Incorrect kernel\n")
            return -1
    f1.close()
    f2 = open(args.result, 'w')
    write_matrix(f2, convolution(matrix, kernel))
    f2.close()


if __name__ == '__main__':
    main()
