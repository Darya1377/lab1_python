class Error(Exception):
    pass


class Common:

    def __init__(self):
        self.salary = 0
        self.job = 0

    def take_salary(self, num):  # is being called for classes Pupa and Lupa individually; gives them salary
        """

        :param num: int
        """
        self.salary += num

    @staticmethod
    def read_matrix(f): # reads matrix from an input file
        """

        :param f: input file
        :return: two-dimensional list
        """
        matrix = []
        for line in f:
            if line.split():
                matrix.append([int(x) for x in line.split()])
            else:
                break
        return matrix

    @staticmethod
    def create_res_matrix(rows1, cols1, rows2, cols2):  # creates an empty matrix for a result
        """

        :param rows1: int
        :param cols1: int
        :param rows2: int
        :param cols2: int
        :return: two-dimensional list
        """
        if rows1 == rows2 or cols1 == cols2:

            work = [[0 for row in range(cols2)] for col in range(rows1)]
            return work
        else:
            raise Error("Matrises are not equal to each other")


class Pupa(Common):

    def __init__(self):
        super().__init__()

    def do_work(self, filename1, filename2):  # matrix addition
        """

        :param filename1: input file
        :param filename2: input file
        :return: two-dimensional list
        """
        f1 = open(filename1)
        f2 = open(filename2)
        matrix1 = self.read_matrix(f1)
        matrix2 = self.read_matrix(f2)
        pupa_work = self.create_res_matrix(len(matrix1), len(matrix1[0]), len(matrix2), len(matrix2[0]))

        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                pupa_work[i][j] = matrix1[i][j] + matrix2[i][j]
                self.job += 1
        f1.close()
        f2.close()
        return pupa_work


class Lupa(Common):

    def __init__(self):
        super().__init__()

    def do_work(self, filename1, filename2):  # matrix substraction
        """

        :param filename1: input file
        :param filename2: input file
        :return: two-dimensional list
        """
        f1 = open(filename1)
        f2 = open(filename2)
        matrix1 = self.read_matrix(f1)
        matrix2 = self.read_matrix(f2)

        lupa_work = self.create_res_matrix(len(matrix1), len(matrix1[0]), len(matrix2), len(matrix2[0]))

        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                lupa_work[i][j] = matrix1[i][j] - matrix2[i][j]
                self.job += 1
        f1.close()
        f2.close()
        return lupa_work


class Accountant:

    def __init__(self):
        pass

    def give_salary(self, worker): # calls method take_salary on a classes Pupa and Lupa individually
        """

        :param worker: class object
        """
        worker.take_salary(1000 * worker.job)


def main():

    lupa = Lupa()
    l = lupa.do_work("matrix1.txt", "matrix2.txt")
    pupa = Pupa()
    p = pupa.do_work("matrix1.txt", "matrix2.txt")
    a1 = Accountant()
    a1.give_salary(lupa)
    a1.give_salary(pupa)

    try:
        print(lupa.salary)
    except Error:
        print(Error)
        exit(-1)

    try:
        print(pupa.salary)
    except Error:
        print(Error)
        exit(-1)


if __name__ == '__main__':
    main()

