import argparse


def fraction_count(numerator, denominator, number, n_list):
    while numerator > 0:
        number = number * numerator // denominator
        n_list.append(number)
        numerator -= 1
        denominator += 1
    return n_list


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("height",
                        type=int,
                        help="Print the height of the triangle\n")
    args = parser.parse_args()
    row = 0
    max_str_len = 160
    if args.height < 0:
        print("The height should be greater than zero!\n")
        exit(-1)
    while row < args.height:
        n_list = []
        number = 1
        denominator = 1
        numerator = row
        n_list.append(number)
        row += 1
        h = args.height
        print(" ".join(map(str,
                           fraction_count(numerator,
                                          denominator,
                                          number,
                                          n_list))).center(max_str_len))


if __name__ == "__main__":
    main()





