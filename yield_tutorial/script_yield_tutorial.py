import cProfile

# taken from this line:
# https://realpython.com/introduction-to-python-generators/

def csv_reader(file_name):
    file = open(file_name)
    result = file.read().split("\n")
    return result

def csv_reader_using_yield(file_name):
    for row in open(file_name, "r"):
        yield row

def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1

def is_palindrome(num):
    # Skip single-digit inputs
    if num // 10 == 0:
        return False
    temp = num
    reversed_num = 0

    while temp != 0:
        reversed_num = (reversed_num * 10) + (temp % 10)
        temp = temp // 10

    if num == reversed_num:
        return num
    else:
        return False

def main():
    cProfile.run('sum([i * 2 for i in range(10000)])')


    # nums_squared_lc = [num ** 2 for num in range(5)]
    # nums_squared_gc = (num ** 2 for num in range(5))
    #
    # i = next(nums_squared_gc)
    # print(i)
    #
    # i = next(nums_squared_gc)
    # print(i)
    #
    # i = next(nums_squared_gc)
    # print(i)
    #
    # counter = 0
    # for i in infinite_sequence():
    #     pal = is_palindrome(i)
    #     if pal:
    #         counter += 1
    #         print(f'{counter}.  {i}')
    # csv_gen = csv_reader("david.csv")
    # #csv_gen = csv_reader_using_yield("david.csv")
    # row_count = 0
    #
    # for row in csv_gen:
    #     row_count += 1
    #
    # print(f"Row count is {row_count}")

main()
