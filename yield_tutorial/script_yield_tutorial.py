# taken from this line:
# https://realpython.com/introduction-to-python-generators/

def csv_reader(file_name):
    file = open(file_name)
    result = file.read().split("\n")
    return result

def csv_reader_using_yield(file_name):
    for row in open(file_name, "r"):
        yield row

def main():
    csv_gen = csv_reader("david.csv")
    #csv_gen = csv_reader_using_yield("david.csv")
    row_count = 0

    for row in csv_gen:
        row_count += 1

    print(f"Row count is {row_count}")

main()
