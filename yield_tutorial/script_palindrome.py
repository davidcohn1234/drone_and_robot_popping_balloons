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
        return True
    else:
        return False

def infinite_palindromes():
    num = 0
    print('infinite_palindromes: before while True')
    while True:
        #print('inside function')
        if is_palindrome(num):
            print(f'infinite_palindromes: num = {num}')
            i = (yield num+1)
            print(f'infinite_palindromes: i={i}')
            #print(f'num={num}: inside infinite_palindromes')
            if i is not None:
                num = i
        num += 1

def main():
    pal_gen = infinite_palindromes()
    for j in range(5):
        print(f'main: j={j}, before next(pal_gen)')
        k = next(pal_gen)
        print(f'main: k={k}, after next(pal_gen)')
        digits = len(str(k))
        pal_gen.send(10 ** (digits))
        print()

    # for i in range(200):
    #     if is_palindrome(i):
    #         print(i)

main()
