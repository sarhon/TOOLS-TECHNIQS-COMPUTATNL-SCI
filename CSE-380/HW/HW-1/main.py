from typing import Optional, List
from sympy import isprime as sympy_is_prime

def is_prime_fixed(n: int) -> bool:
    if n <= 1:
        return False
    if n == 2:
        return True

    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def tests(test_numbers: Optional[List[int]] = None) -> None:
    if test_numbers is None:
        test_numbers = [13, 35, 49, 91]
        # expected_results = [True, False, False, False]

    expected_results = [sympy_is_prime(number) for number in test_numbers]

    test_results = []
    for number in test_numbers:
        result = is_prime_fixed(number)
        test_results.append(result)

    for idx, number in enumerate(test_numbers):
        if test_results[idx] == expected_results[idx]:
            print(f"test({number})  passed!")
        else:
            print(f"test({number}) failed! <=== got({test_results[idx]}), expected({expected_results[idx]})")
    assert test_results == expected_results, "Not all test results equalled expected results"
    print("All tests passed!\n")
    return None


def user_input(print_welcome=True) -> int:
    if print_welcome:
        print("Please input a number to test if it is prime.")
    try:
        number = int(input('number: '))
    except ValueError:
        print(f"Not a valid number, try again.")
        number = user_input(print_welcome=False)
    return number


def user_loop() -> None:
    try:
        print("Press Ctrl+C to stop.")
        while True:
            n = user_input()
            print(f"is_prime({n}) = {is_prime_fixed(n)}\n")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected, exiting.")


def main():
    run_tests = True
    if run_tests:
        tests()

    run_user_loop = True
    if run_user_loop:
        user_loop()


if __name__ == '__main__':
    main()