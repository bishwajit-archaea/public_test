# A simple Python script to calculate a Fibonacci series
def fib(n):
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()

# Call the function
fib(1000)
