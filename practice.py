
def largest_prime_factor(val):
    # fetch factors
    factors = lambda: [num for num in range(1, val+1) if val % num == 0]
    
    # method to check if 
    def is_prime(num):
        if num <2:
            return False
        for value in range(2, num):
            if num % value == 0:
                return False
        return True
    max_facor = max(filter(is_prime, factors()))
    return max_facor
    
# Above method is not efficiant as we are check is_prime for every factor range till end

def updated_is_prime(num):
    import math
    if num <2:
        return False
    for value in range(2, int(math.sqrt(num)) + 1):
        print(value)
        if num % value == 0:
            return False
    return True


import math
def all_prime_factors(num):
    factors = [val for val in range(1, num+1) if num % val ==0]
    def updated_is_prime(num):
        
        if num <2:
            return False
        for value in range(2, int(math.sqrt(num)) + 1):
            print(value)
            if num % value == 0:
                return False
        return True
    return list(filter(updated_is_prime, factors))


def odd_factor_sum(num):
    is_odd = lambda x: x % 2 != 0 
    odd_factor = [val for val in range(1, num+1) if (val % num == 0) and is_odd(val)]
    return sum(odd_factor)


from itertools import combinations_with_replacement

def coinChange(coins, amount):
    ways = 0
    # loop through range of amount
    for r in range(1, amount + 1):
        for subset in combinations_with_replacement(coins, r):
            if sum(subset) == amount:
                ways += 1
    return ways




# coin change problem 
coins = [1,2,5]
all_combinations = [list(filter(lambda x: sum(x) == 5, list(combinations_with_replacement(coins, val)))) for val in range(1, 5+1)]