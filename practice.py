
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



s = "pwwkew"
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        seen = {}
        max_len = 0
        start = 0  # left pointer of the window

        for end in range(len(s)):
            char = s[end]
            if char in seen and seen[char] >= start:
                # Move the start to one position after the last occurrence of char
                start = seen[char] + 1
            
            seen[char] = end  # Update last seen index of current char
            print(seen)
            max_len = max(max_len, end - start + 1)
            print(max_len)

        return max_len
    
board = [["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]

class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        import numpy as np
        sudoku = np.array(board)
        
        def is_unique(arr):
            valid_arr = [val for val in arr if val != '.']
            return len(valid_arr) == len(set(valid_arr))
        
        for i in range(0,9):
            if not is_unique(sudoku[i, :]) or not is_unique(sudoku[:, i]):
                return False
        # 3 * 3 sub sudoku
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                box = sudoku[row:row+3, col:col+3].flatten()
                if not is_unique(box):
                    return False
        return True
                
            
        
        

arr = [2,2,45,23,56,2,8]
def Prefix_sum(Arr):
    last_sum = None
    sum_arr = []
    for indx,val in enumerate(Arr):
        if indx <1:
            last_sum = val
            sum_arr.append(last_sum)
        else:
            last_sum = last_sum + val
            sum_arr.append(last_sum)
    return sum_arr

def Prefix_sum(Arr):
    last_sum = 0
    prefix_sum = []
    for val in Arr:
        last_sum += val
        prefix_sum.append(last_sum)
    return prefix_sum
        
        
        
        
        
        
        
