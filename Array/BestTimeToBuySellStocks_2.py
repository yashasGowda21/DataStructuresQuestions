from typing import List 

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        for ele in range(0, len(prices) - 1):
            if prices[ele +1] > prices[ele]:
                max_profit += prices[ele +1] - prices[ele]
        return max_profit



if __name__ == "__main__":
    obj = Solution()
    prices = [7,1,5,3,6,4]
    obj.maxProfit(prices)