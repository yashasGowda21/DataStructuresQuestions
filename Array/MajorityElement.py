from typing import List  

"""
Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

 

Example 1:

Input: nums = [3,2,3]
Output: 3
Example 2:

Input: nums = [2,2,1,1,1,2,2]
Output: 2
 

Constraints:

n == nums.length
1 <= n <= 5 * 104
-109 <= nums[i] <= 109
 
"""


class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        counts = {}
        for ele in nums:
            if ele in counts.keys():
                counts[ele] +=1
            else:
                counts[ele] =1
        print(counts)
        return max(counts, key=counts.get)
        