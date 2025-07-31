from typing import List  


"""


You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, 
and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, 
but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, 
where the first m elements denote the elements that should be merged, and the last n elements are 
set to 0 and should be ignored. nums2 has a length of n.

 

Example 1:

Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
Example 2:

Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]
Explanation: The arrays we are merging are [1] and [].
The result of the merge is [1].
Example 3:

Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]
Explanation: The arrays we are merging are [] and [1].
The result of the merge is [1].
Note that because m = 0, there are no elements in nums1. The 0 is only there to ensure the 
merge result can fit in nums1.
 


"""


class Solution:
    
    
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Merge nums2 into nums1 as one sorted array.
        Modify nums1 in-place.
        """

        # Set pointers for nums1, nums2, and the end of nums1
        i = m - 1  # Last actual element in nums1
        j = n - 1  # Last element in nums2
        k = m + n - 1  # Last index in nums1 (including extra space)

        # Merge from the end to avoid overwriting nums1 values
        while i >= 0 and j >= 0:
            print(f"nums1.. {nums1[i]}, nums2.. {nums2[j]}")
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]  # Place larger value at the end
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1  # Move the write pointer left

        # If there are leftover elements in nums2, copy them
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1

        # No need to handle leftover nums1 elements — they're already in place


if __name__ == "__main__":
    nums1 = [1,2,3,0,0,0]
    nums2 = [2,5,6]
    m = 3
    n = 3
    obj = Solution()
    obj.merge(nums1, m, nums2, n)
    print(nums1)
    
     
        


