from typing import List  
class Solution:
    # [2,3,1,1,4]
    # [3,2,1,0,4]
    def canJump(self, nums: List[int]) -> bool:
        max_jump = 0
        for ele in range(len(nums)):
            if ele > max_jump:
                return False
            max_jump = max(max_jump, ele + nums[ele])
        return True
                
            
            

def wordPattern(pattern, s):
    str_set = s.split(' ')
    pattern_set = [letter for letter in pattern]
    print(str_set, pattern_set)
    if not len(pattern_set) == len(str_set):
        return False
    


# class Solution:
#     from typing import List
#     def summaryRanges(self, nums: List[int]) -> List[str]:
#         sub_arr = []
#         parsed = set()
#         for item in nums:
#             if item in parsed:
#                 print(f"skipping {item}")
#                 continue
#             arr = []
#             curr = item
#             arr.append(item)
#             while curr + 1 in nums:
#                 arr.append(curr+1)
#                 parsed.add(curr+1)
#                 curr +=1
#             if arr:
#                 if len(arr) > 1:
#                     sub_arr.append(f"{min(arr)}->{max(arr)}")
#                 else:
#                     sub_arr.append(f"{arr[0]}")
#         print(sub_arr)
    
points = [[10,16],[2,8],[1,6],[7,12]]
# [[1, 6], [2, 8], [7, 12], [10, 16]]
# curr_start <= last_arrow_position

# class Solution:
#     from typing import List
#     def findMinArrowShots(self, points: List[List[int]]) -> int:
#         if not points:
#             return 0
#         points.sort(key=lambda x : x[-1])
#         arrow = 1
#         last_arrow_pos = points[0][1]
#         for indx in range(1,len(points)):
#             curr_start, curr_end = points[indx]
#             if curr_start > last_arrow_pos:
#                 arrow+=1
#                 last_arrow_pos = curr_end
                
#         return arrow


def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    start = 0
    total = 0
    min_len = 0

    for end in range(len(nums)):
        total += nums[end]
        while total >= target:
            # Calculate the current window size
            window_size = end - start + 1

            # Update the minimum length if this one is smaller
            if window_size < min_len:
                min_len = window_size

            # Shrink the window from the left
            total -= nums[start]
            start += 1
    return min_len if min_len else 0




import numpy as np


class LinearRegression:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        
    def fit(self, X, y):
        self.m , self.n = X.shape # m --> samples , n -- > features
        self.w = np.zeros(self.n) # Init weights
        self.b = 0
        
        for _ in self.epochs:
            # Run predict
            y_pred = self.predict(X)
            # compute cost by taking derivative of cost function
            dw = (1/self.m) * np.dot(X.T, y_pred - y)
            db = (1/self.m) * np.sum(X.T, y_pred - y)
            
            # modifu the weights
            self.w -= dw * self.lr
            self.b -= db * self.lr
              
    
    def predict(self, X):
        return np.dot(self.w, X) + self.b
                
        
        


class LogisticRegression:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def sig(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, Y):
        self.m , self.n = X.shape
        # init weights
        self.w = np.zeros(self.n)
        self.b = 0
        
        for _ in self.epochs:
            y_pred = self.predict(X)
            # calculate loss & derivative
            dw = 1 / self.m * np.prod(X.T, y_pred - Y)
            dx = 1 / self.m * np.sum(y_pred - Y)
            # update wights
            self.w -= dw * self.lr
            self.b -= dx * self.lr
    
    def predict(self, X):
        # apply sigmoid function to linear regression 
        return (self.sig(np.prod(self.w, X) + self.b) >= .5).astype('int')
    




# from flask import Flask
# from flask_restful import api, Resource


# app = Flask(__name__)
# api = api(app)


# class MyTestApi(Resource):
    
#     def get(self):
#         pass
    
#     def post(self):
#         pass
    
# api.add_resource(MyTestApi, '/myEndPoint')

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/', methods=["GET"])
def home():
    data = request.get_json()
    
    return jsonify(
        {
            'mesage':"Hello from API",
            "data": data
            }
        )

@app.route('/login', methods=["POST"])
def login():
    pass


import torch.nn as nn 


class MyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(10, 32)
        self.relu = nn.ReLU() # activation function
        self.two = nn.Linear(32,1)
        
    def forward(self,x):
        x = self.relu(self.first(x))
        return self.two(x)
        
        
            
