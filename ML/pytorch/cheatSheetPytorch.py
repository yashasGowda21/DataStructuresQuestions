# ===============================================
# 🧠 PyTorch Interview Cheatsheet with Theory + Code
# ===============================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------
# 🔹 1. What is PyTorch?
# PyTorch is an open-source deep learning framework developed by Facebook AI.
# It offers dynamic computation graphs and intuitive Pythonic APIs for building and training neural networks.

# -----------------------------------------------
# 🔹 2. What is a Tensor in PyTorch?
# A tensor is a multi-dimensional array (like NumPy), but with GPU acceleration.

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("Tensor:\n", x)

# -----------------------------------------------
# 🔹 3. How do you create a tensor that tracks gradients?
x = torch.tensor([2.0, 3.0], requires_grad=True)
print("Tensor with requires_grad:\n", x)

# -----------------------------------------------
# 🔹 4. How does autograd work?
# PyTorch builds a dynamic computation graph during the forward pass.
# When .backward() is called, gradients are automatically calculated.

y = x[0] ** 2 + x[1] ** 2
y.backward()
print("Gradients:\n", x.grad)

# -----------------------------------------------
# 🔹 5. What is retain_graph=True in backward?
# Used when you want to perform multiple backward passes on the same graph.

z = x[0] ** 3
z.backward(retain_graph=True)
print("Gradients after second backward:\n", x.grad)

# -----------------------------------------------
# 🔹 6. What is nn.Module? How to define a model?

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = Net()

# -----------------------------------------------
# 🔹 7. What is nn.Sequential used for?
# It's a quick way to stack layers linearly.

seq_model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# -----------------------------------------------
# 🔹 8. What are common loss functions in PyTorch?

mse_loss = nn.MSELoss()               # Regression
bce_loss = nn.BCELoss()               # Binary classification
cross_entropy_loss = nn.CrossEntropyLoss()  # Multi-class classification

# -----------------------------------------------
# 🔹 9. How do you use an optimizer?

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
sample_input = torch.randn(4, 10)
target = torch.randn(4, 1)

output = model(sample_input)
loss = mse_loss(output, target)
loss.backward()
optimizer.step()

# -----------------------------------------------
# 🔹 10. Sample PyTorch training loop

def train(model, dataloader, criterion, optimizer):
    for epoch in range(2):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

# -----------------------------------------------
# 🔹 11. How do you move model/data to GPU?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
sample_input = sample_input.to(device)
target = target.to(device)

# -----------------------------------------------
# 🔹 12. How do you save and load a model?

# Save:
torch.save(model.state_dict(), "model.pth")

# Load:
model.load_state_dict(torch.load("model.pth"))
model.eval()

# -----------------------------------------------
# 🔹 13. How do you create a custom Dataset?

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = MyDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# -----------------------------------------------
# 🔹 14. What is gradient clipping?

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# -----------------------------------------------
# 🔹 15. How do you freeze layers (transfer learning)?

for param in model.parameters():
    param.requires_grad = False

# -----------------------------------------------
# 🔹 16. How to use inference mode?

with torch.no_grad():
    prediction = model(torch.randn(1, 10).to(device))
    print("Inference:\n", prediction)

# -----------------------------------------------
# 🔹 17. What’s the difference between .cpu(), .cuda(), .to()?

tensor = torch.randn(2, 2)
tensor_cuda = tensor.cuda() if torch.cuda.is_available() else tensor
tensor_cpu = tensor_cuda.cpu()
tensor_to = tensor.to(device)

# -----------------------------------------------
# 🔹 18. What is the difference between PyTorch and TensorFlow?

"""
PyTorch: Eager execution by default (debug-friendly), dynamic graphs.
TensorFlow 1.x: Static graph; TF 2.x supports eager execution.
PyTorch is Pythonic; TensorFlow has better deployment tools (TF Serving, TFLite).
"""

# -----------------------------------------------
# 🔹 19. How to define a simple CNN?

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -----------------------------------------------
# 🔹 20. How to apply transfer learning using ResNet?

from torchvision import models

resnet = models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 2)  # 2 classes
resnet.to(device)

# -----------------------------------------------
# 🔹 21. What is TorchScript?
"""
TorchScript is a way to serialize models with code for deployment.
Use `torch.jit.script()` or `torch.jit.trace()` to convert models.
"""

# -----------------------------------------------
# 🔹 22. What is the difference between torch.save() and torch.jit.save()?
"""
torch.save(): Saves model weights (state_dict), not the code.
torch.jit.save(): Saves both model structure + weights (good for deployment).
"""

# -----------------------------------------------
# 🔹 23. What are hooks in PyTorch?
"""
Hooks let you inspect or modify gradients or activations.
Use `.register_forward_hook()` or `.register_backward_hook()`.
"""

# ===============================================
print("\n✅ PyTorch theory + code interview cheatsheet loaded successfully!")
