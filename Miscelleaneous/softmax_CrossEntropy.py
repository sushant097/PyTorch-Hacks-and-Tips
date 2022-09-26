import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def cross_entropy(actual, preds):
    loss = -np.sum(actual * np.log(preds))
    return loss / float(preds.shape[0])

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print("softmax numpy: ", outputs)

# Torch Softmax
x = torch.tensor([2.0, 1.0, 0.1])
print(torch.softmax(x, dim=0))


###############
# Cross Entropy
###############
Y = np.array([1, 0, 0])
# y_pred has probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

# Torch Cross Entropy Loss
"""
nn.CrossEntropyLoss:
It applies nn.LogSoftmax + nn.NLLLoss (negative log likehood loss)
So, there is no Softmax in last Layer!
Y has class labels, not One-Hot!
Y_pred has raw scores (logits), no Softmax!

"""

loss = nn.CrossEntropyLoss()
Y = torch.tensor([2, 0, 1]) # not one-hot encoding
# Nsamplses x Nclasses = 3 x 3
Y_pred_good = torch.tensor([[0.1, 2.0, 3.0],
                            [2.0, 1.0, 0.1],
                            [2.0, 3.0, 0.1]
                            ]) # Not apply softmax
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1],
                            [0.1, 1.0, 2.1],
                            [2.0, 0.1, 3.1]
                            ]) # Not apply softmax

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item()) # 0.11..
print(l2.item()) # 0.76..

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1) # [2, 0, 1]
print(predictions2) # [0, 2, 2]