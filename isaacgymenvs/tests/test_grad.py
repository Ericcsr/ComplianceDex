import torch
torch.manual_seed(0)
N = 100
x = torch.rand(N,1)*5
# Let the following command be the true function
y = 2.3 + 5.1*x
# Get some noisy observations
y_obs = y + 0.2*torch.randn(N,1)

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)


gamma = 0.01
for i in range(500):
    print(i)
    # use new weight to calculate loss
    y_pred = w * x + b
    mse = torch.mean((y_pred - y_obs) ** 2)

    # backward
    mse.backward()
    print('w:', w)
    print('b:', b)
    print('w.grad:', w.grad)
    print('b.grad:', b.grad)

    # gradient descent, don't track
    with torch.no_grad():
        w = w - gamma * w.grad
        b = b - gamma * b.grad
    w.requires_grad = True
    b.requires_grad = True