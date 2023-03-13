import torch
from torch.distributions import Normal, Categorical


def normal():
    d = Normal(torch.Tensor([1.0]), torch.Tensor([2.0]))
    mu, sigma = torch.tensor([0.2], requires_grad=True), torch.tensor([1.0], requires_grad=True)
    d2 = Normal(mu, sigma)
    # d.log_prob(d.sample(torch.Size((10,))))
    
    optimizer = torch.optim.SGD([mu, sigma], lr=0.1)
    loss_func = lambda x: -torch.mean(d2.log_prob(x))
    X = d.sample(torch.Size((1000, )))
    for i in range(1000):
        loss = loss_func(X)
        print(mu, sigma, loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


d = Categorical(torch.Tensor([0.3, 0.5, 0.2]))

ps = torch.tensor([0.32, 0.33, 0.35], requires_grad=True)
# d2 = Categorical(ps)
# d.log_prob(d.sample(torch.Size((10,))))

optimizer = torch.optim.SGD([ps], lr=0.1)
loss_func = lambda ps, x: -torch.mean(Categorical(ps).log_prob(x))

X = d.sample(torch.Size((1000, )))
for i in range(1000):
    print(i)
    optimizer.zero_grad()
    loss = loss_func(ps, X)
    print(ps, loss)
    loss.backward()
    optimizer.step()
