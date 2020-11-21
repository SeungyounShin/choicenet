import torch
from models.choicenet_reg import *
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera

def uniform_corruptions_data(n_samples=1000, show=False, corrupt=True, corruption_ratio = 0.1):
    # target
    eps = np.random.normal(size=(n_samples))
    x = np.random.uniform(-2, 5, n_samples)
    y = 2*np.sin(x) + eps/33

    if corrupt:
        x_corrupt = np.random.uniform(3, 4.3, int(n_samples*corruption_ratio))
        y_corrupt = np.random.uniform(-3, 3, int(n_samples*corruption_ratio))

    x = np.concatenate([x,x_corrupt])
    y = np.concatenate([y,y_corrupt])

    if show:
        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, alpha=0.5)
        plt.show()
    return torch.tensor(x).float(), torch.tensor(y).float()

def train(model, epoches=1000):
    fig, axes = plt.subplots(1)
    camera = Camera(fig)
    clip_value = 1.0
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 ,betas = (0.9,0.999),eps=1e-1)
    x, y = uniform_corruptions_data()
    x,y = x.view(-1,1),y.view(-1,1)
    for epoch in range(epoches):
        out = model(x)
        loss = model.compute_loss(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(model.rho[0].weight.grad)
        #print(model.pi[0].weight.grad)
        #print(model.varOut[0].weight.grad)
        if epoch %50 == 0:
            print(epoch, loss.data)
            if epoch %100 == 0:
                x_line = np.arange(-2,5,0.01)
                y_test = model.sampler(torch.tensor(x_line).view(-1,1).float())
                axes.scatter(x, y, c='r', s=3)
                axes.scatter(x_line, y_test, alpha=0.5, c='b',s=3)
                camera.snap()
    animation = camera.animate(interval=50, blit=True)
    animation.save(
        './result/choicenet_reg_result.gif',
        dpi=100,
        savefig_kwargs={
            'frameon': False,
            'pad_inches': 'tight'
        }
    )
    return model

if __name__=="__main__":
    model = ChoiceNetRegression(xdim=1, ydim=1).float()
    model = train(model, epoches=50000)
