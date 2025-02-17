import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, 3)
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu"))
