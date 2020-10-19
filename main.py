import torch


def print_hi(name):
    print(f'torch.__version__={torch.__version__}')
    print(f'torch.cuda.is_available()={torch.cuda.is_available()}')
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('PyTorch')
