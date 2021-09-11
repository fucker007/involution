import torch
from involution_naive import involution

def test():
    in_c = 16
    inputs = torch.rand(1,in_c,32,32)
    inc = involution(channels=in_c, kernel_size=2, stride=2)
    outputs = inc(inputs)
    print(outputs.shape)


if __name__ == "__main__":

    test()

