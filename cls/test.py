from mmcls.models.backbones import RedNet
import torch
class Fucker(torch.nn.Module):
    def __init__(self, rednet):
        super(Fucker, self).__init__()
        self.backbone = rednet

    def forward(self, x):
        x = self.backbone
        return x

def test_rednet():
    rednet = RedNet(depth=26)
    #rednet.eval()
    inputs = torch.rand(1,3,32,32)
    out = rednet.forward(inputs)
    print(out.shape)

def load_model(model_path):
    rednet = RedNet(depth=26)
    fucker = Fucker(rednet)
    torch.save(fucker.state_dict(), "checkpoint/rednet26.pth")
    #print(fucker.state_dict())
    #rednet.load(to
    params = torch.load(model_path)['state_dict']
    #print(params['state_dict'])
    params.pop('head.fc.weight')
    params.pop('head.fc.bias')
    fucker.load_state_dict(params)
    torch.save(fucker.backbone, "checkpoint/rednet_backbone.pth")
    print(fucker)
    return fucker 

if __name__ == "__main__":
    #test_rednet()
    model_path = "/home/nathan/work/github/involution/cls/checkpoint/rednet26-4948f75f.pth"
    model = load_model(model_path)
    #print(model)
