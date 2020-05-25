import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features
        resnet_list = list(resnet.children())
        del resnet_list[3]
        resnet_list[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 参考论文 B.9 修改第一个卷积层并去掉第一个max pooling并去掉最后一层
        self.features = nn.Sequential(*resnet_list[:-1])    # 重建了一个resnet网络

        # projection MLP
        mid_out = num_ftrs//2
        self.l1 = nn.Linear(num_ftrs, mid_out)
        self.l2 = nn.Linear(mid_out, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x


if __name__ == "__main__":
    model = ResNetSimCLR('resnet18', 64)
    print(model)
