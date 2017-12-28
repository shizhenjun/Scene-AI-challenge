#coding:utf8
import torchvision as tv
from torch import nn
from basic_module import BasicModule
import torch as t
import torch.nn.functional as F

class DenseNet(BasicModule):
    """default size:224
    """
    def __init__(self,model,opt=None,feature_dim=2048,name='densenet'):
        super(DenseNet, self).__init__(opt)
        self.model_name=name

        num_ftrs = model.classifier.in_features
        del model.classifier
        model.classifier = lambda x:x
        self.features = model
        self.classifier = nn.Linear(num_ftrs,80)

    def forward(self,x):
        features = self.features(x)
        #features = F.adaptive_max_pool2d(features, output_size=1)
        logits = self.classifier(features)
        return F.softmax(logits), logits

def densenet121(opt):
    model = tv.models.densenet121(pretrained=not opt.load_path)
    return DenseNet(model,opt,feature_dim=1024,name='dense121')

def densenet169(opt):
    model = tv.models.densenet169(pretrained=not opt.load_path)
    return DenseNet(model,opt,feature_dim=1664,name='dense169')

def densenet201(opt):
    model = tv.models.densenet201(pretrained=not opt.load_path)
    return DenseNet(model,opt,feature_dim=1920,name='dense201')

def densenet161(opt):
    model = tv.models.densenet161(pretrained=not opt.load_path)
    return DenseNet(model,opt,feature_dim=2208,name='dense161')

def densenet161_365(opt):
    model = t.load('checkpoints/whole_densenet161_places365.pth.tar')
    #model = tv.models.densenet161()
    return DenseNet(model,opt,feature_dim=2208,name='dense161_365')
