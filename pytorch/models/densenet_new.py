#coding:utf8
#经过改变，该densenet可以接受任意尺度的输入
import torchvision as tv
from torch import nn
from basic_module import BasicModule
import torch as t
import torch.nn.functional as F
from Dense import densenet121, densenet169, densenet201, densenet161

class DenseNet(BasicModule):
    """default size:every is OK.
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
        logits = self.classifier(features)
        return F.softmax(logits), logits

def densenet121_new(opt):
    model = densenet121(pretrained=not opt.load_path)
    return DenseNet(model,opt,feature_dim=1024,name='dense121_new')

def densenet169_new(opt):
    model = densenet169(pretrained=not opt.load_path)
    return DenseNet(model,opt,feature_dim=1664,name='dense169_new')

def densenet201_new(opt):
    model = densenet201(pretrained=not opt.load_path)
    return DenseNet(model,opt,feature_dim=1920,name='dense201_new')

def densenet161_new(opt):
    model = densenet161(pretrained=not opt.load_path)
    return DenseNet(model,opt,feature_dim=2208,name='dense161_new')

def densenet161_365_new(opt):
    if opt.train:
        model = densenet161(num_classes=365)
        model.load_state_dict(t.load('checkpoints/whole_densenet161_places365_statedict.pth.tar'))
    else:
        model = densenet161()
    return DenseNet(model,opt,feature_dim=2208,name='dense161_365_new')
