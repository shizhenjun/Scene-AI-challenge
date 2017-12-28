#coding:utf8
import torchvision as tv
from torch import nn
from basic_module import BasicModule
import torch as t
import torch.nn.functional as F
from Inception_ori import inception_v3_ori

class Inception(BasicModule):
    def __init__(self,model,opt=None,feature_dim=2048,name='densenet'):
        super(Inception, self).__init__(opt)
        self.model_name=name

        num_ftrs = model.fc.in_features
        del model.fc
        model.fc = lambda x:x        
        self.features = model
        self.classifier = nn.Linear(num_ftrs,80)

    def forward(self,x):
        features,_ = self.features(x)
        #features = F.adaptive_max_pool2d(features, output_size=1)
        logits = self.classifier(features)
        return F.softmax(logits), logits

def inception_v3(opt):
    """
    default size:299
    """
    model = inception_v3_ori(pretrained=not opt.load_path, aux_logits=True)
    return Inception(model,opt,feature_dim=2048,name='inception_v3')

#def densenet161_365(opt):
#    model = t.load('checkpoints/whole_densenet161_places365.pth.tar')
#    #model = tv.models.densenet161()
#    return DenseNet(model,opt,feature_dim=2208,name='dense161_365')
