import torch
import torchvision
import types

# normalizes a batch of tensors according to mean and std
def BatchNormalizeTensor(mean, std, tensor):
        to_return = (tensor-mean)/std
        return to_return

# preprocess the inputs of a classifier to normailze them with ImageNet statistics
class ClassifierInputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return BatchNormalizeTensor(torch.FloatTensor([0.485, 0.456, 0.406]).cuda().view([1,3,1,1]), 
            torch.FloatTensor([0.229, 0.224, 0.225]).cuda().view([1,3,1,1]),(x).expand([-1,3,-1,-1]))

class PatchSlicing(torch.nn.Module):
    def __init__(self, grid_size = 8):
        super().__init__()
        self.grid_size = grid_size
    
    def forward(self, x):
        if x.size(2)>self.grid_size:
            x = torch.nn.functional.max_pool2d(x, x.size(2)-self.grid_size+1, 1,
                            0, 1, False,
                            False)
        elif x.size(2)<self.grid_size:
            x = torch.nn.functional.interpolate(x, scale_factor = self.grid_size/x.size(2), mode='bilinear', align_corners=True)
        return x

class RecognitionNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2048,512, 3, padding = 1,bias = False)
        self.bn = torch.nn.BatchNorm2d(512)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(512,10, 1)
        
    def forward(self,x, return_activations):
        x = self.conv1(x)
        x = self.bn(x)
        if return_activations:
            x1 = self.relu(x)
            x1 = self.conv2(x1)
            return x1, x
        else:
            x = self.relu(x)
            return self.conv2(x)

def forward_inference(out, normalize_fn):
    out = torch.sigmoid(out)
    return 1 - normalize_fn(1-out).view([out.size(0), out.size(1), -1]).prod(dim=2)

def forward_inferencemax(out, normalize_fn):
    return torch.sigmoid(out.view([out.size(0), out.size(1), -1]).max(dim=2)[0])

def p_box(out, box_label, normalize_fn):
    out = torch.sigmoid(out)
    return (normalize_fn(out)*box_label + (1-box_label)).view([out.size(0), out.size(1), -1]).prod(dim=2)*(normalize_fn(1-out)*(1-box_label) + (box_label)).view([out.size(0), out.size(1), -1]).prod(dim=2)

avgpool_pytorch = torch.nn.AdaptiveAvgPool2d((1,1))
ce_pytorch = torch.nn.BCEWithLogitsLoss()

class loss_ce(object):
    def __init__(self, threshold, weight_annotated, normalize_fn):
        pass
    
    def __call__(self, out, labels, box_label, contain_bbox):
        out = avgpool_pytorch(out).squeeze(3).squeeze(2)
        return ce_pytorch(out, labels)

class loss_fn_et(object):
    def __init__(self, threshold, weight_annotated, normalize_fn):
        self.threshold = threshold
        self.weight_annotated = weight_annotated
        self.normalize_fn = normalize_fn
    
    def __call__(self, out, labels, box_label, contain_bbox):
        out_sigmoid =  torch.sigmoid(out)
        fi = forward_inference(out, self.normalize_fn)
        contain_bbox = contain_bbox.unsqueeze(1)
        box_label = (box_label>self.threshold)*1.
        consider_data = ((box_label.view([out.size(0), out.size(1), -1]).sum(dim=2))>0)
        return - self.weight_annotated* (((contain_bbox)*torch.log(1-labels*consider_data*self.normalize_fn(1-out_sigmoid*box_label).view([out.size(0), out.size(1), -1]).prod(dim=2)+1e-20)).sum()
        + ((contain_bbox)*torch.log(self.normalize_fn(1-out_sigmoid*(1-labels.unsqueeze(2).unsqueeze(3))).view([out.size(0), out.size(1), -1]).prod(dim=2)+1e-20)).sum()) \
        - ((1 - contain_bbox)*(labels)*torch.log(fi+1e-20)).sum() \
            - ((1-contain_bbox)*(1-labels)*torch.log(1-fi+1e-20)).sum()

class loss_fn_etmax(object):
    def __init__(self, threshold, weight_annotated, normalize_fn):
        self.threshold = threshold
        self.weight_annotated = weight_annotated
    
    def __call__(self, out, labels, box_label, contain_bbox):
        contain_bbox = contain_bbox.unsqueeze(1)
        box_label = (box_label>self.threshold)*1.
        consider_data = ((box_label.view([out.size(0), out.size(1), -1]).sum(dim=2))>0)
        max_from_all = out.view([out.size(0), out.size(1), -1]).max(dim=2)[0]
        return self.weight_annotated*(contain_bbox*labels*consider_data*ce_pytorch((out*box_label + (1-box_label)*-1000).view([out.size(0), out.size(1), -1]).max(dim=2)[0],labels) + \
        contain_bbox*(1-labels)*ce_pytorch((out*(1-box_label) + (box_label)*-1000).view([out.size(0), out.size(1), -1]).max(dim=2)[0],labels)).sum() + \
        ((1-contain_bbox)*labels*ce_pytorch(max_from_all,labels)).sum() + \
        ((1-contain_bbox)*(1-labels)*ce_pytorch(max_from_all,labels)).sum()

class loss_fn_li(object):
    def __init__(self, threshold, weight_annotated, normalize_fn):
        self.weight_annotated = weight_annotated
        self.normalize_fn = normalize_fn
    
    def __call__(self, out, labels, box_label, contain_bbox):
        fi = forward_inference(out, self.normalize_fn)
        contain_bbox = contain_bbox.unsqueeze(1)
        return -self.weight_annotated*(contain_bbox*(torch.log(p_box(out,box_label, self.normalize_fn)+1e-20))).sum() \
            - ((1-contain_bbox)*labels*torch.log(fi+1e-20)).sum() \
            - ((1-contain_bbox)*(1-labels)*torch.log(1-fi+1e-20)).sum()

class Thoracic(torch.nn.Module):
    def __init__(self, grid_size = 8, pretrained = True, calculate_cam = False):
        super().__init__()
        self.preprocessing = ClassifierInputs()
        self.get_cnn_features = torchvision.models.resnet50(pretrained=pretrained)
        self.gradients = None
        self.activations = None
        self.calculate_cam = calculate_cam
        
        def _forward_impl(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x
        
        self.get_cnn_features._forward_impl = types.MethodType(_forward_impl, self.get_cnn_features)
        self.patch_slicing = PatchSlicing(grid_size)
        self.recognition_network = RecognitionNetwork()
    
    def forward(self,x):
        x = self.preprocessing(x)
        x = self.get_cnn_features(x)
        x = self.patch_slicing(x)
        if (x.requires_grad) and self.calculate_cam:
            x, activations = self.recognition_network(x, True)
            self.activations = activations
            h = activations.register_hook(self.activations_hook)
        else:
            x = self.recognition_network(x, False)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients.detach()
    
    def get_activations(self):
        return self.activations.detach()
