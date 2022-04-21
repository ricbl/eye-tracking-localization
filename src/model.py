# definitions of model architecture and losses
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

# layer from the Li et al. model used to resize the last spatial layer of the network to the desired grid size
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
    def __init__(self, last_layer_index = [4]):
        super().__init__()
        total_channels_last_layer = 0
        
        # the number of channels input to the RecognitionNetwork module
        # depend on which spatial layers from Resnet are being concatenateds
        if 4 in last_layer_index:
            total_channels_last_layer += 2048
        if 3 in last_layer_index:
            total_channels_last_layer += 1024
        if 2 in last_layer_index:
            total_channels_last_layer += 512
        if 1 in last_layer_index:
            total_channels_last_layer += 256
        self.conv1 = torch.nn.Conv2d(total_channels_last_layer,512, 3, padding = 1,bias = False)
        self.bn = torch.nn.BatchNorm2d(512)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(512,10, 1)
        
    def forward(self,x, return_activations):
        x = self.conv1(x)
        x = self.bn(x)
        if return_activations: # for grad-cam calculation
            x1 = self.relu(x)
            x1 = self.conv2(x1)
            return x1, x
        else:
            x = self.relu(x)
            return self.conv2(x)

def forward_inference(out, normalize_fn):
    out = torch.sigmoid(out)
    # normalize all probailities to a different range (e.g. [0.98,1]), to avoid underflow,
    # then calculate the probability of at least one cell in the grid being positive
    return 1 - normalize_fn(1-out).view([out.size(0), out.size(1), -1]).prod(dim=2)

def p_box(out, box_label, normalize_fn, use_balancing):
    out = torch.sigmoid(out)
    # balancing provides the number of grid cells being multiplied to normalize_fn
    # such that the normalization range of the probabilities being multiplied is around the same
    if use_balancing: 
        return (normalize_fn(out, box_label.view([out.size(0), out.size(1), -1]).sum(dim=2))*box_label + (1-box_label)).view([out.size(0), out.size(1), -1]).prod(dim=2)*(normalize_fn(1-out,(1-box_label).view([out.size(0), out.size(1), -1]).sum(dim=2))*(1-box_label) + (box_label)).view([out.size(0), out.size(1), -1]).prod(dim=2)
    else:
        return (normalize_fn(out)*box_label + (1-box_label)).view([out.size(0), out.size(1), -1]).prod(dim=2)*(normalize_fn(1-out)*(1-box_label) + (box_label)).view([out.size(0), out.size(1), -1]).prod(dim=2)

avgpool_pytorch = torch.nn.AdaptiveAvgPool2d((1,1))
ce_pytorch = torch.nn.BCEWithLogitsLoss()

def forward_inference_ce(out, normalize_fn):
    return torch.sigmoid(avgpool_pytorch(out).squeeze(3).squeeze(2))

class loss_ce(object):
    def __init__(self, threshold, weight_annotated, normalize_fn, use_balancing):
        pass
    
    def __call__(self, out, labels, box_label, contain_bbox):
        out = avgpool_pytorch(out).squeeze(3).squeeze(2) # use average pooling over the spatial grid as regular resnet
        return ce_pytorch(out, labels)

class loss_fn_li(object):
    def __init__(self, threshold, weight_annotated, normalize_fn, use_balancing):
        self.weight_annotated = weight_annotated
        self.normalize_fn = normalize_fn
        self.threshold = threshold
        self.use_balancing = use_balancing
    
    def __call__(self, out, labels, box_label, contain_bbox):
        fi = forward_inference(out, self.normalize_fn)
        
        # get binary representation of the eye-tracking heatmap, or ellipses
        box_label = (box_label>torch.tensor(self.threshold, device = box_label.device)[None, :, None, None])*1.
        
        contain_bbox = contain_bbox.unsqueeze(1)
        
        # If spatial annotation present, all cells with positive annotation
        # should be positive and all cells with negative annotation should be
        # negative. Else:
            # if image-level label positive, at least one cell should be positive
            # if image-level label negative, all cells should be negative
        return -self.weight_annotated*(contain_bbox*(torch.log(p_box(out,box_label, self.normalize_fn, self.use_balancing)+1e-20))).sum() \
            - ((1-contain_bbox)*labels*torch.log(fi+1e-20)).sum() \
            - ((1-contain_bbox)*(1-labels)*torch.log(1-fi+1e-20)).sum() 

# full network used for the experiments
class Thoracic(torch.nn.Module):
    def __init__(self, grid_size = 8, pretrained = True, calculate_cam = False, last_layer_index= [4]):
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
            layer_output_max = 0
            
            #concatenating all spatial layers in last_layer_index to include in the output of the module
            list_of_outputs = []
            x = self.layer1(x)
            if 1 in last_layer_index:
                if x.size(2)>layer_output_max:
                    layer_output_max = x.size(2)
                list_of_outputs.append(torch.nn.functional.interpolate(x,layer_output_max))
            x = self.layer2(x)
            if 2 in last_layer_index:
                if x.size(2)>layer_output_max:
                    layer_output_max = x.size(2)
                list_of_outputs.append(torch.nn.functional.interpolate(x,layer_output_max)) # resizing the ayer output to the spatial size of the largest layer size
            x = self.layer3(x)
            if 3 in last_layer_index:
                if x.size(2)>layer_output_max:
                    layer_output_max = x.size(2)
                list_of_outputs.append(torch.nn.functional.interpolate(x,layer_output_max))
            x = self.layer4(x)
            if 4 in last_layer_index:
                if x.size(2)>layer_output_max:
                    layer_output_max = x.size(2)
                list_of_outputs.append(torch.nn.functional.interpolate(x,layer_output_max))
            to_return = torch.cat(list_of_outputs, dim = 1)
            
            return to_return
        
        # changing the forward pass of the Resnet newotrk, whil keeping the same weights
        self.get_cnn_features._forward_impl = types.MethodType(_forward_impl, self.get_cnn_features)
        
        self.patch_slicing = PatchSlicing(grid_size)
        self.recognition_network = RecognitionNetwork(last_layer_index)
    
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
    
    #methods used for gradcam calculation
    def activations_hook(self, grad):
        self.gradients = grad
    def get_activations_gradient(self):
        return self.gradients.detach()
    def get_activations(self):
        return self.activations.detach()
