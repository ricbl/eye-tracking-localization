# functions defining the calculation of gradcam heatmaps
import torch

#apply teh gradcam equation to calculated gradients and activation maps
def get_heatmap_index(gradients, activations):
    pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdims = True)
    map = activations * pooled_gradients
    heatmap = torch.nn.functional.relu(torch.sum(map, dim=1))
    return heatmap

def get_cam(model, nonspatial_predictions,  n_labels):
    pred_global = nonspatial_predictions
    grad_cam_all_labels = []
    for _, label_idx in enumerate(range(n_labels)):
        grad_cam_all_labels.append(get_heatmap(pred_global, label_idx, model).detach())
    return torch.stack(grad_cam_all_labels, dim=1)

# get gradcam heatmaps for one label
def get_heatmap(pred, label_idx, model):
    pred[:, label_idx].sum().backward(retain_graph = True)
    
    # get activations of the last spatial layer and the gradient of the outputs of the model with respect to the last spatial layer
    gradient = model.get_activations_gradient()
    activation = model.get_activations().detach()
    
    #calculate gradcam using its equation
    cams = get_heatmap_index(gradient, activation)
    
    cams = cams/torch.max(cams.view([cams.size(0),-1]), dim=1, keepdims = False)[0].unsqueeze(1).unsqueeze(1)
    
    return cams