import torch

def get_heatmap_index(gradients, activations):
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdims = True)
    map = activations * pooled_gradients
    
    heatmap = torch.sum(map, dim=1)
    return heatmap

def get_cam(model, nonspatial_predictions,  n_labels):
    pred_global = nonspatial_predictions
    a = []
    for i, label_idx in enumerate(range(n_labels)):
        a.append(get_heatmap(pred_global, label_idx, model).detach())
    return torch.stack(a, dim=1)

def get_heatmap(pred, label_idx, model):
    pred[:, label_idx].sum().backward(retain_graph = True)
    
    gradient = model.get_activations_gradient()
    activation = model.get_activations().detach()
    cams = get_heatmap_index(gradient, activation)
    return torch.maximum(cams, torch.tensor(0.))/torch.max(cams.view([cams.size(0),-1]), dim=1, keepdims = False)[0].unsqueeze(1).unsqueeze(1)