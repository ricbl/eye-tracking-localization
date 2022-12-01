# script called for training and validating the proposed method
import torch
torch.backends.cudnn.benchmark = True
from . import model
from . import output as outputs
from . import metrics
from . import opts
from . import gradcam
from . import seq_model
import torchvision
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import contextlib
import torchvision.models as models
from itertools import chain

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

def init_optimizer(params, opt):
    if opt.optimizer == 'adamw':
        optimizer_d = torch.optim.AdamW(params, lr=0.001, betas=(
                    0.5, 0.999), weight_decay=opt.weight_decay)
    if opt.optimizer == 'adamams':
        optimizer_d = torch.optim.Adam(params, lr=0.001, weight_decay=opt.weight_decay, amsgrad=True)
    lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.1)
    return optimizer_d, lr_scheduler_d

# normalization of probabilities that will be multiplied. Used to avoid underflow
# If n_squares is not provided, use the normalization proposed by li et al, where all 
# probabilities are mapped to [0.98,1].
# n_squares can be provided for cases when the multiplication is not performed for all cells
# of a 16x16 grid. The value 0.0056738 was chosen such that 0.0056738**(1/(16*16))=0.98
def normalize_p_(p, n_squares=None):
    if n_squares is None:
        lower_bound = 0.98
    else:
        lower_bound = 0.0056738**(1/n_squares)
        lower_bound = lower_bound.unsqueeze(2).unsqueeze(2)
    return p*(1-lower_bound)+lower_bound

# generic training class that is inherited for the two training cycles used for 
# the paper
class TrainingLoop():
    def __init__(self, output,metric, opt):
        if opt.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        self.context_forward_pass = torch.cuda.amp.autocast() if opt.use_mixed_precision else contextlib.nullcontext()
        self.metric = metric
        self.output = output
        self.opt = opt
        self.normalize_fn = normalize_p_
        self.sequence_model = opt.sequence_model
        self.unet = opt.unet
        self.threshold = opt.threshold_box_label
        
        self.sigma_loss_multiplier = opt.sigma_loss_multiplier
        if opt.sigma_loss_multiplier>0:
            self.sigma_loss = LossSigma(opt).cuda()
        
        #defining how to calculate the loss and the image-level labels from the grid
        if opt.sigma_loss_multiplier>0 or self.opt.loss == 'ce': #loss only used for the unannotated model, simply using cross-entropy over the average-pooling of the grid
            self.loss_fn = model.loss_ce(opt.threshold_box_label, opt.weight_loss_annotated, self.normalize_fn, opt.use_grid_balancing_loss)
            self.forward = model.forward_inference_ce
        elif self.opt.loss == 'li': #loss as defined in the paper, as proposed by li et al.
            self.loss_fn = model.loss_fn_li(opt.threshold_box_label, opt.weight_loss_annotated, self.normalize_fn, opt.use_grid_balancing_loss)
            self.forward = model.forward_inference
        elif self.opt.loss == 'ce_grid':
            self.loss_fn = model.loss_ce_grid(opt.threshold_box_label, opt.weight_loss_annotated, self.normalize_fn, opt.use_grid_balancing_loss)
            self.forward = model.forward_inference
        
    #method that iterates through all epochs through all batches for training and validation sets
    def train(self, train_dataloader, val_dataloader_ann, val_dataloader_all, net_d):
        #load the optimizer
        if self.opt.sigma_loss_multiplier>0:
            optim_d, lr_scheduler_d = init_optimizer(opt=self.opt, params = chain(net_d.parameters(), self.sigma_loss.parameters()))
        else:
            optim_d, lr_scheduler_d = init_optimizer(opt=self.opt, params = net_d.parameters())
        
        last_best_validation_metric = self.opt.initialization_comparison
        
        for epoch_index in range(self.opt.nepochs):
            self.metric.start_time('epoch')
            if not self.opt.skip_train:
                self.metric.start_time('train')
                net_d.train()
                for batch_index, batch_example in enumerate(train_dataloader):
                    if batch_index%100==0:
                        print(batch_index)
                    image, label, contain_box, boxes = batch_example
                    image = image.cuda()
                    label = label.cuda().float()
                    contain_box = contain_box.cuda()
                    boxes = boxes.cuda()
                    # call the train_fn, to be defined 
                    # in the child classes. This function does the 
                    # training of the model for the current batch of examples
                    self.train_fn(image, label, contain_box, boxes,net_d,optim_d)
                self.metric.end_time('train')
            #validation
            if not self.opt.skip_validation:
                self.metric.start_time('validation')
                with torch.no_grad():
                    net_d.eval()
                    if self.opt.validate_iou:
                        for batch_index, batch_example in enumerate(val_dataloader_ann):
                            image, label, contain_box, boxes, pixelated_boxes, mimic_label = batch_example
                            image = image.cuda()
                            
                            label = label.cuda().float()
                            contain_box = contain_box.cuda()
                            boxes = boxes.cuda()
                            pixelated_boxes = pixelated_boxes.cuda()

                            mimic_label = mimic_label.cuda()
                            if batch_index%10==0:
                                print(batch_index)
                            # call the validation_fn function, to be defined 
                            # in the child classes, that defines what to do
                            # during the validation loop
                            self.validation_fn_ann(image, label, contain_box, boxes, pixelated_boxes, mimic_label, net_d)
                    if self.opt.validate_auc:
                        for batch_index, batch_example in enumerate(val_dataloader_all):
                            image, label  = batch_example
                            image = image.cuda()
                            label = label.cuda().float()
                            if batch_index%100==0:
                                print(batch_index)
                            # call the validation_fn function, to be defined 
                            # in the child classes, that defines what to do
                            # during the validation loop
                            self.validation_fn_all(image, label, net_d)
                    net_d.train()
                self.metric.end_time('validation')
            self.metric.end_time('epoch')
            self.metric.end_time('full_script')
            
            # get a dictionary containing the average metrics for this epoch, and writes all metrics to the log files
            average_dict = self.output.log_added_values(epoch_index, self.metric)
            
            if not self.opt.skip_train:
                if self.opt.use_lr_scheduler:
                    lr_scheduler_d.step()
                
                #if training, check if the model from the current epoch is the 
                # best model so far, and if it is, save it
                this_validation_metric = average_dict[self.opt.metric_to_validate]
                if self.opt.function_to_compare_validation_metric(this_validation_metric,last_best_validation_metric):
                    self.output.save_models(net_d, 'best_epoch')
                    last_best_validation_metric = this_validation_metric
                
                #save the model for the current epoch
                self.output.save_models(net_d, str(epoch_index))

class LossSigma(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.sigma = torch.nn.Parameter(torch.ones([]))
        self.enforce_negatives = opt.enforce_negatives_sigma
    
    def forward(self, cam, gt_localization, contain_box, label):
        if self.enforce_negatives:
            cases_to_enforce = torch.logical_or(contain_box[:,None,None,None], 1-label[:,:,None,None])
        else:
            cases_to_enforce = torch.logical_and(contain_box[:,None,None,None], label[:,:,None,None])
        return 0.5/(self.sigma**2)*(((cam-gt_localization)**2)*cases_to_enforce).mean() +torch.log(self.sigma)

#class defining how training and validation for each batch is performed
class SpecificTraining(TrainingLoop):
    def loss_unet(self, d_x, output_heatmap, activations, label, box_label, contain_box, gamma_unet, threshold):
        boxes = box_label
        if (contain_box[:,None,None,None]*label[:,:,None,None]).sum()>0:
            seg_loss = torch.nn.BCEWithLogitsLoss()(output_heatmap*contain_box[:,None,None,None]*label[:,:,None,None],boxes)
        else:
            seg_loss = 0.
        class_loss = self.loss_fn(d_x, label, torch.nn.AdaptiveMaxPool2d((self.opt.grid_size,self.opt.grid_size))(box_label), contain_box)
        
        return (gamma_unet)*seg_loss + (class_loss)
    
    def train_fn(self,image, label, contain_box, boxes,net_d,optim_d):
        
        for p in net_d.parameters(): p.grad = None
        
        with self.context_forward_pass:
            if not self.sequence_model:
                if self.unet:
                    d_x, activations, cam, _, output_heatmap  = net_d(image, return_activations = True)
                    classifier_loss = self.loss_unet(d_x, output_heatmap, activations, label, boxes, contain_box, self.opt.gamma_unet, self.threshold )
                    assert(self.sigma_loss_multiplier==0)
                else:
                    if self.sigma_loss_multiplier>0:
                        d_x, output_heatmap, cam, _, _ = net_d(image, return_activations = True)
                        classifier_loss = self.loss_fn(d_x, label, boxes, contain_box)
                        classifier_loss += self.sigma_loss_multiplier*self.sigma_loss(cam, boxes, contain_box, label)
                    else:
                        d_x = net_d(image)
                        classifier_loss = self.loss_fn(d_x, label, boxes, contain_box)
            else:
                d_x, classifier_loss, labels_input_all_iterations, lengths = net_d.loss(boxes.data, label.data, image, label.batch_sizes)
                for i in range(lengths.size(0)):
                    self.metric.add_value('lengths_train', lengths[i])
        
        if self.opt.use_mixed_precision:
            self.scaler.scale(classifier_loss).backward()
            self.scaler.step(optim_d)
            self.scaler.update()
        else:
            classifier_loss.backward()
            optim_d.step()
        self.metric.add_value('classifier_loss', classifier_loss)
        
        # adding model outputs for calculating training auc
        if not self.sequence_model:
            self.metric.add_score(label, self.forward(d_x, self.normalize_fn), 'train')
        else:
            self.metric.add_score(labels_input_all_iterations[:,:-2], d_x, 'train')
    
    #validation for annotated functions
    def validation_fn_ann(self,image, label, contain_box, boxes, pixelated_boxes, mimic_label, net_d):
        #saving model and gradient state to restore it after the end of validation
        original_model_mode = net_d.training
        prev_grad_enabled = torch.is_grad_enabled()
        
        if self.opt.calculate_cam:
            torch.set_grad_enabled(True)
        
        net_d.eval()
        if not self.sequence_model:
            if self.unet:
                nonspatial_predictions, _ , _, d_x, unet_output  = net_d(image, return_activations = True)
                nonspatial_predictions = self.forward(nonspatial_predictions, self.normalize_fn)
            else:
                d_x = net_d(image)
                # calculating the image-level outputs for the model and adding them to the validation AUC calculation
                nonspatial_predictions = self.forward(d_x, self.normalize_fn)
        else:
            d_x, lengths = net_d(image)
            for i in range(lengths.size(0)):
                self.metric.add_value('lengths_val_ann', lengths[i])
        
        # calculate iou for the last spatial layer of the network using several thresholds over the activations
        out_map = torch.sigmoid(d_x)
        if not self.sequence_model:
            if boxes.size(2)==512:
                out_map = torchvision.transforms.Resize(512, torchvision.transforms.InterpolationMode.NEAREST)(out_map)
            for threshold in self.opt.thresholds_iou:
                iou_1 = localization_score_fn(boxes, out_map,threshold)
                self.metric.add_iou(f'val_ellipse_iou_{threshold}', iou_1, label)
        if self.unet:
            # calculate iou for the last spatial layer of the network using several thresholds over the activations
            out_map_unet = torch.sigmoid(unet_output)
            if not self.sequence_model:
                if boxes.size(2)==512:
                    out_map_unet = torchvision.transforms.Resize(512, torchvision.transforms.InterpolationMode.NEAREST)(out_map_unet)
                for threshold in self.opt.thresholds_iou:
                    iou_2 = localization_score_fn(boxes, out_map_unet,threshold)
                    self.metric.add_iou(f'val_unet_iou_{threshold}', iou_2, label)
        if self.unet and (self.opt.load_checkpoint_d is None or not 'unannotated' in self.opt.load_checkpoint_d):
            out_map = out_map_unet
        self.metric.add_score(label, nonspatial_predictions, 'val_rad')
        self.metric.add_score(label, nonspatial_predictions, 'val_mimic_ann')
        
        #calculating cam can be a bit slow, so leave it turned off while training
        if self.opt.calculate_cam:
            
            cam = gradcam.get_cam(net_d, nonspatial_predictions, 10).detach()
            
            if boxes.size(2)==512:
                cam = torchvision.transforms.Resize(512, torchvision.transforms.InterpolationMode.NEAREST)(cam)
            for threshold in self.opt.thresholds_iou:
                iou = localization_score_fn(boxes, cam, threshold)
                self.metric.add_iou(f'val_cam_iou_{threshold}', iou.detach(), label.detach())
        
        # self.opt.draw_images is only true for when running scripts to generate examples to put in the paper
        if self.opt.draw_images:
            plt.imshow(image[0,0,...].cpu().numpy(), cmap='gray')
            plt.imshow((out_map[0,self.opt.label_to_draw,...].detach().cpu().numpy())*1., cmap='jet', alpha = 0.3)
            plt.axis('off')
            plt.savefig(f'{self.output.output_folder}/out_map{self.opt.sm_suffix}.png', bbox_inches='tight', pad_inches = 0)
            
            plt.imshow(image[0,0,...].cpu().numpy(), cmap='gray')
            plt.imshow(boxes[0][self.opt.label_to_draw].cpu().numpy(), cmap='jet', alpha = 0.3)
            plt.axis('off')
            plt.savefig(f'{self.output.output_folder}/gt{"".join([i for i in self.opt.sm_suffix.split() if i.isdigit()])}.png', bbox_inches='tight', pad_inches = 0)
        
        # restoring the state of the model and gradients before entering this method
        net_d.zero_grad()
        torch.set_grad_enabled(prev_grad_enabled)
        if original_model_mode:
            net_d.train()
    
    #validation for unannotated functions (mimic-cxr dataset)
    def validation_fn_all(self,image, label, net_d):
        if not self.sequence_model:
            if self.unet:
                nonspatial_predictions, _,_, _, _ = net_d(image, True)
                nonspatial_predictions = self.forward(nonspatial_predictions, self.normalize_fn)
            else:
                d_x = net_d(image)
                nonspatial_predictions = self.forward(d_x, self.normalize_fn)
        else:
            d_x, lengths, _ = net_d(image)
            for i in range(lengths.size(0)):
                self.metric.add_value('lengths_val_all', lengths[i])
        #adding output of model to calculate AUC
        if self.sequence_model:
            self.metric.add_score(label, d_x, 'val_mimic_all')
        else:
            self.metric.add_score(label, nonspatial_predictions, 'val_mimic_all')

# calculating iou score from the GPU tensors
def localization_score_fn(y_true, y_predicted, threshold):    
    y_predicted = (y_predicted>threshold)*1.
    intersection = (y_predicted*y_true).view([y_true.size(0), y_true.size(1), -1]).sum(axis=2)
    union = (torch.maximum(y_predicted,y_true)).view([y_true.size(0), y_true.size(1), -1]).sum(axis=2)
    iou = intersection/union
    # iou for images without ellipses is set to 0, and is not used for the calculation of the average iou
    iou =  torch.nan_to_num(iou)
    return iou

def main():
    #get user options/configurations
    opt = opts.get_opt()
    
    #load Outputs class to save metrics, images and models to disk
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
    output = outputs.Outputs(opt, opt.save_folder + '/' + opt.experiment + '_' + opt.timestamp)
    output.save_run_state(os.path.dirname(__file__))
    
    from .get_together_dataset import get_together_dataset as get_dataloaders
    
    #load class to store metrics and losses values
    metric = metrics.Metrics(opt.threshold_ior, opt.validate_auc)
    metric.start_time('full_script')
    if opt.skip_train:
        loader_train = None
    else:
        loader_train = get_dataloaders(split='train',type_ = opt.dataset_type, use_et = opt.use_et, crop = (opt.use_center_crop), batch_size = opt.batch_size, use_data_aug = opt.use_data_augmentation, num_workers = opt.num_workers, percentage_annotated=opt.percentage_annotated, percentage_unannotated=opt.percentage_unannotated, repeat_annotated = opt.repeat_annotated, load_to_memory=opt.load_to_memory, data_aug_seed = opt.data_aug_seed, index_produce_val_image = opt.index_produce_val_image, grid_size = opt.grid_size if not opt.unet else opt.grid_size*8, dataset_type_et = opt.dataset_type_et, sequence_model = opt.sequence_model, dataset_name = opt.dataset_name, type_toy=opt.type_toy, calculate_label_specific_heatmaps = opt.calculate_label_specific_heatmaps)
    loader_val_rad = get_dataloaders(split=opt.split_validation + '_ann',type_ = opt.dataset_type, use_et = opt.use_et, crop = (opt.use_center_crop), batch_size = opt.batch_size, use_data_aug = opt.use_data_augmentation, num_workers = opt.num_workers, percentage_annotated=1., percentage_unannotated=0., repeat_annotated = False, load_to_memory=opt.load_to_memory, data_aug_seed = opt.data_aug_seed, index_produce_val_image = opt.index_produce_val_image, grid_size = opt.grid_size if not opt.unet else opt.grid_size*8, dataset_type_et = opt.dataset_type_et, sequence_model = opt.sequence_model, dataset_name = opt.dataset_name, type_toy=opt.type_toy, calculate_label_specific_heatmaps = opt.calculate_label_specific_heatmaps)
    loader_val_all = get_dataloaders(split=opt.split_validation+ '_all',type_ = opt.dataset_type, use_et = opt.use_et, crop = (opt.use_center_crop), batch_size = opt.batch_size, use_data_aug = opt.use_data_augmentation, num_workers = opt.num_workers, percentage_annotated=0., percentage_unannotated=1., repeat_annotated = False, load_to_memory=opt.load_to_memory, data_aug_seed = opt.data_aug_seed, index_produce_val_image = opt.index_produce_val_image, grid_size = opt.grid_size if not opt.unet else opt.grid_size*8, dataset_type_et = opt.dataset_type_et, sequence_model = opt.sequence_model, dataset_name = opt.dataset_name, type_toy=opt.type_toy, calculate_label_specific_heatmaps = opt.calculate_label_specific_heatmaps)
    
    #load the deep learning architecture for the critic and the generator
    if opt.sequence_model:
        net_d = seq_model.Model(opt.checkpoint_sequence)
    else:
        net_d = model.Thoracic(opt.grid_size, pretrained = opt.use_pretrained, calculate_cam = opt.calculate_cam, last_layer_index = opt.last_layer_index).cuda()
    if opt.unet:
        net_d.add_decoder(model.DecodingNetwork().cuda())
    if opt.load_checkpoint_d is not None:
        net_d.load_state_dict(torch.load(opt.load_checkpoint_d))
    
    SpecificTraining(output,metric, opt).train(loader_train, loader_val_rad, loader_val_all,
          net_d=net_d)
    

if __name__ == '__main__':
    main()