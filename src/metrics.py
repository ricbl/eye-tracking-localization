import collections
from sklearn.metrics import roc_auc_score
import numpy as np
from .list_label import list_labels as label_names

class Metrics():
    def __init__(self, threshold_ior):
        self.values = collections.defaultdict(list)
        self.score_fn = roc_auc_score 
        self.registered_score = {}
        self.registered_iou = {}
        self.threshold_ior = threshold_ior
    
    def add_list(self, key, value):
        value = value.detach().cpu().tolist()
        self.values[key] += value
        
    def add_value(self, key, value):
        value = value.detach().cpu()
        self.values[key].append( value)
    
    def calculate_average(self):
        self.average = {}
        for key, element in self.values.items():
            if key[:7]=='y_true_' or key[:12]=='y_predicted_' or key[:11]=='iou_metric_' or key[:13]=='iou_consider_':
                continue
            n_values = len(element)
            if n_values == 0:
                self.average[key] = 0
                continue
            sum_values = sum(element)
            self.average[key] = sum_values/float(n_values)
        
        for suffix in self.registered_score.keys():
            y_true = np.array(self.values['y_true_' + suffix])
            y_predicted = np.array( self.values['y_predicted_' + suffix])
            auc_average = []
            for i in range(y_true.shape[1]):
                if (y_true[:,i]>0).any():
                    self.average['score_' + label_names[i].replace(' ','_') + '_' + suffix] = self.score_fn(y_true[:,i], y_predicted[:,i])
                    auc_average.append(self.average['score_' + label_names[i].replace(' ','_') + '_' + suffix])
            self.average['auc_average_' + suffix] = sum(auc_average)/len(auc_average)
        for suffix in self.registered_iou.keys():
            metric = np.array(self.values['iou_metric_' + suffix])
            consider = np.array( self.values['iou_consider_' + suffix])
            iou_average = []
            for i in range(metric.shape[1]):
                if (consider[:,i]>0).any():
                    if self.threshold_ior:
                        self.average['iou_' + label_names[i].replace(' ','_') + '_' + suffix] = ((metric[:,i]>0.1)*1).sum()/consider[:,i].sum()
                    else:
                        self.average['iou_' + label_names[i].replace(' ','_') + '_' + suffix] = (metric[:,i]).sum()/consider[:,i].sum()
                    iou_average.append(self.average['iou_' + label_names[i].replace(' ','_') + '_' + suffix])
            self.average['iou_average_' + suffix] = sum(iou_average)/len(iou_average)
        self.values = collections.defaultdict(list)
    
    def get_average(self):
        self.calculate_average()
        return self.average
    
    def add_iou(self, name, metric_value, consider):
        self.registered_iou[name] = 0
        metric_value = metric_value.detach()
        consider = consider.detach()
        self.add_list('iou_metric_' + name, metric_value)
        self.add_list('iou_consider_' + name, consider)
    
    # for adding predictions and groundtruths, used later to calculate accuracy.
    # The input epsilon is greater than 0 for predctions calculated against 
    # adversarial attacks
    def add_score(self, y_true, y_predicted, suffix):
        self.registered_score[suffix] = 0
        y_predicted = y_predicted.detach().squeeze(1)
        if len(y_true.size())>1:
            y_true = y_true.detach().squeeze(1)
        self.add_list('y_true_' + suffix, y_true)
        self.add_list('y_predicted_' + suffix, y_predicted)