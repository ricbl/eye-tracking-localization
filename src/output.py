from tensorboardX import SummaryWriter
import logging
import os
import torch
import glob
import shutil
import sys
import csv

class Outputs():
    def __init__(self, opt, output_folder):
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        logging.basicConfig(filename = output_folder +'/log.txt' ,level = logging.INFO)
        self.log_configs(opt)
        self.writer = SummaryWriter(output_folder + '/tensorboard/')
        self.csv_file =  output_folder +'/log.csv' 
        with open(self.csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['key','value','epoch'])
        
    def log_configs(self, opt):
        logging.info('-------------------------------used configs-------------------------------')
        for key, value in sorted(vars(opt).items()):
            logging.info(key + ': ' + str(value).replace('\n', ' ').replace('\r', ''))
        logging.info('-----------------------------end used configs-----------------------------')
    
    # activate the average calculation for all metric values and save them to log and tensorboard
    def log_added_values(self, epoch, metrics):
        averages = metrics.get_average()
        logging.info('Metrics for epoch: ' + str(epoch))
        for key, average in averages.items():
            self.writer.add_scalar(key, average, epoch)
            logging.info(key + ': ' + str(average))
            with open(self.csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([key, str(average), str(epoch)])
        self.writer.flush() 
        return averages
    
    #save the source files used to run this experiment
    def save_run_state(self, py_folder_to_save):
        if not os.path.exists('{:}/src/'.format(self.output_folder)):
            os.mkdir('{:}/src/'.format(self.output_folder))
        [shutil.copy(filename, ('{:}/src/').format(self.output_folder)) for filename in glob.glob(py_folder_to_save + '/*.py')]
        self.save_command()
    
    #saves the command line command used to run these experiments
    def save_command(self, command = None):
        if command is None:
            command = ' '.join(sys.argv)
        with open("{:}/command.txt".format(self.output_folder), "w") as text_file:
            text_file.write(command)
    
    def save_models(self, net_d, suffix):
        torch.save(net_d.state_dict(), '{:}/state_dict_d_'.format(self.output_folder) + str(suffix)) 
