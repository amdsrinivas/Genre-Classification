import os
import subprocess
import numpy as np
import json
#import model as EnsembleModel


class feature_extractor:
    def __init__(self, audio_paths = '/tmp/temp_audio_paths.txt', features_path='/tmp/temp_features.npy', audio_files_dir='./',workers = '1'):
        self.original_config_file = os.path.expanduser('~') + '/.keras/keras.json'
        with open(self.original_config_file) as f:
            self.original_config_data = json.load(f)
        with open('keras.json') as f:
            self.required_config_data = json.load(f)
        with open(self.original_config_file,'w') as f:
            json.dump(self.required_config_data,f)
        self.audio_paths = audio_paths
        self.features_path = features_path
        self.workers = workers
        self.features_command = 'python2 easy_feature_extraction.py ' + self.audio_paths + ' ' + self.features_path + ' ' + self.workers
        self.audio_files_dir = audio_files_dir

    def extract(self,files):
        with open(self.audio_paths, 'w+') as f:
            for file_item in files:
                f.write(self.audio_files_dir+file_item.strip()+'\n')
        try:
            temp = subprocess.check_output(self.features_command.split())
            features = np.load(self.features_path)
            os.remove(self.audio_paths)
            os.remove(self.features_path)
            self.revert_changes()
            return features
        except Exception as e:
            self.revert_changes()
            print(e)
    def revert_changes(self):
        with open(self.original_config_file,'w') as f:
            json.dump(self.original_config_data,f)
        
