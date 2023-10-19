from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import pandas as pd
from numpy import asarray

from pathlib import Path



class MovingApples(data.Dataset):
    
    def __init__(self, root, 
                 train=True, 
                 split=0, 
                 transform=None, 
                 target_transform=None, 
                 download=False, 
                 process=False, 
                 size=32,
                 data_name='dummy_name'):
        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train  # training set or test set
        self.processed_folder = 'processed'
        self.size = size
        self.training_file = data_name+'_'+str(size)+'.pt'

        if download:
            raise RuntimeError('Apples Dataset can not be downloaded')

        if not self._check_exists(): 
            self.process()


        self.train_data = torch.load(
            os.path.join(Path(self.root).parent, 
                         self.processed_folder, 
                         self.training_file))

    def __getitem__(self, index):
        return self.train_data[index]


    def __len__(self):
        return len(self.train_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(Path(self.root).parent, 
                                           self.processed_folder, 
                                           self.training_file))

    def process(self):
        # process and save as torch files
        print('Processing...')

        #annotation_file = os.path.join('../../../Datasets/AppleSet/centeredApples_augmented/list_attr_apples.txt')
        annotation_file = os.path.join(self.root,'list_attr_apples.txt')
        annotations = pd.read_csv(annotation_file, sep=' ', header=0)

        # np.empty(shape(144,20,64,64))
        numpydata = None

        # size = (20,1,64,64)

        for i in range(0,20):
            apple = annotations.loc[annotations['evol_landmark']==i]
            evol = None
            # print(apple['img'])
            for filename in apple['img']:
                f = os.path.join(self.root, filename)
                if os.path.isfile(f):
                    image = Image.open(f)
                    new_image = image
                    new_image = image.resize((self.size, self.size))
                    #new_image = new_image.convert('L')
                    if evol is None:
                        evol = asarray(new_image)
                        evol = evol[np.newaxis, :, :]
                    else: 
                        individual = asarray(new_image)
                        individual = individual[np.newaxis, :, :]
                        evol = np.concatenate((evol, individual), axis=0)
                    # new_image.save('.\\DATA\\TGAN_images_dataset_64_BW\\'+filename, progressive=True)

        #     print(evol.shape)
        #     print(evol)

            if numpydata is None:
                numpydata = evol
                numpydata = numpydata[np.newaxis, :, :, :]
            else:
                evol = evol[np.newaxis, :, :, :]
                numpydata = np.concatenate((numpydata, evol), axis=0)

        np.save(os.path.join(Path(self.root).parent, (Path(self.root).name+'_'+str(self.size)+'.npy')), numpydata)
        training_set = torch.from_numpy(numpydata.swapaxes(0, 1))
        

        with open(os.path.join(Path(self.root).parent, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
