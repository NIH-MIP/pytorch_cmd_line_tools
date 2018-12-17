#!/usr/bin/env python3

#author @t_sanf

import os
import pandas as pd
import shutil
import argparse
import pdb
import re
import random


class CreateImageFolder:
    '''
    Class to create pytorch ImageFolder format automatically from a folder full of images, meant to be
    run from command line
    the label for the data should be in the filename somewhere (i.e. should have 'neg' somewhere in the filename
    note - out directory should be an empty file
    optional flag to perform simple oversamling of minority class
    '''

    def __init__(self,path_to_input='',path_to_output='',percent=0.2,cat1='neg',cat2='pos'):
        '''
        Args:
            path_to_input: path to the folder containing images.  Expect them to have image extension (i.e. .jpg) and label after final underscore
            path_to_output: the path where you want your image folder saved
            percent: percent of patients in validation (if you have 100 patients and this value is 0.2, you will have 20 validation patients)
        '''

        self.inpath=path_to_input
        self.outpath=path_to_output
        self.percent=percent
        self.cat1=cat1
        self.cat2=cat2

    def create_image_folder(self):
        '''creates a pytorch ImageFolder format from list of files in folder'''

        #use function to below to make sure all files present in output directory
        self.make_out_dir(cat1=self.cat1,cat2=self.cat2)

        #get all file paths
        all_file_path=pd.Series([os.path.join(self.inpath,file) for file in os.listdir(self.inpath)])

        #split into training and test tests, place into dictionary
        val=all_file_path.sample(frac=self.percent, random_state=200)
        train=all_file_path.drop(val.index)
        split={'train':train.tolist(),'val':val.tolist()}

        #iteratve over train/val data and copy files into negative and positive based on filename
        for state in split.keys():
            state_item=split[state]
            for path in state_item:
                neg=re.compile(self.cat1); pos=re.compile(self.cat2)
                if neg.search(os.path.basename(path)):
                    shutil.copy2(path,os.path.join(self.outpath,state,self.cat1))
                if pos.search(os.path.basename(path)):
                    shutil.copy2(path, os.path.join(self.outpath, state, self.cat2))


    def make_out_dir(self,cat1,cat2):
        '''
        looks through all the values in the output directory and makes the correct data structure and makes folderss
        for all necessary folder for pytorch ImageFolder.
        '''

        #define all directories
        states=['train','val']

        for state in states:
            path_state_dir=os.path.join(self.outpath,state)
            if not os.path.exists(path_state_dir):
                os.mkdir(path_state_dir)
                os.chdir(path_state_dir)
                if not os.path.exists(os.path.join(path_state_dir,cat1)):
                    os.mkdir(os.path.join(path_state_dir,cat1))
                if not os.path.exists(os.path.join(path_state_dir,cat2)):
                    os.mkdir(os.path.join(path_state_dir,cat2))

    def balance_dataset(self,cat1,cat2):
        '''
        Figures out which subset of data is the smallest and oversamples until data are balanced
        :return:
        '''
        diff=1 #initialization
        while diff>=0:

            cat1_num=len(os.listdir(os.path.join(self.outpath,'train',cat1)))
            cat2_num = len(os.listdir(os.path.join(self.outpath, 'train', cat2)))

            if cat1_num==cat2_num:
                break
            elif cat1_num>cat2_num:
                smaller=cat2; larger_num=cat1_num;smaller_num=cat2_num
            else:
                smaller=cat1; larger_num=cat2_num;smaller_num=cat1_num

            diff=larger_num-smaller_num
            print("difference classes is {}".format(diff))
            output=[os.path.join(self.outpath,'train',smaller,file) for file in os.listdir(os.path.join(self.outpath,'train',smaller))]
            output_series=pd.Series(output)
            val = output_series.sample(diff,replace=True)
            print('performing oversampling!')

            for file in val:
                jpeg_removed = file.split('.jpeg')[0]
                random_num=str(''.join(random.sample('0123456789', 5)))
                new_filename=jpeg_removed+random_num+'.jpeg'
                shutil.copy2(os.path.join(self.outpath,'train',file),os.path.join(self.outpath,'train',new_filename))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create ImageFolder')
    parser.add_argument('--input_path')
    parser.add_argument('--output_path',default='')
    parser.add_argument('--percent', default=0.2, type=float)
    parser.add_argument('--cat1_lab', default='neg', type=str)
    parser.add_argument('--cat2_lab',default='pos', type=str)
    parser.add_argument('--balance', default=True, type=bool)
    args = parser.parse_args()

    c=CreateImageFolder(args.input_path,args.output_path,args.percent,args.cat1_lab,args.cat2_lab)
    c.create_image_folder()
    if args.balance==True:
        c.balance_dataset(args.cat1_lab,args.cat2_lab)
