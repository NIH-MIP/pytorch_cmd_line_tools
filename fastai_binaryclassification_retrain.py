#!/usr/bin/env python3

#author @t_sanf

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *



import argparse
from sklearn.metrics import confusion_matrix
import os
import datetime as dt

class BinaryImageClassification(object):

    '''uses fast.ai for binary image classification with command line interface for training on cluster
    writes out confusion matrix as well as val loss/
    '''

    def __init__(self,inpath='',savepath='',model_name='resnet',bs=1,img_size=224,lr=0.0004,epochs=1,cycle_mult=10, \
                 unfreeze=True,retrain_last=True,retrain_all=False):

        '''
        :param inpath (str) path to imagefolder that has already been processed ad
        :param savepath (str) path to the location you want to save the file
        :param model_name (str) - right now only supports resnets
        :param bs: (int) batch size
        :param img_size: (int) the size of the image
        :param lr (float) learning rate
        :param epochs (int) number of training for last layer only
        :param cycle_mult (int) number of cycles to train full model on
        :param unfreeze (bool) weather or not to unfreeze the last layer
        :param retrain_last (bool) weather or not to unfreeze all layers
        :param retrain_all (bool) weather to finetune the entire network
        '''
        self.inpath=inpath
        self.savepath=savepath
        self.model_name=model_name
        self.bs=bs
        self.img_size=img_size
        self.lr=lr
        self.epochs=epochs
        self.cycle_mult=cycle_mult
        self.unfreeze=unfreeze
        self.retrain_last=retrain_last
        self.retrain_all=retrain_all

    def data_aug(self):

        tfms = tfms_from_model(self.model_name,self.img_size,transforms_side_on, max_zoom=1.1)
        return(tfms)


    def classifier_binary(self,precompute=True,save_dir=''):
        '''
        Trains a resnet image classifier that differentiates between two objects (i.e. cats/dogs)
        :param precompute(bool): if true, uses weights trained on imagenet
        model names are 'model_all' if all layers are used, and 'model lastlayer' if layers are frozen
        :return: model (saved) and
        model_dict={'resnet18':resnet18,'resnet34':resnet34,'resnet50':re
        '''
        model_dict = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101}

        #train the model
        tfms = self.data_aug()
        data = ImageClassifierData.from_paths(self.inpath, tfms=tfms)
        learn = ConvLearner.pretrained(model_dict[self.model_name], data, precompute=True)
        accuracy=learn.fit(self.lr, self.epochs)

        #save the weights
        learn.save(os.path.join(self.savepath,'model_lastlayer'))
        learn.load(os.path.join(self.savepath,'model_lastlayer'))

        if self.unfreeze==True:
            learn.unfreeze()

            lr = np.array([1e-6, 3e-6, 1e-5])
            learn.fit(lr, 3, cycle_len=1, cycle_mult=self.cycle_mult)

            learn.save(os.path.join(self.savepath,'model_all'))
            learn.load(os.path.join(self.savepath,'model_all'))

        return learn,data,accuracy

    def retrain_lastlayer(self,path):
        model_dict={'resnet18':resnet18,'resnet34':resnet34,'resnet50':resnet50,'resnet101':resnet101}


        tfms = self.data_aug()
        data = ImageClassifierData.from_paths(path, tfms=tfms)
        learn = ConvLearner.pretrained(model_dict[self.model_name], data, precompute=True)

        #load weights
        learn.load(os.path.join(self.savepath,'model_lastlayer'))
        accuracy = learn.fit(self.lr, self.epochs)
        learn.save(os.path.join(self.savepath, 'model_lastlayer'))
        learn.load(os.path.join(self.savepath, 'model_lastlayer'))

        return learn, data, accuracy

    def retrain_all_layers(self, path):
        model_dict={'resnet18':resnet18,'resnet34':resnet34,'resnet50':resnet50,'resnet101':resnet101}

        tfms = self.data_aug()
        data = ImageClassifierData.from_paths(path, tfms=tfms)
        learn = ConvLearner.pretrained(model_dict[self.model_name], data, precompute=False)

        learn.unfreeze()

        learn.load(os.path.join(self.savepath, 'model_all'))
        lr = np.array([1e-5, 3e-5, 1e-4])
        accuracy=learn.fit(lr, 3, cycle_len=1, cycle_mult=self.cycle_mult)
        learn.save(os.path.join(self.savepath, 'model_all'))
        learn.load(os.path.join(self.savepath, 'model_all'))

        os.chdir(self.savepath)
        self.save_metrics(learner=learn,data=data,accuracy=accuracy)

        return learn, data, accuracy

    def save_metrics(self,learner,data,accuracy):
        learn=learner
        log_preds = learn.predict()
        preds = np.argmax(log_preds, axis=1)
        probs = np.exp(log_preds[:, 1])

        log_preds, y = learn.TTA()
        probs = np.mean(np.exp(log_preds), 0)

        # save confusion matrix
        os.chdir(args.savepath)
        cm = confusion_matrix(y, preds)
        plot_confusion_matrix(cm, data.classes)
        plt.savefig('confusion_matrix' + '_' + drug_name + '_' + dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y"))

        file = open("accuracy_file" + '_' + drug_name + '_' + dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y"), "w")
        output = 'validation loss ' + "%.2f" % accuracy[0] + '\n' + 'precent correct ' + "%.2f" % accuracy[
            1] + '\n'
        file.write(output)
        file.close()


if __name__=="__main__":

    #parser
    parser = argparse.ArgumentParser(description='Train_Model')
    parser.add_argument('--inpath',default='/home/tom/Dropbox/Github/PyTorch/Image_Class_Detect/data/dogscats')  # image directory in ImageFolder format
    parser.add_argument('--savepath', default='changeme')  # where you want model saved
    parser.add_argument('--model_name', '--mn ', default='resnet18', type=str)
    parser.add_argument('--num_class', '--nc', default=2, type=int)
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--cycle_mult', default=10, type=int)
    parser.add_argument('--input_sz', default=224, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--unfreeze', default=False, type=bool)
    parser.add_argument('--init_train', default='False', type=str)
    parser.add_argument('--retrain_last', default=True, type=bool)
    parser.add_argument('--retrain_all',default=True,type=bool)
    args = parser.parse_args()

    #create list of paths to all drugs in the directory
    list_files= [os.path.join(args.inpath,file) for file in os.listdir(args.inpath)]

    # define first and last paths
    first_path=list_files[0]; remaining_paths=list_files[1:]

    #initially train just once, save model
    c=BinaryImageClassification(inpath=first_path,savepath=args.savepath,model_name=args.model_name,bs=args.bs,\
                                img_size=args.input_sz,lr=args.lr,epochs=args.epochs, cycle_mult=args.cycle_mult,\
                                unfreeze=args.unfreeze, retrain_last=args.retrain_last,retrain_all=args.retrain_all)



    if args.init_train=='False':
        pass
        print('skipping first drug')
    else:
        learn,data,accuracy=c.classifier_binary(args)

    for drug_path in remaining_paths:
        drug_name=str(os.path.basename(os.path.normpath(drug_path)))
        print(drug_name)

        if args.retrain_last==True:
            learn, data, accuracy =c.retrain_lastlayer(drug_path)

        if args.retrain_all==True:
            learn, data, accuracy = c.retrain_all_layers(drug_path)
            os.chdir(args.savepath)
            c.save_metrics(learn, data, accuracy)


    log_preds = learn.predict()
    preds = np.argmax(log_preds, axis=1)
    probs = np.exp(log_preds[:, 1])

    log_preds, y = learn.TTA()
    probs = np.mean(np.exp(log_preds), 0)

    #save confusion matrix
    os.chdir(args.savepath)
    cm = confusion_matrix(y, preds)
    plot_confusion_matrix(cm, data.classes)
    plt.savefig('confusion_matrix'+'_'+ drug_name+'_'+dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y"))


    file = open("accuracy_file" +'_'+ drug_name+ '_' + dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y"), "w")
    output = 'validation loss ' + "%.2f" % accuracy[0] + '\n' + 'percent correct ' + "%.2f" % accuracy[
        1] + '\n' + dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    file.write(output)
    file.close()

#wd /home/tom/Dropbox/Fellowship/Research/Prediction/gene_heat_map/code/python_code/fastai_model
