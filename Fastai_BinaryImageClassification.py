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



    def __init__(self,inpath='',outname='',model_name='resnet',bs=1,img_size=24,lr=0.001,epochs=1,unfreeze=False):
        '''
        :param inpath (str) path to imagefolder that has already been processed ad
        :param outname (str) the name of the file
        :param model_name (str) - right now only supports resnets
        :param bs: (int) batch size
        :param img_size: (int) the size of the image
        :param lr (float) learning rate
        :param epochs (int) number of training
        :param unfreeze (bool) weather or not to unfreeze the last layer
        '''
        self.inpath=inpath
        self.outpath=outname
        self.model_name=model_name
        self.bs=bs
        self.img_size=img_size
        self.lr=lr
        self.sz=img_size
        self.epochs=epochs
        self.unfreeze=unfreeze

    def data_aug(self):

        tfms = tfms_from_model(self.model_name,self.sz,transforms_side_on, max_zoom=1.1)
        return(tfms)


    def classifier_binary(self,precompute=True):
        '''
        Trains a resnet image classifier that differentiates between two objects (i.e. cats/dogs)

        :param precompute(bool): if true, uses weights trained on imagenet
        :return: model (saved) and
        '''

        model_dict={'resnet18':resnet18,'resnet34':resnet34,'resnet50':resnet50,
            'resnet101':resnet101}

        tfms=self.data_aug()
        data = ImageClassifierData.from_paths(self.inpath, tfms=tfms)
        learn = ConvLearner.pretrained(model_dict[self.model_name], data, precompute=precompute)
        accuracy=learn.fit(self.lr, self.epochs)

        learn.save('224_lastlayer')
        learn.load('224_lastlayer')

        if self.unfreeze==True:
            learn.unfreeze()

            lr = np.array([1e-4, 3e-4, 1e-3])
            learn.fit(lr, 3, cycle_len=1, cycle_mult=1)

            learn.save('224_all')
            learn.load('224_all')

        return learn,data,accuracy


if __name__=="__main__":

    #parser
    parser = argparse.ArgumentParser(description='Train_Model')
    parser.add_argument('--inpath',default='/home/tom/Dropbox/Github/PyTorch/Image_Class_Detect/data/dogscats')  # image directory in ImageFolder format
    parser.add_argument('--outname', default='kinda_sucky')  # where you want model saved
    parser.add_argument('--model_name', '--mn ', default='resnet18', type=str)
    parser.add_argument('--num_class', '--nc', default=2, type=int)
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--input_sz', default=224, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--unfreeze', default=False, type=bool)
    parser.add_argument('--feat_ext', default=True,type=bool)  # when false, finetune entire model, if true only reshaped layers
    parser.add_argument('--pretrain', default=True, type=bool)
    args = parser.parse_args()

    c=BinaryImageClassification(inpath=args.inpath,outname=args.outname,model_name=args.model_name,bs=args.bs,img_size=args.input_sz,lr=args.lr,epochs=args.epochs,unfreeze=args.unfreeze)
    learn,data,accuracy=c.classifier_binary(args)

    log_preds = learn.predict()
    preds = np.argmax(log_preds, axis=1)
    probs = np.exp(log_preds[:, 1])

    log_preds, y = learn.TTA()
    probs = np.mean(np.exp(log_preds), 0)

    cm = confusion_matrix(y, preds)
    plot_confusion_matrix(cm, data.classes)
    plt.savefig('confusion_matrix'+'_'+dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y"))

    file = open("accuracy_file" + '_' + dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y"), "w")
    output = 'validation loss ' + "%.2f" % accuracy[0] + '\n' + 'precent correct ' + "%.2f" % accuracy[
        1] + '\n' + dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    file.write(output)
    file.close()




