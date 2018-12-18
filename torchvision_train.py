#!/usr/bin/env python3

#author @t_sanf

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import datetime as dt
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def initialize_model(model_name, input_size, num_classes, feature_extract=True, use_pretrained=True):
    '''

    Args:
        model_name (str): one of the following 'resnet18','resnet34','resnet50','resnet101','resnet152'
        input_size(str):
        num_classes (int): usually
        feature_extract:
        use_pretrained (bool): This is the part that determines weather you tain the full dataset or not

    Returns: a model

    '''
    # Initialize these variables which will be set in this if statement. Each of these  variables is model specific.
    type_model={'resnet18':models.resnet18(pretrained=use_pretrained),
                'resnet34':models.resnet34(pretrained=use_pretrained),
                'resnet50':models.resnet50(pretrained=use_pretrained),
                'resnet101':models.resnet101(pretrained=use_pretrained),
                'resnet152':models.resnet152(pretrained=use_pretrained)
                }

    if model_name in type_model.keys():
        model_ft = type_model[model_name]
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = input_size

    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = input_size #224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = input_size #224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = input_size #224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = input_size #224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = input_size #224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = input_size #299


    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



def load_data(args):
    '''create dataset from image folder'''

    #check to see if you want to normalize using stephanie's deconvolved images
    if args.deconv==True:
        data_tfs=data_transforms_deconv(args)

    if args.deconv==False:
        data_tfs=data_transforms(args)


    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.image_dir, x), data_tfs[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.bs, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    return dataloaders_dict,device,criterion



def data_transforms(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(args.input_sz),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(args.input_sz),
            transforms.CenterCrop(args.input_sz),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms

def data_transforms_deconv(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(args.input_sz),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.75441, 0.83097, 0.87799], [0.15875, 0.13227, 0.1212])
        ]),
        'val': transforms.Compose([
            transforms.Resize(args.input_sz),
            transforms.CenterCrop(args.input_sz),
            transforms.ToTensor(),
            transforms.Normalize([0.75443, 0.83162, 0.8778], [0.15951, 0.13228, 0.12160])
        ]),
    }

    return data_transforms


def create_optimizer(args,model_ft,):

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if args.feat_ext:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum)

    return optimizer_ft

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(args,model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_log=[]
    train_counter=[]
    val_loss_log=[]
    val_counter=[]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model


            if phase == 'val':
                val_acc_history.append(epoch_acc)

            if phase=='train':
                train_counter.append(epoch)
                train_loss_log.append(epoch_loss)

            if phase=='val':
                val_counter.append(epoch)
                val_loss_log.append(epoch_loss)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    log_dict={'train_counter':train_counter,'train_loss_log':train_loss_log,'val_counter':val_counter,
              'val_loss_log':val_loss_log,'val_acc_history':val_acc_history,'best_acc':best_acc}



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, log_dict



if __name__=='__main__':
    #parser
    parser = argparse.ArgumentParser(description='Train_Model')
    parser.add_argument('--image_dir')     #image directory in ImageFolder format
    parser.add_argument('--out_dir')       #where you want model saved
    parser.add_argument('--model_name', '--mn ', default='resnet', type=str)
    parser.add_argument('--num_class', '--nc', default=2, type=int)
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum','--mo', default=0.9, type=float)
    parser.add_argument('--input_sz', default=224, type=int)
    parser.add_argument('--feat_ext', default=True,type=bool)  # when false, finetune entire model, if true only reshaped layers
    parser.add_argument('--pretrain', default=True, type=bool)
    parser.add_argument('--deconv', default=False, type=bool) #set to True if using Stephanie's deconvolved path images
    args = parser.parse_args()


    #initalize model
    model_ft, input_size = initialize_model(args.model_name,args.input_sz, args.num_class, args.feat_ext, args.pretrain)

    # load data
    dataloaders_dict, device, criterion_ft=load_data(args)

    #create optimizer
    optimizer_ft=create_optimizer(args,model_ft)

    print("training_model")

    model,log_dict=train_model(args,model=model_ft, dataloaders=dataloaders_dict, criterion=criterion_ft, optimizer=optimizer_ft,
                num_epochs=args.num_epochs, is_inception=False)

    print("saving model")
    torch.save(model, os.path.join(args.out_dir,args.model_name+'_'+dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y")))
    torch.save(model.state_dict(), os.path.join(args.out_dir,args.model_name+'_'+dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y")))
    torch.save(optimizer_ft.state_dict(), os.path.join(args.out_dir,args.model_name+'_'+dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y")))

    fig = plt.figure()
    plt.plot(log_dict['train_counter'], log_dict['train_loss_log'], color='blue')
    plt.plot(log_dict['val_counter'], log_dict['val_loss_log'], color='red')
    plt.legend(['Train Loss','Val_loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(args.out_dir+'/train_log'+dt.datetime.now().strftime("%I:%M%p_%B_%d_%Y"))
