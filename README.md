Tools for running deep learning models from command line (to make it easier to run on Biowulf cluster)

# create_imagefolder.py
command line program that puts all your files in Pytorch ImageFolder.  Expects .jpeg images with label after final underscore

flags:

--input_path      full path to directory with images

--output_path     full path to directory you want your ImageFolder output to go to. iMust be empty folder

--percent(float)  percent of total patients allocated for validation (default=0.2)

-- cat1_lab (str) name of first label (default='neg')

-- cat2_lab (str) name of 2nd label (default='pos')

-- balance (bool) if True, balances the dataset for you.  (default=True)

Example Usage:
1) Download file and navigate to folder in command line
2) type:
python3 create_imagefolder.py --input_path '/path_to_folder_with_images' --output_path '/path_to_output_folder' --percent=0.2

# torchvision_train.py
command line program that allows you to train neural networks using transfer learning in pytorch.

flags:


--image_dir        required, image directory in ImageFolder format (can use create_imagefolder script above to create

--out_dir          required, filepath to where you want your model saved

--mn (str)        name of model.  Options include resnet, resnet18 through resnet 152, alexnet, vgg, squeezenet.  Default resnet18

--nc (int)         num classes.  default 2

--bs (int)         batch size.  Default is 1

--num_epochs (int) number of training iterations, default 1

--lr (float)       learning rate.  Default is 0.001

-- mo (float)      momentum. default is 0.9

--input_size(int) size of input image, default 224 (imagenet size)

-- feat_ext        needs to be false (train entire model) in the case of non-imagenet models

-- pretrain        true for transfer learning

-- deconv          normalize for deconvolved images.  Default is False (normalize to imagenet).  Set to True for Dr Harmon Deconvolution


# Fastai_BinaryImageClassification.py
command line tool that uses fastai library with command line interface for easier training on cluster

flags:
-- inpath path to ImageFolder 

-- outname (not used right now)

-- model_name (str) right now only resnets supports (18 through 152)

-- bs (int) batch size

-- epochs (int): number training epochs

-- input_sz (int) size of image

-- lr (float) learning rate

-- unfreeze (bool) train the whole network

# fastai_binaryclassification_retrain.py
command line to that uses fastai library with command line intervace.  Expects multiple directories in parent directoy, each of which has files in imagefolder format.  Will train model across multiple imagefolder

--inpath (path) absolute path to 
--savepath (path) path to save the model
--model_name (str) one of many resnet models
--num_class (int) default is 2
--bs (int) batch size
--epochs (int) number of ephocs
--cycle_mult (int) number of cycles for finetuning
--input_sz (int) pixel size
--lr (float) learning rate
--unfreeze (bool) True if you want to unfreeze
--init_train 
--retrain_last
--retrain_all


