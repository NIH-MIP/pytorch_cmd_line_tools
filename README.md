Tools for running pytorch from command line (to make running on cluster easier)

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


--image_dir        image directory in ImageFolder format (can use create_imagefolder script above to create

--out_dir          filepath to where you want your model saved

--mn (path)        name of model.  Options include resnet, resnet18 through resnet 152, alexnet, vgg, squeezenet.  Default resnet18

--nc (int)         num classes.  default 2

--num_epochs (int) number of training iterations, default 1


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
