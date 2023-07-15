import os

import pandas as pd
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset


class FetalPlaneDataset(Dataset):
    """Fetal Plane dataset."""

    def __init__(self,
                 root_dir,
                 csv_file,
                 plane,
                 brain_plane=None,
                 us_machine=None,
                 operator_number=None,
                 transform=None,
                 train=None,
                 train_size=100,
                 downsampling_factor=2
                 ):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            plane: 'Fetal brain'; 'Fetal thorax'; 'Maternal cervix'; 'Fetal femur'; 'Fetal thorax'; 'Other'
            brain_plane: 'Trans-ventricular'; 'Trans-thalamic'; 'Trans-cerebellum'
            us_machine: 'Voluson E6';'Voluson S10'
            operator_number: 'Op. 1'; 'Op. 2'; 'Op. 3';'Other'
            train: Flag denotes if test or train data is used
            train_size:  Limit dataset size to 100 images (for training)
            
        return image
        """
        self.transform = transform
        self.downsampling_factor = downsampling_factor
        self.root_dir = root_dir

        self.csv_file = pd.read_csv(csv_file, sep=';')
        self.csv_file = self.csv_file[self.csv_file['Plane'] == plane]
        if plane == 'Fetal brain':
            self.csv_file = self.csv_file[self.csv_file['Brain_plane'] == brain_plane]
        if us_machine is not None:
            self.csv_file = self.csv_file[self.csv_file['US_Machine'] == us_machine]
        if operator_number is not None:
            self.csv_file = self.csv_file[self.csv_file['Operator'] == operator_number]

        self.train_size = train_size
        if train:
            self.csv_file = self.csv_file[:self.train_size]
        else:
            self.csv_file = self.csv_file[self.train_size:]

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # Load the image from file

        # print(f'idx: {idx} \n')
        # print(f'self.csv_file.iloc[idx, 0]: {self.csv_file.iloc[idx, 0]} \n')

        img_name = os.path.join(self.root_dir,
                                self.csv_file.iloc[idx, 0] + '.png')
        # print(img_name)
        image = io.imread(img_name)  # <class 'numpy.ndarray'>

        # print(type(image))
        # print(image.dtype)

        # Preprocess and augment the image
        if self.transform:
            image = self.transform(image)

        ## Downsample image for SRGAN
        ds_image = resize(
            image.cpu().numpy(),
            (image.shape[0], image.shape[1] / self.downsampling_factor, image.shape[2] / self.downsampling_factor)
        )
        # .cpu().numpy()#TypeError: Cannot interpret 'torch.float32' as a data type

        return image, ds_image


class AfricanFetalPlaneDataset(Dataset):
    """African Fetal Plane dataset."""

    def __init__(self,
                 root_dir,
                 csv_file,
                 plane=None,
                 country=None,
                 transform=None,
                 train=None,
                 train_size=100,
                 downsampling_factor=2
                 ):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            plane: 'Fetal brain', 'Fetal abdomen','Fetal femur', 'Fetal thorax'
            country: 'Algeria', 'Egypt', 'Malawi', 'Uganda', 'Ghana'
            transform (callable, optional): Optional transform to be applied on a sample.
            train: Flag denotes if test or train data is used
            train_size:  Limit dataset size to 100 images (for training)
            downsampling_factor: downsampling image

        return image, downsampled_image
        """
        self.transform = transform
        self.downsampling_factor = downsampling_factor
        self.root_dir = root_dir
        self.plane = plane
        self.country = country
        self.csv_file = pd.read_csv(csv_file, sep=',')
        self.train_size = train_size

        if self.plane is not None:
            self.csv_file = self.csv_file[self.csv_file['Plane'] == self.plane]
        if self.country is not None:
            self.csv_file = self.csv_file[self.csv_file['Center'] == self.country]

        if train:
            self.csv_file = self.csv_file[:self.train_size]
        else:
            self.csv_file = self.csv_file[self.train_size:]

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # Load the random image from file

        # print(f'idx: {idx} \n')
        # print(f'self.csv_file.iloc[idx, 4]: {self.csv_file.iloc[idx, 4]} \n')#Filename

        img_name = os.path.join(self.root_dir,
                                self.csv_file.iloc[idx, 4] + '.png') #Filename
        image = io.imread(img_name)  # <class 'numpy.ndarray'>

        # print(type(image))
        # print(image.dtype)

        if self.transform:
            image = self.transform(image)

        ## Downsample image for SRGAN
        ds_image = resize(
            image.cpu().numpy(),
            (image.shape[0], image.shape[1] / self.downsampling_factor, image.shape[2] / self.downsampling_factor)
        )

        return image, ds_image
