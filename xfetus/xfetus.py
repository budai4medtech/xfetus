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
                 split_type=None,
                 split="train",
                 train_size=100,
                 downsampling_factor=2,
                 return_labels=False
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
            split_type (string, optional): Method to split dataset, supports "manual", and "csv"
            split (string): Which partition to return, supports "train", and "valid"
            train_size:  Limit dataset size to 100 images (for training)
            return_labels: Return the plane of the image 
            
        return image, downsampled_image
        """
        self.transform = transform
        self.downsampling_factor = downsampling_factor
        self.root_dir = root_dir
        self.csv_file = pd.read_csv(csv_file, sep=';')
        self.csv_file = self.csv_file[self.csv_file['Plane'] == plane]
        self.return_labels = return_labels

        if plane == 'Fetal brain':
            self.csv_file = self.csv_file[self.csv_file['Brain_plane'] == brain_plane]
        if us_machine is not None:
            self.csv_file = self.csv_file[self.csv_file['US_Machine'] == us_machine]
        if operator_number is not None:
            self.csv_file = self.csv_file[self.csv_file['Operator'] == operator_number]

        self.train_size = train_size
        
        # Split data manually
        if split_type == "manual":
            if split == "train":
                self.csv_file = self.csv_file[:self.train_size]
            elif split == "valid":
                self.csv_file = self.csv_file[self.train_size:]
        # Split data according to CSV
        elif split_type == "csv":
            if split == "train":
                self.csv_file = self.csv_file[self.csv_file['Train'] == 1]
            elif split == "valid":
                self.csv_file = self.csv_file[self.csv_file['Train'] == 0]

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
            image,
            (image.shape[0], image.shape[1] / self.downsampling_factor, image.shape[2] / self.downsampling_factor)
        )
        # .cpu().numpy()#TypeError: Cannot interpret 'torch.float32' as a data type

        # Return labels for classification task
        if self.return_labels:
            plane = self.csv_file.iloc[idx, 2]
            return image, ds_image, plane
        else:
            return image, ds_image


class AfricanFetalPlaneDataset(Dataset):
    """African Fetal Plane dataset."""

    def __init__(self,
                 root_dir,
                 csv_file,
                 plane=None,
                 country=None,
                 transform=None,
                 split_type=None,
                 split="Train",
                 train_size=100,
                 downsampling_factor=2,
                 return_labels=False
                 ):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            plane: 'Fetal brain', 'Fetal abdomen','Fetal femur', 'Fetal thorax'
            country: 'Algeria', 'Egypt', 'Malawi', 'Uganda', 'Ghana'
            transform (callable, optional): Optional transform to be applied on a sample.
            split_type (string, optional): Method to split dataset, supports "manual", and "csv"
            split (string): Which partition to return, supports "train", and "valid"
            train: Flag denotes if test or train data is used
            train_size:  Limit dataset size to 100 images (for training)
            downsampling_factor: downsampling image
            return_labels: Return the plane and country of the image 

        return image, downsampled_image
        """
        self.transform = transform
        self.downsampling_factor = downsampling_factor
        self.root_dir = root_dir
        self.plane = plane
        self.country = country
        self.csv_file = pd.read_csv(csv_file, sep=',')
        self.return_labels = return_labels

        if self.plane is not None:
            self.csv_file = self.csv_file[self.csv_file['Plane'] == self.plane]
        if self.country is not None:
            self.csv_file = self.csv_file[self.csv_file['Center'] == self.country]

        # Split data manually
        if split_type == "manual":
            if split == "train":
                self.csv_file = self.csv_file[:train_size]
            elif split == "valid":
                self.csv_file = self.csv_file[train_size:]
        # Split data according to CSV
        elif split_type == "csv":
            if split == "train":
                self.csv_file = self.csv_file[self.csv_file['Train'] == 1]
            elif split == "valid":
                self.csv_file = self.csv_file[self.csv_file['Train'] == 0]

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
            image,
            (image.shape[0], image.shape[1] / self.downsampling_factor, image.shape[2] / self.downsampling_factor)
        )

        # Return labels for classification task
        if self.return_labels:
            plane = self.csv_file.iloc[idx, 1]
            country = self.csv_file.iloc[idx, 3]
            return image, ds_image, plane, country
        else:
            return image, ds_image
