import os

import pandas as pd
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset


class FetalPlaneDataset(Dataset):
    """Fetal Plane dataset."""

    def __init__(self, root_dir, ref,
                 plane,
                 brain_plane=None,
                 us_machine=None,
                 operator_number=None,
                 transform=None,
                 train=None
                 ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            plane: 'Fetal brain'; 'Fetal thorax'; 'Maternal cervix'; 'Fetal femur'; 'Fetal thorax'; 'Other'
            brain_plane: 'Trans-ventricular'; 'Trans-thalamic'; 'Trans-cerebellum'
            us_machine: 'Voluson E6';'Voluson S10'
            operator_number: 'Op. 1'; 'Op. 2'; 'Op. 3';'Other'
            train: Flag denotes if test or train data is used
            
        return image
        """
        self.root_dir = root_dir
        self.ref = pd.read_csv(ref, sep=';')
        self.ref = self.ref[self.ref['Plane'] == plane]
        if plane == 'Fetal brain':
            self.ref = self.ref[self.ref['Brain_plane'] == brain_plane]
        if us_machine is not None:
            self.ref = self.ref[self.ref['US_Machine'] == us_machine]
        if operator_number is not None:
            self.ref = self.ref[self.ref['Operator'] == operator_number]

        size = 256  # Limit dataset size to 256 images (for training)
        if train:
            self.ref = self.ref[:size]
        else:
            self.ref = self.ref[size:]

        self.transform = transform

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, idx):
        # Load the image from file

        # print(f'idx: {idx} \n')
        # print(f'self.ref.iloc[idx, 0]: {self.ref.iloc[idx, 0]} \n')

        img_name = os.path.join(self.root_dir,
                                self.ref.iloc[idx, 0] + '.png')
        # print(img_name)
        image = io.imread(img_name)  # <class 'numpy.ndarray'>

        # print(type(image))
        # print(image.dtype)

        # Preprocess and augment the image
        if self.transform:
            image = self.transform(image)

        ## Make a half sized image for SRGAN
        downsampling = 2
        ds_image = resize(
            image.cpu().numpy(),
            (image.shape[0], image.shape[1] / downsampling, image.shape[2] / downsampling)
        )
        # .cpu().numpy()#TypeError: Cannot interpret 'torch.float32' as a data type

        return ds_image, image
