import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class AttributesDataset():
    def __init__(self, annotation_path):
        age_labels = []
        gender_labels = []
        ethnicity_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                age_labels.append(row['age'])
                gender_labels.append(row['gender'])
                ethnicity_labels.append(row['ethnicity'])

        self.age_labels = np.unique(age_labels)
        self.gender_labels = np.unique(gender_labels)
        self.ethnicity_labels = np.unique(ethnicity_labels)

        self.num_age = len(self.age_labels)
        self.num_gender = len(self.gender_labels)
        self.num_ethnicity = len(self.ethnicity_labels)

        self.age_id_to_name = dict(zip(range(len(self.age_labels)), self.age_labels))
        self.age_name_to_id = dict(zip(self.age_labels, range(len(self.age_labels))))

        self.gender_id_to_name = dict(zip(range(len(self.gender_labels)), self.gender_labels))
        self.gender_name_to_id = dict(zip(self.gender_labels, range(len(self.gender_labels))))

        self.ethnicity_id_to_name = dict(zip(range(len(self.ethnicity_labels)), self.ethnicity_labels))
        self.ethnicity_name_to_id = dict(zip(self.ethnicity_labels, range(len(self.ethnicity_labels))))


class FaceDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # initialize the arrays to store the ground truth labels and paths to the image
        self.data = []
        self.age_labels = []
        self.gender_labels = []
        self.ethnicity_labels = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.age_labels.append(self.attr.age_name_to_id[row['age']])
                self.gender_labels.append(self.attr.gender_name_to_id[row['gender']])
                self.ethnicity_labels.append(self.attr.ethnicity_name_to_id[row['ethnicity']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]

        # read image
        #img = Image.open(img_path)
        # print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # img = np.expand_dims(img, axis=0).astype('uint8')
        # img = img.transage((2,0,1))
        # print(img_path, img.shape)
        # img = img.reshape(120,128)
        img = Image.fromarray(img)
        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)

        # return the image and all the associated labels
        dict_data = {
            'img': img,
            'labels': {
                'age_labels': self.age_labels[idx],
                'gender_labels': self.gender_labels[idx],
                'ethnicity_labels': self.ethnicity_labels[idx]
            }
        }
        return dict_data
