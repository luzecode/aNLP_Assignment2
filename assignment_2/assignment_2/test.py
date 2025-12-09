from utils import load_dataset

from model.model_utils import bag_of_words_matrix,labels_matrix

DATA_PATH = 'C:/Users/lucia/OneDrive/Documents/CognitiveSystems/WiSe1/aNLP/assignment_2/assignment_2/data/dataset.csv'

data = load_dataset(DATA_PATH)

print(labels_matrix(data))