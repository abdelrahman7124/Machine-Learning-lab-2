from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets


trainset = datasets.MNIST(
    root="./data", train=True, download=True, transform=None
)

print(f"Number of training samples: {len(trainset)}")

# Normalizing the data from [0, 255] to [0, 1]

train_data = trainset.data.float() / 255.0

# Splitting the dataset into training, validation, and test sets


x_train, x_val, y_train, y_val = train_test_split(
    train_data, trainset.targets, test_size=0.4, random_state=42 , stratify=trainset.targets
)
x_test , x_val, y_test , y_val  = train_test_split(
    x_val,y_val, test_size=0.5, random_state=42 ,stratify=y_val
)
print (f"Shape of x_train: {x_train.shape}, y_train: {y_train.shape}")
print (f"Shape of x_val: {x_val.shape}, y_val: {y_val.shape}")
print (f"Shape of x_test: {x_test.shape}, y_test: {y_test.shape}")



#  Flattening the images for logistic regression

x_train_flat = x_train.reshape(-1, 28*28)
x_val_flat   = x_val.reshape(-1, 28*28)
x_test_flat  = x_test.reshape(-1, 28*28)

y_train_flat = y_train
y_val_flat   = y_val
y_test_flat  = y_test


# Filter for only classes 0 and 1
mask_train = (y_train_flat == 0) | (y_train_flat == 1)
mask_val = (y_val_flat == 0) | (y_val_flat == 1)
mask_test = (y_test_flat == 0) | (y_test_flat == 1)

x_train_filtered = x_train_flat[mask_train]
y_train_filtered = y_train_flat[mask_train]

x_val_filtered = x_val_flat[mask_val]
y_val_filtered = y_val_flat[mask_val]

x_test_filtered = x_test_flat[mask_test]
y_test_filtered = y_test_flat[mask_test]

print("After filtering for classes 0 and 1:")
print (f"Shape of x_train_filtered: {x_train_filtered.shape}, y_train_filtered: {y_train_filtered.shape}")
print (f"Shape of x_val_filtered: {x_val_filtered.shape}, y_val_filtered: {y_val_filtered.shape}")
print (f"Shape of x_test_filtered: {x_test_filtered.shape}, y_test_filtered: {y_test_filtered.shape}")

print("---------")



print (f"Shape of x_train: {x_train_flat.shape}, y_train: {y_train_flat.shape}")
print (f"Shape of x_val: {x_val_flat.shape}, y_val: {y_val_flat.shape}")
print (f"Shape of x_test: {x_test_flat.shape}, y_test: {y_test_flat.shape}")

#  Creating data loaders for batching

train_dataset_linear = TensorDataset(x_train_flat, y_train_flat)
val_dataset_linear   = TensorDataset(x_val_flat, y_val_flat)
test_dataset_linear  = TensorDataset(x_test_flat, y_test_flat)
batch_size = 64

train_loader_linear = DataLoader(train_dataset_linear, batch_size=batch_size, shuffle=True)
val_loader_linear   = DataLoader(val_dataset_linear, batch_size=batch_size, shuffle=False)
test_loader_linear  = DataLoader(test_dataset_linear, batch_size=batch_size, shuffle=False)


train_dataset_linear_filtered = TensorDataset(x_train_filtered, y_train_filtered)
val_dataset_linear_filtered   = TensorDataset(x_val_filtered, y_val_filtered)
test_dataset_linear_filtered  = TensorDataset(x_test_filtered, y_test_filtered)


train_loader_linear_filtered = DataLoader(train_dataset_linear_filtered, batch_size=batch_size, shuffle=True)
val_loader_linear_filtered   = DataLoader(val_dataset_linear_filtered, batch_size=batch_size, shuffle=False)
test_loader_linear_filtered  = DataLoader(test_dataset_linear_filtered, batch_size=batch_size, shuffle=False)


train_dataset_nn = TensorDataset(x_train, y_train)
val_dataset_nn   = TensorDataset(x_val, y_val)
test_dataset_nn  = TensorDataset(x_test, y_test)

train_loader_nn = DataLoader(train_dataset_nn, batch_size=batch_size, shuffle=True)
val_loader_nn   = DataLoader(val_dataset_nn, batch_size=batch_size, shuffle=False)
test_loader_nn  = DataLoader(test_dataset_nn, batch_size=batch_size, shuffle=False)
