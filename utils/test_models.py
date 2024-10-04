seed = 123
import os
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import random
import torch
import configparser

random.seed(seed)     # python random generator
np.random.seed(seed)  # numpy random generator

torch.manual_seed(seed) # pytorch random generator
torch.cuda.manual_seed_all(seed) # for multi-gpu

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#--------------------------------------------
from models.models_v3 import *
from utils.utils_semi import *
from utils.training import *   
#--------------------------------------------
path = os.getcwd()
folder_data = r'Data'
#-------------------------------
if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"  
print('Your actual device',device)
#----------------------------------------------
general_path  = os.path.join(os.getcwd(), 'Results_save_models') 
options = [f for f in os.listdir(general_path) if not f.endswith('.ini') and not f.endswith('.csv') and not f.endswith('.ipynb') and not f.endswith('.py') and not f.endswith('.txt') and not f.endswith('.md')]
print(options)

# Display options
for i, option in enumerate(options):
    print(f"{i}: {option}")

# Prompt user for choice
choice = input("Choose an option (enter index): ")
# Check if choice is valid
if choice.isdigit() and int(choice) in range(len(options)):
    index = int(choice)
    print(f"You chose {options[index]}.")
else:
    print("Invalid choice.")
file_data = [f for f in sorted(os.listdir(general_path)) if f.endswith('.ini') and options[index] in f]
# Load configuration files
data_config = configparser.ConfigParser()
data_config.read(os.path.join(general_path, file_data[0]))
model_config = configparser.ConfigParser()
model_config.read(os.path.join(general_path, file_data[1]))

# Get input values from configuration files or prompt user for input
type_image = data_config.get('data_settings', 'type_image', fallback=None)
size = data_config.getint('data_settings', 'size', fallback=None)
p = data_config.getint('data_settings', 'p', fallback=None)
print(f"Data info: {type_image}, {size}, {p}")
#----------------------------------------------
# Path to save the model
path_data = os.path.join(path, folder_data,type_image+'_'+str(size))
print('Exist this folder', os.path.exists(path_data))
#----------------------------------------------
list_images = []
names_images = []
for data in sorted(os.listdir(path_data)):
    if data.endswith('.npy') and str(p) in data:
        list_images.append(np.load(os.path.join(path_data, data)))
        names_images.append(data[:-4])

index = data_config.getint('data_settings', 'index', fallback=None)
name_image = names_images[index]
print('index', index, name_image)
image_x, y_true, image_y = list_images[index]
print(np.max(image_x), np.min(image_x), np.max(image_y), np.min(image_y))
x = torch.tensor(image_x.reshape(size*size, 1), dtype=torch.float32)
y = torch.tensor(image_y.reshape(size*size, 1), dtype=torch.float32)
x = x.to(device)
y = y.to(device)
#----------------------------------------------
#! General model settings
x_dim = 1
y_dim = 1
weight_decay_ = 1e-4
clip = 10
# add_loss =True # input("Enter 'True' if add_loss should be True, or 'False' otherwise: ").lower() == 'true'

z_dim = model_config.getint('model_settings', 'z_dim', fallback=None)
h_dim = model_config.getint('model_settings', 'h_dim', fallback=None)
num_neurons = model_config.getint('model_settings', 'num_neurons', fallback=None)
learning_rate = model_config.getfloat('model_settings', 'learning_rate', fallback=None)
n_epochs = model_config.getint('model_settings', 'n_epochs', fallback=None)
epoch_init = model_config.getint('model_settings', 'epoch_init', fallback=None)
print_every = model_config.getint('model_settings', 'print_every', fallback=None)
save_every = model_config.getint('model_settings', 'save_every', fallback=None)
add_loss = model_config.get('model_settings', 'add_loss', fallback=None)
sel_model = model_config.get('model_settings', 'sel_model', fallback=None)
setting = model_config.get('model_settings', 'setting', fallback=None)

if sel_model == 'tmm':
    model = TMM(x_dim, z_dim, y_dim, h_dim, num_neurons, device, add_loss)
if sel_model == 'vls':
    model = VSL( x_dim, z_dim, y_dim, h_dim, num_neurons, device)
if sel_model == 'svrnn':
    model = SVRNN(x_dim, z_dim, h_dim, y_dim, num_neurons, device, add_loss)


#--------------------------------------------
# Save models
#--------------------------------------------
model.to(device)
data = model.__class__.__name__.casefold()+'_'+name_image+setting
path_save = os.path.join(general_path, data)
print(f'Actual path to save our models for {data} is \n {path_save} \n')
print( f'epoch_init = {epoch_init}, z_dim = {z_dim}, num_neurons = {num_neurons}, h_dim = {h_dim}, add_loss = {add_loss}, learning_rate = {learning_rate}, n_epochs = {n_epochs}, print_every = {print_every}, save_every = {save_every}, model = {model.__class__.__name__}')


epoch_init = int(input("epoch_init to initialize the model for reconstruction: "))
model = model_reconstruction(model,epoch_init, path_save, device)

#--------------------------------------------
#* Reconstruction
y_ = model.reconstruction(x,y)
if device == 'cuda:0':
    y_ = y_.cpu()

y_ = y_.detach().numpy().astype('int64')

# y_true_ = image_to_chain(y_true).astype('int64')
# y1 = image_to_chain(image_y).astype('int64')
y_true_ = y_true.reshape(size*size).astype('int64')
y1 = image_y.reshape(size*size).astype('int64')
# y_ = y_.reshape(size,size)

print(y_)
print(y_true_)
print(y1)

print('True labels',np.unique(y_true_, return_counts=True))
print('Predicted ',np.unique(y_, return_counts=True))
print('missing',np.unique(image_y, return_counts=True))

print(len(y_true_), len(y_), len(y1))

y_pred_m = y_[np.where(y1 == -1)]
y_true_m = y_true_[np.where(y1 == -1)]
print('y_pred_m',np.unique(y_pred_m, return_counts=True), y_pred_m)
print('y_true_m',np.unique(y_true_m, return_counts=True), y_true_m)
error_rate = 1-accuracy_score(y_true_m, y_pred_m)
print(np.sum(1*(y_.reshape(size,size)==y_true)))
print(f'{name_image}: {model.__class__.__name__ } with  {num_param(model)} parameters \n after {epoch_init} has an error rate of {error_rate}\n folder {data}' )

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 15))
# Plot the first image on the first subplot
ax1.imshow(y_.reshape(size,size), cmap='gray')
ax1.set_title(f'{name_image}: {model.__class__.__name__ } error  {np.round(100*error_rate, 4)}')
# Plot the second image on the second subplot
ax2.imshow(image_y, cmap='gray')
ax2.set_title('Missing Labels')
# Plot the third image on the third subplot
ax3.imshow(1*(y_true), cmap='gray')
ax3.set_title('True Labels')
ax4.imshow(image_x, cmap='gray')
ax4.set_title('Image')
plt.savefig(os.path.join(path_save,f'{name_image}_resumen_{model.__class__.__name__}.png'),transparent=True,format='png', bbox_inches='tight', pad_inches=0)
# Display the subplots in separate windows
plt.show()
#--------------------------------------------
save_images = input('Do you want to save the images? (y/n): ')

# If the user wants to save the images, save them
if save_images.lower() == 'y':
    def img_save(im, val = 0):
        n, m = im.shape
        img = im.copy()+1
        img[n-1,m-1] = val
        return img
    def img_missing(im, val = 0):
        img = im.copy()+1
        img[np.where(im==-1)] = val
        return img
    plt.imshow(img_save(image_x), cmap='gray')
    plt.axis('off')
    plt.margins(x=0)
    plt.savefig(os.path.join(path_save,f'{name_image}.png'), transparent=True, format='png', bbox_inches='tight', pad_inches=0)
    plt.imshow(img_save(image_y), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(path_save,f'{name_image}_miss.png'),transparent=True,format='png', bbox_inches='tight', pad_inches=0)
    plt.imshow(img_save(y_true), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(path_save,f'{name_image}_label.png'),transparent=True,format='png', bbox_inches='tight', pad_inches=0)
    plt.imshow(img_save(y_), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(path_save,f'{name_image}_pred_{model.__class__.__name__}.png'),transparent=True,format='png', bbox_inches='tight', pad_inches=0)
else:
    print('Images not saved.')
