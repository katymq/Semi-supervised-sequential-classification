# test
seed = 123
import random
import torch
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from utils.training import num_param

random.seed(seed)     # python random generator
np.random.seed(seed)  
torch.manual_seed(seed) # pytorch random generator
torch.cuda.manual_seed_all(seed) # for multi-gpu

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def img_save(im, val = 0):
    n, m = im.shape
    img = im.copy()+1
    img[n-1,m-1] = val
    return img
def img_missing(im, val = 0):
    img = im.copy()+1
    img[np.where(im==-1)] = val
    return img
def model_reconstruction(model,epoch_model, path_save, device, print_loss =False):
    # print('Actual  path for to initialize our models: ', path_save)
    path = os.path.join(path_save, model.__class__.__name__.casefold()+'_state_'+str(epoch_model)+'.pth') 

    if device == 'cpu':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    # print(f'Initialization of the {model.__class__.__name__} model  at epoch {epoch_model}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if print_loss:
        print('loss: ',{checkpoint['loss']}, 'and epoch: ', {checkpoint['epoch']})
    return model


def test_model(model, device, name_image, setting, general_path, image_x, image_y, y_true,size,epoch_init, change_folder=False): 
    x = torch.tensor(image_x.reshape(size*size, 1), dtype=torch.float32)
    y = torch.tensor(image_y.reshape(size*size, 1), dtype=torch.float32)
    x = x.to(device)
    y = y.to(device)
    model.to(device)
    model.eval()
    data = model.__class__.__name__.casefold()+'_'+name_image+setting
    path_save = os.path.join(general_path, data)
    path = os.path.join(path_save, model.__class__.__name__.casefold()+'_state_'+str(epoch_init)+'.pth') 
    
    if os.path.exists(path_save) and os.path.exists(path):
        # print(f'Actual path to save our models for {data} is \n {path_save} \n')
        model = model_reconstruction(model,epoch_init, path_save, device)
        #--------------------------------------------
        #* Reconstruction
        y_ = model.reconstruction(x,y)
        if device == 'cuda:0':
            y_ = y_.cpu()
        y_ = y_.detach().numpy().astype('int64')
        y_true_ = y_true.reshape(size*size).astype('int64')
        y1 = image_y.reshape(size*size).astype('int64')
        y_pred_m = y_[np.where(y1 == -1)]
        y_true_m = y_true_[np.where(y1 == -1)]
        # print('y_pred_m',np.unique(y_pred_m, return_counts=True), y_pred_m)
        # print('y_true_m',np.unique(y_true_m, return_counts=True), y_true_m)
        error_rate = 1-accuracy_score(y_true_m, y_pred_m)
        # print(np.sum(1*(y_.reshape(size,size)==y_true)))
        print(f'{name_image}: {model.__class__.__name__ } error {error_rate}' )

        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,30))
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
        plt.close()
        #--------------------------------------------
        plt.imshow(img_save(image_x), cmap='gray')
        plt.axis('off')
        plt.margins(x=0)
        plt.savefig(os.path.join(path_save,f'{name_image}.png'), transparent=True, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.imshow(img_save(image_y), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(path_save,f'{name_image}_miss.png'),transparent=True,format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.imshow(img_save(y_true), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(path_save,f'{name_image}_label.png'),transparent=True,format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.imshow(img_save(y_.reshape(size,size)), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(path_save,f'{name_image}_pred_{model.__class__.__name__}.png'),transparent=True,format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        print('Images saved.')
        # print('Folder moved.')

        if change_folder & os.path.exists(os.path.join(os.path.dirname(general_path), r'Results_outputs\v3_128_junio', data)):
            print("Destination folder already exists in the new path!")
            # Delete the existing folder in the source path
            shutil.rmtree(os.path.join(os.path.dirname(general_path), r'Results_outputs\v3_128_junio', data))
            shutil.move(path_save, os.path.join(os.path.dirname(general_path), r'Results_outputs\v3_128_junio'))
            print("Folder moved successfully!")

    else:
        print("The model doesn't exist in this path")
        error_rate = None
    return error_rate
    
