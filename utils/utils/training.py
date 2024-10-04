seed = 123
import random
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

random.seed(seed)     # python random generator
np.random.seed(seed)  
torch.manual_seed(seed) # pytorch random generator
torch.cuda.manual_seed_all(seed) # for multi-gpu

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

alpha = 0.1
beta = 0.1



def run_save_model(model, device, name_image, setting, general_path, learning_rate, weight_decay_, clip, x, y, epoch_init, n_epochs, print_every, save_every):
    model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay_)
    print(f'{model.__class__.__name__ } has {num_param(model)} parameters to train \n')
    data = model.__class__.__name__.casefold()+'_'+name_image+setting
    path_save = os.path.join(general_path, data)
    os.makedirs(path_save, exist_ok=True)
    print(f'Actual path to save our models for {data} is \n {path_save} \n')
    loss = run_model_seq(x, y,model, device,optimizer,clip, path_save, n_epochs,save_every, print_every, epoch_init)
    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss of ' + data+ ' with ' + str(n_epochs) + ' epochs')
    plt.savefig(os.path.join(path_save, data + '_loss_' + str(n_epochs) +'.png'))
    plt.close()


def num_param(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_loss_epoch(model, path_save,data, epoch_model):
    path = os.path.join(path_save, model.__class__.__name__.casefold()+'_state_train_'+str(epoch_model)+'.npy') 
    train_LOSS = np.load(path, allow_pickle=True)
    import matplotlib.pyplot as plt
    plt.plot(train_LOSS)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss of ' + data+ ' with ' + str(epoch_model) + ' epochs')
    plt.savefig(os.path.join(path_save, data + '_loss_' + str(epoch_model) +'.png'))
    plt.show()
    plt.close()
    return train_LOSS

def run_model_seq(x, y,model,device, optimizer,clip, path_save_model, n_epochs,save_every=5, print_every=1, epoch_init=1):
    train_LOSS = []
    #model.device(device)
    path_save = os.path.join(path_save_model, model.__class__.__name__.casefold() +'_state_')
    print('The model is saved in this path', path_save)
    new_init = len(os.listdir(path_save_model))>0
    if epoch_init > 1:
        #print(model.__class__.__name__.casefold())
        path = path_save+str(epoch_init)+'.pth'#os.path.join(path_save_model, model.__class__.__name__.casefold()+'_state_'+str(epoch_init)+'.pth') 
        if device == 'cpu':
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(path)
        print(f'Initialization of the {model.__class__.__name__} model  at epoch {epoch_init}')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if new_init and epoch_init == 1:
        files = sorted([f for f in os.listdir(path_save_model) if f.endswith('.pth')])
        path = os.path.join(path_save_model,files[-1])
        if device == 'cpu':
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_init = checkpoint['epoch']
        print(f'Initialization of the {model.__class__.__name__} model  at epoch {epoch_init}')

    if epoch_init < n_epochs:
        for epoch in range(epoch_init, n_epochs + 1):
            model.train()
            optimizer.zero_grad()
            if  model.__class__.__name__.casefold() == 'vsl':
                kld_loss_u, rec_loss_u, y_loss_l = model(x,y)
                loss_l = y_loss_l
                loss_u = kld_loss_u + rec_loss_u        
                loss = loss_l + beta*loss_u
            else:
                # 'svrnn_2' "tmm" all versions
                kld_loss_l, rec_loss_l, y_loss_l, kld_loss_u, rec_loss_u, y_loss_u, add_term= model(x,y)
                loss_l = kld_loss_l + rec_loss_l + y_loss_l
                loss_u = kld_loss_u + rec_loss_u + y_loss_u        
                loss = loss_l + loss_u + alpha*add_term

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            train_LOSS.append(loss.item())

            if epoch % print_every == 0:
                print('Loss Labeled: {:.6f} \t Loss Unlabeled: {:.6f} \t Total Loss: {:.6f}'.format(
                        loss_l, loss_u, loss))     
                # print('\n kdl_loss_l: {:.4f} \t rec_loss_l: {:.4f} \t y_loss_l: {:.4f}'.format(kld_loss_l, rec_loss_l, y_loss_l))
                # print('\n kdl_loss_u: {:.4f} \t rec_loss_u: {:.4f} \t y_loss_u: {:.4f}'.format(kld_loss_u.item(), rec_loss_u.item(), y_loss_u.item()))
                
            if epoch % save_every == 0:
                fn = path_save+str(epoch)+'.pth'
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, fn)
                #torch.save(model.state_dict(), fn)
                # print('Saved model to '+fn)
                np.save(path_save+'train_'+str(epoch)+'.npy', train_LOSS)     
    if epoch_init == n_epochs:
        print('The model is already trained')
    return train_LOSS

def final_model(model, optimizer, epoch_model, path_save,device, print_loss =True):
    print('Actual  path for to initialize our models: ', path_save)
    model.eval()
    path = os.path.join(path_save, model.__class__.__name__.casefold()+'_state_'+str(epoch_model)+'.pth') 
    print(path)
    if device == 'cpu':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    print(f'Initialization of the {model.__class__.__name__} model  at epoch {epoch_model}')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if print_loss:
        print(f'loss: {loss} and epoch: {epoch}')
    return model