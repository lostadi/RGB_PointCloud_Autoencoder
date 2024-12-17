#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.getcwd())


# In[2]:


import numpy as np
import time
import utils
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import model
import torch.optim as optim

from geomloss import SamplesLoss



# In[3]:


#Note to self: I changed the fig_count to 2 
#and am doing normalizing instead of padding for constant 
#number of points 


# In[4]:


import torch
from torch.utils.data import Dataset, DataLoader
#from Dataloaders import GetDataLoaders
from RGB_Dataloaders import GetDataLoaders

#Config
batch_size = 32
output_folder = os.path.expanduser("~/meoutputsv2_2.2/") # folder path to save the results
save_results = True # save the results to output_folder
use_GPU = True # use GPU, False to use CPU
latent_size = 128 # bottleneck size of the Autoencoder model
pc_test_array = np.load("data/ModelNet10/alltest.npy", allow_pickle=True)
pc_train_array = np.load("data/ModelNet10/alltrain.npy", allow_pickle=True)
print(pc_test_array.shape)
print(pc_train_array.shape)


# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
#from Dataloaders import GetDataLoaders
from RGB_Dataloaders import GetDataLoaders


pc_test_array = np.load("data/ModelNet10/alltest.npy", allow_pickle=True)
pc_train_array = np.load("data/ModelNet10/alltrain.npy", allow_pickle=True)
print(pc_test_array.shape)
print(pc_train_array.shape)
###################################pad fct#######################################################
#if point clouds have varying sizes, pad them to a fixed size ((e.g., )1024 points)
#def pad_point_clouds(pc_list, fixed_size=1024):
#   padded_pc = []
#   for pc in pc_list:
#       num_points = pc.shape[0]
#       if num_points < fixed_size:
#           # Pad with zeros
#           padding = np.zeros((fixed_size - num_points, 3))
#           padded = np.vstack((pc, padding))
#       elif num_points > fixed_size:
#           # Truncate to fixed_size
#           padded = pc[:fixed_size, :]
#       else:
#           padded = pc
#       padded_pc.append(padded)
#   return np.array(padded_pc)
####################################selection fct#######################################################3

import numpy as np

def adjust_point_clouds(pc_list, fixed_size=1028):
    adjusted_pc = []
    for pc in pc_list:
        num_points = pc.shape[0]
        if num_points < fixed_size:
            # Upsample by randomly duplicating points
            indices = np.random.choice(num_points, fixed_size - num_points, replace=True)
            upsampled_points = pc[indices, :]
            adjusted_pc_single = np.vstack((pc, upsampled_points))
        elif num_points > fixed_size:
            # Downsample by randomly selecting points
            indices = np.random.choice(num_points, fixed_size, replace=False)
            adjusted_pc_single = pc[indices, :]
        else:
            adjusted_pc_single = pc
        adjusted_pc.append(adjusted_pc_single)
    return np.array(adjusted_pc)



################################################################################################
#Converts array to list
pc_test_list = pc_test_array.tolist()
pc_train_list = pc_train_array.tolist()

#NORMING b4 the padding:
#--------------------------------------------------------------------
def normalize_point_cloud(npArray):
    
    #Find mean of each axis
    centroid = np.mean(npArray, axis=0)
    
    npArray = npArray - centroid
    
    d_max = np.max(np.sqrt(np.sum(npArray**2, axis=1)))
    
    #Standardize the values 
    #########Hart12/12/24: Take out the division of std#########
    npArray = npArray / d_max
    
    return npArray
#-------------------------------------------------------------------
#Normed each pc in pc_list
normed_pc_test_array = [normalize_point_cloud(pc) for pc in pc_test_list]
normed_pc_train_array = [normalize_point_cloud(pc) for pc in pc_train_list]


################################padding AFTER norming########################
#Pads point clouds
#fxed_size = 3000  # Adjust as needed
#ped_pc_array = pad_point_clouds(normed_pc_array, fixed_size=fixed_size)
#p.save('data/toilet_padded.npy', padded_pc_array)

#print(f"Padded point cloud array shape: {padded_pc_array.shape}")
########################################################################
#Selects point clouds
fixed_size = 1028  # Adjust as needed


adjusted_pc_test_array = adjust_point_clouds(normed_pc_test_array, fixed_size=fixed_size)
np.save('data/alltest_adjusted.npy', adjusted_pc_test_array)

print(f"Adjusted point cloud test array shape: {adjusted_pc_test_array.shape}")



adjusted_pc_train_array = adjust_point_clouds(normed_pc_train_array, fixed_size=fixed_size)
np.save('data/alltrain_adjusted.npy', adjusted_pc_train_array)

print(f"Adjusted point cloud train array shape: {adjusted_pc_train_array.shape}")




############################################################################

#---------------replacing with seperate train and test paths---------------
#load dataset from numpy array and divide 90%-10% randomly for train and test sets
#train_loader, test_loader = GetDataLoaders(npArray=adjusted_pc_array, batch_size=batch_size, shuffle=True, num_workers=8)
#--------------replaced with-----------------------------------------------------------------

testing_dataset = '/home/lee-ostadi/Point-Cloud-Autoencoder/data/alltest_adjusted.npy'

training_dataset = '/home/lee-ostadi/Point-Cloud-Autoencoder/data/alltrain_adjusted.npy'





#load from the saved .npy files
train_loader, test_loader = GetDataLoaders(training_dataset, testing_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


#test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

#train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

#---------------------------------------------------------------------------------------------

# l models have the same size, get the point size from the first model
point_size = len(train_loader.dataset[0])
#point_size = fixed_size
print(point_size)


# In[6]:


#ONLY NEED TO DO THIS PART ONCE 
##############################################################################################3
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#-----------------------------------------------------------------------

def plot_and_save_pointcloud(pc, save_path, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.axis('off')  

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    if show:
        plt.show()
    else:
        plt.close()

#------------------------------------------------------------------------        
        
#maakes folder to save input images
input_test_images_folder = os.path.join(output_folder, "input_test_pointclouds")
os.makedirs(input_test_images_folder, exist_ok=True)


#using the padded input
num_test_pointclouds = adjusted_pc_test_array.shape[0]

for idx in range(num_test_pointclouds):
    pc = adjusted_pc_test_array[idx]
    save_path = os.path.join(input_test_images_folder, f"input_{idx}.png")
    plot_and_save_pointcloud(pc, save_path, show=False)
    print(f"Saved input point cloud {idx} to {save_path}")
    


#maakes folder to save input images
input_train_images_folder = os.path.join(output_folder, "input_train_pointclouds")
os.makedirs(input_train_images_folder, exist_ok=True)


#using the padded input
num_train_pointclouds = adjusted_pc_train_array.shape[0]

for idx in range(num_train_pointclouds):
    pc = adjusted_pc_train_array[idx]
    save_path = os.path.join(input_train_images_folder, f"input_{idx}.png")
    plot_and_save_pointcloud(pc, save_path, show=False)
    print(f"Saved input point cloud {idx} to {save_path}")



# In[7]:


net = model.PointCloudAE(point_size,latent_size)

#######################################################################
#LOADS WEIGHTS OF NETWORK
#load the weights
if os.path.isfile('Generalv2_1028pc_Encoder.pth'):
    net.load_state_dict(torch.load('Generalv2_1028pc_Encoder.pth'))
    print("Loaded weights from 'Generalv2_1028pc_Encoder.pth'")
else:
    print("No saved model weights found at 'Generalv2_1028pc_Encoder.pth'. Starting training from scratch.")
#####################################################3




if(use_GPU):
    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1: # if there are multiple GPUs use all
        net = torch.nn.DataParallel(net)
else:
    device = torch.device("cpu")

net = net.to(device)


# In[8]:


from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance

optimizer = optim.Adam(net.parameters(), lr=0.0006)



# In[9]:


def train_epoch():
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        
        data = data.to(device)
        output = net(data.permute(0,2,1)) # transpose data for NumberxChannelxSize format
        loss, _ = chamfer_distance(data, output) 
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss/i


# In[ ]:





# In[10]:


def test_batch(data): # test with a batch of inputs
    with torch.no_grad():
        data = data.to(device)
        output = net(data.permute(0,2,1))
        loss, _ = chamfer_distance(data, output)
        
    return loss.item(), output.cpu()


# In[11]:


def test_epoch(): # test with all test set
    with torch.no_grad():
        epoch_loss = 0
        for i, data in enumerate(test_loader):
            loss, output = test_batch(data)
            epoch_loss += loss

    return epoch_loss/i


# In[12]:


#if(save_results):
#    utils.clear_folder(output_folder)


# In[13]:


train_loss_list = []  
test_loss_list = []  
loss_list = []
counter = 0

for i in range(200) :

    startTime = time.time()
    
    train_loss = train_epoch() #train one epoch, get the average loss
    train_loss_list.append(train_loss)
    
    test_loss = test_epoch() # test with test set
    test_loss_list.append(test_loss)
    
    epoch_time = time.time() - startTime
    
    writeString = "epoch " + str(i) + " train loss : " + str(train_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n"
    
    #increments counter
    counter += 1

    #prints and saves loss of every 20th iteration
    loss_list.append(train_loss)
    if counter % 20 == 0:
        print(f'Iteration {counter}, Loss: {train_loss}')
        print(output_folder)
        torch.save(net.state_dict(), 'Generalv2_1028pc_Encoder.pth')
    # plot train/test loss graph
    plt.plot(train_loss_list, label="Train")
    plt.plot(test_loss_list, label="Test")
    plt.legend()

    if(save_results): # save all outputs to the save folder

        # write the text output to file
        with open(output_folder + "prints.txt","a") as file: 
            
            file.write(writeString)

        # update the loss graph
        plt.savefig(output_folder + "loss.png")
        plt.close()

        # save input/output as image file
        if(i%50==0):
            
            test_samples = next(iter(test_loader))
            loss , test_output = test_batch(test_samples)
            utils.plotPCbatch(test_samples, test_output, show=False, save=True, name = (output_folder  + "epoch_" + str(i) + "_test_set"))

            #If you wanted to plot the training outputs aswell
            #train_samples = next(iter(train_loader))
            #loss , train_output = test_batch(train_samples)
            #utils.plotPCbatch(train_samples, train_output, show=False, save=True, name = (output_folder  + "epoch_" + str(i) + "_train_set"))
    else : # display all outputs
        
            test_samples = next(iter(test_loader))
            loss , test_output = test_batch(test_samples)
            utils.plotPCbatch(test_samples, test_output, show=False, save=True, name = (output_folder  + "epoch_" + str(i) + "_test_set"))

            
            train_samples = next(iter(train_loader))
            loss , train_output = test_batch(train_samples)
            utils.plotPCbatch(train_samples, train_output, show=False, save=True, name = (output_folder  + "epoch_" + str(i) + "_train_set"))

            print(writeString)

            plt.show()

    #Added a clear cache for me gpu
    torch.cuda.empty_cache()


# In[14]:


print(output_folder)
torch.save(net.state_dict(), 'Generalv2_1028pc_Encoder.pth')

print(writeString)

plt.show()


# In[ ]:





# In[15]:


test_samples = next(iter(test_loader))
loss , test_output = test_batch(test_samples)
utils.plotPCbatch(test_samples,test_output)

print(writeString)

plt.show()


#torch.save(net.state_dict(), '/home/lee-ostadi/me1stnet.pth')


# In[16]:


train_samples = next(iter(train_loader))
loss , train_output = test_batch(train_samples)
utils.plotPCbatch(train_samples, train_output)
print(writeString)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




