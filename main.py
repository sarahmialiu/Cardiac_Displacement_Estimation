#!/usr/bin/env python

## USE TO VISUALIZE PREDICTED IMGS AGAINST REAL IMGS (SliceViewer)

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Paths
import neurite as ne
from sklearn.model_selection import train_test_split
import glob
from scipy.io import loadmat
from tqdm import tqdm
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import torch

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import generators


# ------------ MODEL HYPERPARAMETERS AND IMAGE PATHS ---------------

folders_path = 'Jad_RFData'
bidirectional = False # enable bidirectional cost function
batch_size = 6
model_dir = 'VoxelMorph\out' # output model directory
gpus = [0]
device = 'cuda:0'
cudnn_nondet = True # disable cudnn determinism - might slow down training
load_model = None # optional model file to initialize with
lr = 1e-4 # learning rate (default: 1e-4)
epochs = 500 # number of training epochs (default: 1500)
steps_per_epoch = 100 # number of batches per epoch (default: 100)
val_steps_per_epoch = 20
initial_epoch = 0 # initial epoch number (default: 0)
weights = [1, 5000, 100] # weighting each loss
min_val_loss = 100
debug = True
weights_path = 'out\JadSim_Outputs\model_weights_NCC.pth'


# ----------------------- DATA PREPROCESSING -----------------------

# load and prepare training data
wd=3*128
ht=128*6 

folders = glob.glob(folders_path + '\*')
fixed = []
moving = []

print("Loading B mode Data")
with tqdm(total=90) as pbar:
    for i, folder_path in enumerate(folders):
        sims = glob.glob(folder_path + '\*')

        for j, sim_path in enumerate(sims):
            bmode_path = Path(glob.glob(sim_path+'\ppb.mat')[0])
            bmode = loadmat(bmode_path)['BmodeRFScanConv']
            num_slices = bmode.shape[2]
            
            for slice_num in range(bmode.shape[2]):
                slice = bmode[:,:,slice_num]
                slice=cv2.resize(slice, # resize frame by maintaining consistent aspect ratio for each image
                        (176,1312),
                        interpolation=cv2.INTER_NEAREST)
                #slice = slice[455:1175, 32:144]
                slice = np.nan_to_num(slice, nan=0) # changing nans to 0 
                
                nonzeroindexs = np.nonzero(slice)
                avg_slice = slice / np.max(np.absolute(slice))

                if slice_num < num_slices-1: # != num_slices-1:
                    fixed.append(avg_slice)
                if slice_num > 0: # != 0:
                    moving.append(avg_slice)
            
            pbar.update()

            if j == 2 and debug == True: break
        if i == 0 and debug == True: break
pbar.close()

fixed = np.array(fixed) # [..., np.newaxis]
moving = np.array(moving) # [..., np.newaxis]

print("Testing Dataset Length: %d" % len(fixed))

test_generator = generators.ordered_generator(fixed, moving, batch_size=1)


# ----------------------- MODEL LOADING AND PREDICTION -----------------------

inshape = next(test_generator)[0][0].shape[1:-1]

# enc_nf = [16, 16, 32, 64]
# dec_nf = [64, 64, 64, 32, 16, 16, 16]

enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=False,
        int_steps=7, # number of integration steps (default: 7)
        int_downsize=2 # flow downsample factor for integration (default: 2)
    )

model.load_state_dict(torch.load(weights_path))
model.to(device)
model.eval()
total_flow = np.zeros((ht//2, wd//2, 2)) # wd


class SliceViewer(tk.Frame):
    def __init__(self, parent, array, title=""):
        super().__init__(parent)
        self.array = array
        self.axis = 'z'
        self.slice_index = 0
        self.title = title

        self.label = tk.Label(self, text=self.title)
        self.label.pack()

        # Create a Canvas to display the slice
        self.canvas = tk.Canvas(self, width=500, height=500)
        self.canvas.pack()

        self.slider = ttk.Scale(self, from_=0, to=array.shape[0]-1, orient='horizontal', command=self.update_image)
        self.slider.pack()

        self.update_image()
    
    def update_image(self, event=None):
        self.slice_index = int(self.slider.get())
        slice_2d = self.array[self.slice_index, :, :]

        # Normalize the slice for display
        slice_2d = ((slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255).astype(np.uint8)

        # Convert the slice to a PhotoImage
        image = Image.fromarray(slice_2d)
        image = image.resize((500, 500))
        self.photo = ImageTk.PhotoImage(image)

        # Display the image on the canvas
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)


if __name__ == "__main__":
    input, _ = next(test_generator)
    inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in input]
    outputs = model(*inputs)
    pred_img = outputs[0].detach().cpu().numpy()
    pred_imgs = pred_img[np.newaxis,0,0,:,:]
    true_imgs = input[1][np.newaxis,0,:,:,0]

    for i in range(27):
        input, _ = next(test_generator)
        first_img = input[0][0,:,:,0]
        true_img = input[1][0,:,:,0]
        true_imgs = np.stack((first_img, true_img), axis=0)

        inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in input]
        outputs = model(*inputs)
        pred_img = outputs[0].detach().cpu().numpy()
        pred_imgs = np.stack((first_img, pred_img[0,0,:,:]), axis=0)

        # true_imgs = np.concatenate((true_imgs, true_img[np.newaxis,...]), axis=0)
        # print(true_imgs.shape)
        # pred_imgs = np.concatenate((pred_imgs, pred_img[np.newaxis,0,0,:,:]), axis=0)
        # print(pred_imgs.shape)
    
        root = tk.Tk()
        frame = tk.Frame(root)
        frame.pack()

        truth = SliceViewer(frame, true_imgs, title="Ground Truth")
        truth.pack(side='left')

        pred = SliceViewer(frame, pred_imgs, title="Predicted Images")
        pred.pack(side='right')

        root.mainloop()
