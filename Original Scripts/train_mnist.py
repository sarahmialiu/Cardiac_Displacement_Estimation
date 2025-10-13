#!/usr/bin/env python

import os
import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import generators
import losses


# ------------ MODEL HYPERPARAMETERS AND IMAGE PATHS ---------------

output_dir = 'VoxelMorph\out'                               # output model directory
gpus = [0]
device = 'cuda:0'
cudnn_nondet = True                                         # disable cudnn determinism - might slow down training
bidirectional = False                                       # enable bidirectional cost function
batch_size = 8
lr = 1e-4                                                   # learning rate (default: 1e-4)
epochs = 50                                                # number of training epochs (default: 1500)
steps_per_epoch = 200                                       # number of training batches per epoch (default: 100)
val_steps_per_epoch = int(0.2*steps_per_epoch)              # number of validation batches per epoch
initial_epoch = 0                                           # initial epoch number (default: 0)
weights = [1, 100000000]                                   # weights to apply to loss (NCC or MSE) and deformation loss (l2 Grad)
debug = True                                                # when debug = True, script only loads two scan and trains for two epochs


# ----------------------- DATA PREPROCESSING -----------------------

# load and prepare training data
interp=3 
rsz=np.zeros(2)
wd=28
ht=28

mnist = fetch_openml('mnist_784', version=1)

mnist_samples = mnist['data'][mnist['target'].astype(int) == 5]
mnist_samples = mnist_samples.astype('float') / 255.0
mnist_samples = np.array(mnist_samples)

fixed = []
moving = []

for i in range(mnist_samples.shape[0]):
    idx1 = np.random.randint(0, mnist_samples.shape[0])
    fix = mnist_samples[idx1, :].reshape((ht,wd))
    fix = cv2.resize(fix, (ht*4, wd*4), interpolation=cv2.INTER_NEAREST)
    fixed.append(fix)
    
    idx2 = np.random.randint(0, mnist_samples.shape[0])
    mov = mnist_samples[idx2, :].reshape((ht,wd))
    mov = cv2.resize(mov, (ht*4, wd*4), interpolation=cv2.INTER_NEAREST)
    moving.append(mov)          

    if i == 2 and debug == True: break

fixed = np.array(fixed) # [..., np.newaxis]
moving = np.array(moving) # [..., np.newaxis]

train_fixed, val_fixed, train_moving, val_moving = train_test_split(moving, fixed, test_size=0.2, random_state=70)

print("Training Dataset Length: %d" % len(train_fixed))
print("Validation Dataset Length: %d" % len(val_fixed))
print()

train_generator = generators.custom_generator(train_moving, train_fixed, batch_size=batch_size)
val_generator = generators.custom_generator(val_moving, val_fixed, batch_size=batch_size)

# Visualize inputs
# while True:
#     input, _ = next(train_generator)
#     plt.imshow(input[0][0,:,:,0])
#     plt.show()
#     images = [input[0][0,:,:,0], input[1][0,:,:,0]] 
#     titles = ['moving', 'fixed']
#     ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)


# ----------------------- MODEL CREATION -----------------------

inshape = next(train_generator)[0][0].shape[1:-1]

os.makedirs(output_dir, exist_ok=True)

nb_gpus = len(gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
assert np.mod(batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (batch_size, nb_gpus)

torch.backends.cudnn.deterministic = not cudnn_nondet

# unet architecture
enc_nf = [16, 16, 32, 64] 
dec_nf = [64, 64, 64, 32, 16, 16, 16] 

model = vxm.networks.VxmDense(
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf],
    bidir=bidirectional,
    int_steps=7, # number of integration steps (default: 7)
    int_downsize=2 # flow downsample factor for integration (default: 2)
)

model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# loss: NCC, deformation loss: L2
losses = [vxm.losses.NCC(win=[8, 8]).loss, losses.Grad('l2', loss_mult=2).loss]  # losses.NCC(win=[100,15]).loss
training_val_losses = []
min_val_loss = 100

# training loops
for epoch in range(initial_epoch, epochs):
    # save model checkpoint
    if Path(output_dir + "\\model_weights.pth").is_file(): 
            model.load_state_dict(torch.load(output_dir + "\\model_weights.pth"))

    # ------------------------- TRAINING -------------------------

    epoch_loss = [] # list of each loss for each step
    epoch_total_loss = [] # list of sum of all losses for each step
    epoch_step_time = []
    model.train()

    for step in range(steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(train_generator) # [moving_images, fixed_images], [fixed_images, zero_phi] 

        inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true]
        # moving_images.shape = fixed_images.shape = [bs, 1, ht, wd] = [bs, 1, 1312, 176]

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs) # tuple

        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses): # calculating losses for both MSE and l2
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            # if n == 1: # 2
            #     curr_loss = loss_function(y_true[1], y_pred[1]) * weights[n]
            # else:
            #     curr_loss = loss_function(y_true[0], y_pred[0]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)
    
    # print training info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'training loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

    # ------------------------- VALIDATION -------------------------

    val_epoch_loss = []
    val_epoch_total_loss = []
    model.eval()

    for step in range(val_steps_per_epoch): # iterate through entire validation set
        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(val_generator)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true]

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs) # tuple

        # calculate total loss
        val_loss = 0
        val_loss_list = []
        for n, loss_function in enumerate(losses):
            if n == 1:  # 2
                val_curr_loss = loss_function(y_true[1], y_pred[1]) * weights[n]
            else:
                val_curr_loss = loss_function(y_true[0], y_pred[0]) * weights[n]
            val_loss_list.append(val_curr_loss.item())
            val_loss += val_curr_loss

        val_epoch_loss.append(loss_list)
        val_epoch_total_loss.append(val_loss.item())

    # print validation info
    val_losses_info = ', '.join(['%.4e' % f for f in np.mean(val_epoch_loss, axis=0)])
    val_info = 'validation loss: %.4e  (%s)' % (np.mean(val_epoch_total_loss), val_losses_info)
    print(val_info, flush=True)

    training_val_losses.append(np.mean(val_epoch_total_loss))
    np.savetxt(Path(output_dir + "\\val_losses.txt"), training_val_losses, fmt='%.5f')

    if np.mean(val_epoch_total_loss) < min_val_loss: 
        min_val_loss = np.mean(val_epoch_total_loss)
        torch.save(model.state_dict(), Path(output_dir + "\\model_weights.pth"))
        print("Validation loss decreased, saving new model weights")
    
    print()

    if epoch == 1 and debug == True: break

