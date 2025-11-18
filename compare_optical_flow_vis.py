import numpy as np
import struct
# import viren2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, RadioButtons
from tqdm import tqdm
import glob
import matplotlib
from PIL import Image
from scipy.ndimage import zoom
matplotlib.use("TkAgg")   # Or "Qt5Agg"

def get_png(paths, frame_num, slice_num):
    path = glob.glob(paths + "/*_frame"+str(frame_num)+"_slice"+str(slice_num)+".png")
    return np.array(Image.open(path[0]))

def resize_4d(data, new_shape=(64,64,64)):
    data = data.astype(np.float32)
    T, X, Y, Z = data.shape
    out = np.zeros((T, *new_shape), dtype=np.float32)

    zoom_factors = (new_shape[0]/X, new_shape[1]/Y, new_shape[2]/Z)
    for t in range(T):
        out[t] = zoom(data[t], zoom_factors, order=1)  # order=1 = linear

    out = out.astype(np.float32)
    out = out - out.min()
    if out.max() > 0: out = out / out.max()

    return out

def vis_GUI(us_scan, optical_viren_folder, vxm_viren_folder, opt_flow, hzn_flow, vert_flow, dep_flow, lowest_res):
    # Default orientation and indices
    orientation = "X, Y"

    us = resize_4d(us_scan[:30])

    hzn_dep_folder1 = optical_viren_folder + "/hzn_dep"
    hzn_vert_folder1 = optical_viren_folder + "/hzn_vert"
    vert_dep_folder1 = optical_viren_folder + "/vert_dep"

    hzn_dep_folder2 = vxm_viren_folder + "/hzn_dep"
    hzn_vert_folder2 = vxm_viren_folder + "/hzn_vert"
    vert_dep_folder2 = vxm_viren_folder + "/vert_dep"

    hzn_opt_flow = resize_4d(opt_flow[:30, 0, ...])
    vert_opt_flow = resize_4d(opt_flow[:30, 1, ...])
    dep_opt_flow = resize_4d(opt_flow[:30, 2, ...])

    slice_index = lowest_res // 2
    total_frames = len(glob.glob(hzn_dep_folder2 + "/*")) // lowest_res
    time_index = total_frames //2

    # --- Figure setup ---
    fig, ((usax, optvirenax, optvectorax), (ax, vxmvirenax, vxmvectorax)) = plt.subplots(2, 3)
    plt.subplots_adjust(left=0.2, bottom=0.2)

    ax.set_axis_off()

    usimg = usax.imshow(us[time_index, :, :, slice_index], cmap= 'gray')
    usax.set_axis_off()
    usax.set_title("US Scan")

    optvirenimg = optvirenax.imshow(get_png(hzn_vert_folder1, time_index, slice_index), cmap="gray")
    optvirenax.set_title(f"Optical Displacement")
    optvirenax.set_axis_off()

    optvectorimg = optvectorax.imshow(us[time_index, :, :, slice_index], cmap= 'gray')
    optY,optX = np.mgrid[0:64, 0:64]
    optvect = optvectorax.quiver(optX, optY, 5*hzn_opt_flow[time_index, :, :, slice_index], 5*vert_opt_flow[time_index, :, :, slice_index])
    optvectorax.set_title(f"Optical Displacement")
    
    vxmvirenimg = vxmvirenax.imshow(get_png(hzn_vert_folder2, time_index, slice_index), cmap="gray")
    vxmvirenax.set_title(f"VoxelMorph Displacement")
    vxmvirenax.set_axis_off()

    vxmvectorimg = vxmvectorax.imshow(us[time_index, :, :, slice_index], cmap= 'gray')
    Y,X = np.mgrid[0:64, 0:64]
    vxmvect = vxmvectorax.quiver(X, Y, 5*hzn_flow[time_index, :, :, slice_index], 5*vert_flow[time_index, :, :, slice_index], color='red', scale=60)
    vxmvectorax.set_title(f"VoxelMorph Displacement") # \nFrames {time_index}-{time_index+1}

    fig.suptitle(f"Unmasked Training Data. Slice: {slice_index}, Frame: {time_index}")

    # --- Slider for slice and time indices ---
    
    slice_ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=slice_ax_slider,
        label="Slice",
        valmin=0,
        valmax=lowest_res-1,
        valinit=slice_index,
        valstep=1,
    )

    time_ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
    time_slider = Slider(
        ax=time_ax_slider,
        label="Time",
        valmin=0,
        valmax=total_frames-1,
        valinit=time_index,
        valstep=1,
    )

    # --- Radio buttons for orientation ---
    ax_radio = plt.axes([0.05, 0.4, 0.15, 0.15])
    radio = RadioButtons(ax_radio, ("X, Y", "X, Z", "Y, Z"))

    # --- Update functions ---
    def update_display():
        time_idx = int(time_slider.val)
        slice_idx = int(slice_slider.val)
        fig.suptitle(f"Masked Training Data. Slice: {slice_idx}, Frame: {time_idx}")

        if orientation == "X, Y":
            us_data = us[time_idx, :, :, slice_idx]
            optviren_data = get_png(hzn_vert_folder1, time_idx, slice_idx)
            vxmviren_data = get_png(hzn_vert_folder2, time_idx, slice_idx)

            optvect_data1 = hzn_opt_flow[time_idx, :, :, slice_idx]
            optvect_data2 = vert_opt_flow[time_idx, :, :, slice_idx]
            vxmvect_data1 = hzn_flow[time_idx, :, :, slice_idx]
            vxmvect_data2 = vert_flow[time_idx, :, :, slice_idx]

        elif orientation == "X, Z":
            us_data = us[time_idx, :, slice_idx, :].T
            optviren_data = get_png(hzn_dep_folder1, time_idx, slice_idx)
            vxmviren_data = get_png(hzn_dep_folder2, time_idx, slice_idx)
            optviren_data = optviren_data.transpose(1, 0, 2)
            vxmviren_data = vxmviren_data.transpose(1, 0, 2)

            optvect_data1 = hzn_opt_flow[time_idx, :, slice_idx, :].T
            optvect_data2 = dep_opt_flow[time_idx, :, slice_idx, :].T
            vxmvect_data1 = hzn_flow[time_idx, :, slice_idx, :].T
            vxmvect_data2 = dep_flow[time_idx, :, slice_idx, :].T

        elif orientation == "Y, Z":
            us_data = us[time_idx, slice_idx, :, :].T
            optviren_data = get_png(vert_dep_folder1, time_idx, slice_idx)
            vxmviren_data = get_png(vert_dep_folder2, time_idx, slice_idx)
            optviren_data = optviren_data.transpose(1, 0, 2)
            vxmviren_data = vxmviren_data.transpose(1, 0, 2)

            optvect_data1 = vert_opt_flow[time_idx, slice_idx, :, :].T
            optvect_data2 = dep_opt_flow[time_idx, slice_idx, :, :].T
            vxmvect_data1 = vert_flow[time_idx, slice_idx, :, :].T
            vxmvect_data2 = dep_flow[time_idx, slice_idx, :, :].T

        usimg.set_data(us_data)
        optvirenimg.set_data(optviren_data)
        vxmvirenimg.set_data(vxmviren_data)
        optvect.set_UVC(5*optvect_data1, 5*optvect_data2)
        vxmvect.set_UVC(5*vxmvect_data1, 5*vxmvect_data2)
        fig.canvas.draw_idle()

    def update_image(val):
        update_display()

    def update_orientation(label):
        nonlocal orientation
        orientation = label

        slice_slider.ax.set_xlim(slice_slider.valmin, slice_slider.valmax)
        slice_slider.set_val(slice_slider.val)  # trigger update
        update_display()

    def on_key(event):
        val = time_slider.val
        if event.key == 'right':
            val = min(time_slider.valmax, val + 1)
        elif event.key == 'left':
            val = max(time_slider.valmin, val - 1)
        time_slider.set_val(val)

    # --- Connect events ---
    time_slider.on_changed(update_image)
    slice_slider.on_changed(update_image)
    radio.on_clicked(update_orientation)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

# Load volume
us = np.load("Optical Flow from Jan Lebert/2022-12-25_US08/2022-12-25_Cam08_US08.npy")
optical_viren_path = "out/Optical_Unmasked/Viren2d_Flows_refFrame_Jan"
vxm_viren_path = "out/Optical_Unmasked/Viren2d_Flows_refFrame_Predicted"
hzn_flow = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/out/Masked/RefFrame/hzn_flow_jump.npy")
vert_flow = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/out/Masked/RefFrame/vert_flow_jump.npy")
dep_flow = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/out/Masked/RefFrame/dep_flow_jump.npy")

optical_flow = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/Optical Flow from Jan Lebert/2022-12-25_US08/2022-12-25_Cam08_US08_flows_ref_4.npy")

vis_GUI(us, optical_viren_path, vxm_viren_path, optical_flow, hzn_flow, vert_flow, dep_flow, 64)