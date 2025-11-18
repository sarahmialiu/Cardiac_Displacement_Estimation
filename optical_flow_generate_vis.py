import numpy as np
import struct
import viren2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, RadioButtons
from tqdm import tqdm
import glob
import os
import matplotlib
# matplotlib.use("TkAgg")   # Or "Qt5Agg"

def generate_viren_pngs(flow1, flow2, flow3, output_folder):
    """
    Convert 4D flow frame (t, x, y, z) to viren2d-compatible .flo files: (x,y), (x,z), (y,z)
    
    Args:
        flow_3d: numpy array of shape (height, width, 3) containing (u, v, w) flow
        output_filename: path to output .flo file
    """

    print("Generating Viren2D images from displacement .npy files...")
    
    flow = np.concatenate((flow1[..., np.newaxis], flow2[..., np.newaxis], flow3[..., np.newaxis]), axis=4)
    flow = flow.astype(np.float32)

    height, width, dep = flow.shape[1:4]
    num_frames = flow.shape[0]

    with tqdm(total=num_frames) as pbar:
        for frame_num in range(num_frames):
            for slice_num in range(dep):
                virenflow_hzn_vert = flow[frame_num, :, :, slice_num, :2].astype('float32')
                norm = np.linalg.norm(virenflow_hzn_vert, axis=2).max()
                if norm < 1e-6: norm = 1.0
                colorized_hzn_vert = viren2d.colorize_optical_flow(
                    virenflow_hzn_vert,
                    colormap=viren2d.ColorMap('orientation6'),
                    motion_normalizer=norm
                )
                hzn_vert_folder = output_folder + "/hzn_vert"
                if not os.path.exists(hzn_vert_folder): os.makedirs(hzn_vert_folder)
                viren2d.save_image_uint8(hzn_vert_folder + '/hzn_vert_frame'+str(frame_num)+'_slice'+str(slice_num)+'.png', colorized_hzn_vert)

                virenflow_hzn_dep = flow[frame_num, :, slice_num, :, :2].astype('float32')
                norm = np.linalg.norm(virenflow_hzn_dep, axis=2).max()
                if norm < 1e-6: norm = 1.0
                colorized_hzn_dep = viren2d.colorize_optical_flow(
                    virenflow_hzn_dep,
                    colormap=viren2d.ColorMap('orientation6'),
                    motion_normalizer=norm
                )
                hzn_dep_folder = output_folder + "/hzn_dep"
                if not os.path.exists(hzn_dep_folder): os.makedirs(hzn_dep_folder)
                viren2d.save_image_uint8(hzn_dep_folder + '/hzn_dep_frame'+str(frame_num)+'_slice'+str(slice_num)+'.png', colorized_hzn_dep)

                virenflow_vert_dep = flow[frame_num, slice_num, :, :, :2].astype('float32')
                norm = np.linalg.norm(virenflow_vert_dep, axis=2).max()
                if norm < 1e-6: norm = 1.0
                colorized_vert_dep = viren2d.colorize_optical_flow(
                    virenflow_vert_dep,
                    colormap=viren2d.ColorMap('orientation6'),
                    motion_normalizer=norm
                )
                vert_dep_folder = output_folder + "/vert_dep"
                if not os.path.exists(vert_dep_folder): os.makedirs(vert_dep_folder)
                viren2d.save_image_uint8(vert_dep_folder + '/vert_dep_frame'+str(frame_num)+'_slice'+str(slice_num)+'.png', colorized_vert_dep)

            pbar.update()

    pbar.close()        

def get_png(paths, frame_num, slice_num):
    path = glob.glob(paths + "/*_frame"+str(frame_num)+"_slice"+str(slice_num)+".png")
    return mpimg.imread(path[0])

def vis_GUI(viren_folder):
    # Default orientation and indices
    orientation = "X, Y"
    hzn_dep_folder = viren_folder + "/hzn_dep"
    hzn_vert_folder = viren_folder + "/hzn_vert"
    vert_dep_folder = viren_folder + "/vert_dep"
    slice_index = 32
    total_frames = len(glob.glob(hzn_dep_folder + "/*")) // 64
    time_index = total_frames //2

    # --- Figure setup ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.2, bottom=0.2)

    img = ax.imshow(get_png(hzn_vert_folder, time_index, slice_index), cmap="gray")
    ax.set_title(f"Displacement")
    ax.set_axis_off()

    fig.suptitle(f"Masked Training Data. Slice: {slice_index}, Frame: {time_index}")

    # --- Slider for slice and time indices ---
    
    slice_ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=slice_ax_slider,
        label="Slice",
        valmin=0,
        valmax=64,
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
            img_data = get_png(hzn_vert_folder, time_idx, slice_idx)
        elif orientation == "X, Z":
            img_data = get_png(hzn_dep_folder, time_idx, slice_idx)
        elif orientation == "Y, Z":
            img_data = get_png(vert_dep_folder, time_idx, slice_idx)

        img.set_data(img_data)
        fig.canvas.draw_idle()

    def update_image(val):
        update_display()

    def update_orientation(label):
        nonlocal orientation
        orientation = label

        # # Update slider range based on orientation
        # if orientation == "X, Y":
        #     slice_slider.valmax = get_png(hzn_vert_folder, time_slider.val, slice_slider.val)
        # elif orientation == "X, Z":
        #     slice_slider.valmax = get_png(hzn_dep_folder, time_slider.val, slice_slider.val)
        # elif orientation == "Y, Z":
        #     slice_slider.valmax = get_png(vert_dep_folder, time_slider.val, slice_slider.val)

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
viren2d_savepath = "out/Unmasked_newMSE/Viren2d_Flows_refFrame"
hzn_flow = np.load("out/Unmasked_newMSE/hzn_flow_refFrame.npy")
vert_flow = np.load("out/Unmasked_newMSE/vert_flow_refFrame.npy")
dep_flow = np.load("out/Unmasked_newMSE/dep_flow_refFrame.npy")

# Save as .flo (only x and y will be saved)
generate_viren_pngs(hzn_flow, vert_flow, dep_flow, viren2d_savepath)

# vis_GUI(viren2d_savepath)
