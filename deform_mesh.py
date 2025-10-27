import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from sklearn.metrics import mean_squared_error

def deform_mask(mask):#, hzn_flow, vert_flow):

    # assert hzn_flow.shape == vert_flow.shape, \
    #     f"Trying to visualize images with different shapes. \
    #         Horizontal Flow: {hzn_flow.shape}, Vertical Flow: {vert_flow.shape}"

    # Default orientation and indices
    orientation = "X, Y"
    slice_index = mask.shape[3] // 2
    time_index = mask.shape[0] // 2

    # --- Figure setup ---
    fig = plt.figure()
    ax = fig.add_subplot()

    img = ax.imshow(mask[time_index, :, :, slice_index], cmap="gray")
    ax.set_title(f"Frame: {time_index})")
    ax.set_axis_off()


    # --- Slider for slice and time indices ---
    
    slice_ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=slice_ax_slider,
        label="Slice",
        valmin=0,
        valmax=mask.shape[3] - 1,
        valinit=slice_index,
        valstep=1,
    )

    time_ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
    time_slider = Slider(
        ax=time_ax_slider,
        label="Time",
        valmin=0,
        valmax=mask.shape[0] - 1,
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
        ax.set_title(f"Frame: {time_idx}")   

        if orientation == "X, Y":
            data = mask[time_idx, :, :, slice_idx]
        elif orientation == "X, Z":
            data = mask[time_idx, :, slice_idx, :].T
        elif orientation == "Y, Z":
            data = mask[time_idx, slice_idx, :, :].T

        img.set_data(data)
        fig.canvas.draw_idle()

    def update_image(val):
        update_display()

    def update_orientation(label):
        nonlocal orientation
        orientation = label

        # Update slider range based on orientation
        if orientation == "X, Y":
            slice_slider.valmax = mask.shape[3] - 1
        elif orientation == "X, Z":
            slice_slider.valmax = mask.shape[2] - 1
        elif orientation == "Y, Z":
            slice_slider.valmax = mask.shape[1] - 1

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
mask = np.load("/home/sarahl/Documents/Fall Rotation/DataVisualization/data/ultrasound_4D_npy/2024-06-26_US30_biv.npy", allow_pickle=True)
deform_mask(mask)