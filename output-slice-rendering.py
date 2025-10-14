import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

file_path = '/home/sarahl/Documents/Fall Rotation/DataVisualization/data/ultrasound_4D_npy/2024-06-26_US30.npy'

volume = np.load(file_path, allow_pickle=True) # shape (T, Z, Y, X) <- incorrect, likely shape (T, X, Y, Z)
#volume = np.rot90(volume, k=1, axes=(1,3))

# Initial orientation and slice
orientation = "X, Y"   
slice_index = volume.shape[3] // 2
time_index = volume.shape[0] // 2

# --- Figure setup ---
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)  

# Initial image (short axis slice)
img = ax.imshow(volume[time_index, :, :, slice_index], cmap="gray")
ax.set_title(f"({orientation}), Slice: {slice_index}, Time: {time_index}")
#ax.invert_yaxis()

# --- Slider for slice and time indices ---
slice_ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slice_slider = Slider(
    ax=slice_ax_slider,
    label="Slice",
    valmin=0,
    valmax=volume.shape[3] - 1,
    valinit=slice_index,
    valstep=1,
)

time_ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
time_slider = Slider(
    ax=time_ax_slider,
    label="Time",
    valmin=0,
    valmax=volume.shape[0] - 1,
    valinit=time_index,
    valstep=1,
)

# --- Radio buttons for orientation ---
ax_radio = plt.axes([0.05, 0.4, 0.15, 0.15])
radio = RadioButtons(ax_radio, ("X, Y", "X, Z", "Y, Z"))

# --- Update functions ---
def update_image(val):
    update_display()

def update_orientation(label):
    global orientation
    orientation = label
    # reset slider range based on new axis length
    if orientation == "X, Y":
        slice_slider.valmax = volume.shape[3] - 1
    elif orientation == "X, Z":
        slice_slider.valmax = volume.shape[2] - 1
    elif orientation == "Y, Z":
        slice_slider.valmax = volume.shape[1] - 1
    slice_slider.ax.set_xlim(slice_slider.valmin, slice_slider.valmax)
    slice_slider.set_val(slice_slider.val)  # trigger update
    update_display()

def update_display():
    slice_index = int(slice_slider.val)
    time_index = int(time_slider.val)
    if orientation == "X, Y":
        data = volume[time_index, :, :, slice_index]
    elif orientation == "X, Z":
        data = volume[time_index, :, slice_index, :].T
    elif orientation == "Y, Z":
        data = volume[time_index, slice_index, :, :].T
    img.set_data(data)
    ax.set_title(f"({orientation}), Slice: {slice_index}, Time: {time_index}")
    fig.canvas.draw_idle()

# --- Connect events ---
slice_slider.on_changed(update_image)
time_slider.on_changed(update_image)
radio.on_clicked(update_orientation)

plt.show()


