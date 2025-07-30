import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
import numpy as np
import os

# --- Global variables to store data and references ---
full_trajectory_data = None
full_timestamps = None
full_positions = None
min_time_abs = 0.0
max_time_abs = 1.0
selected_start_time = 0.0
selected_end_time = 1.0
loaded_filename = ""

fig = None
ax_3d = None
line_selected = None
line_full_faint = None
point_start_selected = None
point_end_selected = None
slider_start_time = None
slider_end_time = None
text_start_time = None
text_end_time = None
x_lims_full_data, y_lims_full_data, z_lims_full_data = None, None, None # Renamed for clarity


def load_trajectory_file():
    global full_trajectory_data, full_timestamps, full_positions, min_time_abs, max_time_abs
    global selected_start_time, selected_end_time, loaded_filename
    global slider_start_time, slider_end_time, text_start_time, text_end_time
    global x_lims_full_data, y_lims_full_data, z_lims_full_data, button_save, ax_3d # Added ax_3d

    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select gt.txt trajectory file",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )
    root.destroy()

    if not filepath:
        print("File selection cancelled.")
        if fig: fig.suptitle("File selection cancelled. Load a trajectory.", color='red')
        plt.draw()
        return

    loaded_filename = filepath
    print(f"Loading data from: {filepath}")
    if fig: fig.suptitle(f"Loading: {os.path.basename(filepath)}...", color='blue')
    plt.draw()

    try:
        data_list = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) == 8:
                    try:
                        data_list.append([float(p) for p in parts])
                    except ValueError:
                        print(f"Skipping malformed line: {line}")
                else:
                    print(f"Skipping line with incorrect number of columns: {line}")
        
        if not data_list:
            raise ValueError("No valid data points found in the file.")

        full_trajectory_data = np.array(data_list)
        full_timestamps = full_trajectory_data[:, 0]
        full_positions = full_trajectory_data[:, 1:4]

        min_time_abs = np.min(full_timestamps)
        max_time_abs = np.max(full_timestamps)

        if max_time_abs <= min_time_abs:
            max_time_abs = min_time_abs + 1.0

        selected_start_time = min_time_abs
        selected_end_time = max_time_abs

        slider_start_time.ax.set_xlim(min_time_abs, max_time_abs)
        slider_start_time.valmin = min_time_abs
        slider_start_time.valmax = max_time_abs
        slider_start_time.set_val(min_time_abs)

        slider_end_time.ax.set_xlim(min_time_abs, max_time_abs)
        slider_end_time.valmin = min_time_abs
        slider_end_time.valmax = max_time_abs
        slider_end_time.set_val(max_time_abs)
        
        if full_positions.shape[0] > 0:
            min_pos = np.min(full_positions, axis=0)
            max_pos = np.max(full_positions, axis=0)
            center_pos = (min_pos + max_pos) / 2
            range_pos = np.abs(max_pos - min_pos)
            max_range_dim = np.max(range_pos) if np.max(range_pos) > 0 else 1.0
            
            buffer = max_range_dim * 0.1
            # These define the maximum extent of the *data* for initial view
            x_lims_full_data = (center_pos[0] - max_range_dim/2 - buffer, center_pos[0] + max_range_dim/2 + buffer)
            y_lims_full_data = (center_pos[1] - max_range_dim/2 - buffer, center_pos[1] + max_range_dim/2 + buffer)
            z_lims_full_data = (center_pos[2] - max_range_dim/2 - buffer, center_pos[2] + max_range_dim/2 + buffer)
            
            # Set initial view to encompass all data
            ax_3d.set_xlim(x_lims_full_data)
            ax_3d.set_ylim(y_lims_full_data)
            ax_3d.set_zlim(z_lims_full_data)
            ax_3d.view_init(elev=20., azim=-60) # A common starting view
            
        else:
            x_lims_full_data, y_lims_full_data, z_lims_full_data = (-1,1), (-1,1), (-1,1)


        button_save.set_active(True)
        update_plot(reset_view=True) # Pass a flag to reset view on initial load
        if fig: fig.suptitle(f"Loaded: {os.path.basename(filepath)} ({len(full_timestamps)} points)", color='green')
        print(f"Data loaded successfully: {len(full_timestamps)} points.")

    except Exception as e:
        print(f"Error loading file: {e}")
        full_trajectory_data = None
        if button_save: button_save.set_active(False)
        if fig: fig.suptitle(f"Error loading file: {e}", color='red')
    plt.draw()


def update_plot(val=None, reset_view=False): # Added reset_view flag
    global selected_start_time, selected_end_time
    global line_selected, line_full_faint, point_start_selected, point_end_selected
    global text_start_time, text_end_time, ax_3d, x_lims_full_data, y_lims_full_data, z_lims_full_data

    if full_trajectory_data is None:
        if ax_3d:
            ax_3d.clear()
            ax_3d.set_xlabel('X Position')
            ax_3d.set_ylabel('Y Position')
            ax_3d.set_zlabel('Z Position')
            ax_3d.set_title("Load a trajectory file")
            ax_3d.plot([], [], [], '.-', color='gray', label='Full Trajectory (load data)')
            ax_3d.legend()
        plt.draw()
        return

    # --- Store current view before clearing (if not resetting) ---
    current_xlim = None
    current_ylim = None
    current_zlim = None
    current_elev = None
    current_azim = None
    if ax_3d and not reset_view: # Only store if not explicitly resetting
        current_xlim = ax_3d.get_xlim()
        current_ylim = ax_3d.get_ylim()
        current_zlim = ax_3d.get_zlim()
        current_elev = ax_3d.elev
        current_azim = ax_3d.azim
    # --- End store current view ---

    current_start_slider_val = slider_start_time.val
    current_end_slider_val = slider_end_time.val

    if current_start_slider_val >= current_end_slider_val:
        # Check which slider caused the overlap
        # If val is one of the slider values, assume that one was just moved.
        # Otherwise, we might be in an initial setup or programmatic change.
        just_moved_start = (val is not None and np.isclose(val, current_start_slider_val))
        just_moved_end = (val is not None and np.isclose(val, current_end_slider_val))

        if just_moved_start: # Start slider moved and caused overlap
             new_end_val = min(current_start_slider_val + 0.001 * (max_time_abs - min_time_abs if max_time_abs > min_time_abs else 1.0), max_time_abs)
             slider_end_time.set_val(new_end_val)
        elif just_moved_end: # End slider moved and caused overlap
             new_start_val = max(current_end_slider_val - 0.001 * (max_time_abs - min_time_abs if max_time_abs > min_time_abs else 1.0), min_time_abs)
             slider_start_time.set_val(new_start_val)
        else: # If val is None or not matching, default to adjusting end if start is the one "pushing"
            if current_start_slider_val >= slider_end_time.val: # Check against actual slider value
                 slider_end_time.set_val(min(current_start_slider_val + 0.001 * (max_time_abs - min_time_abs if max_time_abs > min_time_abs else 1.0), max_time_abs))
            #This case should ideally not be hit if the above logic is correct for direct slider moves
    
    selected_start_time = slider_start_time.val
    selected_end_time = slider_end_time.val
    
    if text_start_time: text_start_time.set_text(f'Start: {selected_start_time:.3f}s')
    if text_end_time: text_end_time.set_text(f'End: {selected_end_time:.3f}s')

    epsilon = 1e-9 
    indices_selected = np.where(
        (full_timestamps >= selected_start_time - epsilon) & 
        (full_timestamps <= selected_end_time + epsilon)
    )[0]

    ax_3d.clear()

    ax_3d.plot(full_positions[:, 0], full_positions[:, 1], full_positions[:, 2],
               '.-', color='lightgray', alpha=0.7, markersize=2, label='Full Trajectory')

    if len(indices_selected) > 0:
        selected_pos = full_positions[indices_selected, :]
        ax_3d.plot(selected_pos[:, 0], selected_pos[:, 1], selected_pos[:, 2],
                   '.-b', markersize=5, linewidth=1.5, label='Selected Interval')
        ax_3d.plot([selected_pos[0, 0]], [selected_pos[0, 1]], [selected_pos[0, 2]],
                   'go', markersize=8, label='Interval Start')
        ax_3d.plot([selected_pos[-1, 0]], [selected_pos[-1, 1]], [selected_pos[-1, 2]],
                   'ro', markersize=8, label='Interval End')
        ax_3d.set_title(f"Interval: [{selected_start_time:.3f}s - {selected_end_time:.3f}s]")
    else:
        ax_3d.set_title(f"No points in interval [{selected_start_time:.3f}s - {selected_end_time:.3f}s]")

    ax_3d.set_xlabel('X Position')
    ax_3d.set_ylabel('Y Position')
    ax_3d.set_zlabel('Z Position')
    
    # --- Reapply view or set to full data extent ---
    if reset_view or not current_xlim: # If resetting or no previous view stored
        if x_lims_full_data:
            ax_3d.set_xlim(x_lims_full_data)
            ax_3d.set_ylim(y_lims_full_data)
            ax_3d.set_zlim(z_lims_full_data)
        if current_elev is None and current_azim is None: # Set a default view if not even initial was set by load
             ax_3d.view_init(elev=20., azim=-60)

    else: # Reapply stored view
        ax_3d.set_xlim(current_xlim)
        ax_3d.set_ylim(current_ylim)
        ax_3d.set_zlim(current_zlim)
        if current_elev is not None and current_azim is not None:
            ax_3d.view_init(elev=current_elev, azim=current_azim)
    # --- End reapply view ---
    
    ax_3d.legend(loc='upper left', fontsize='small')
    ax_3d.grid(True)
    plt.draw()

def save_selected_interval(event):
    global loaded_filename, selected_start_time, selected_end_time, full_trajectory_data

    if full_trajectory_data is None:
        print("No data loaded to save.")
        if fig: fig.suptitle("No data loaded to save.", color='red')
        plt.draw()
        return

    root_save = tk.Tk()
    root_save.withdraw()

    if loaded_filename:
        base, ext = os.path.splitext(os.path.basename(loaded_filename))
        default_savename = f"{base}_interval_{selected_start_time:.2f}-{selected_end_time:.2f}{ext}"
    else:
        default_savename = f"trajectory_interval_{selected_start_time:.2f}-{selected_end_time:.2f}.txt"

    save_filepath = filedialog.asksaveasfilename(
        title="Save selected trajectory interval",
        initialfile=default_savename,
        defaultextension=".txt",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )
    root_save.destroy()

    if not save_filepath:
        print("Save cancelled.")
        if fig: fig.suptitle("Save cancelled.", color='orange')
        plt.draw()
        return

    print(f"Saving selected interval to: {save_filepath}")
    if fig: fig.suptitle(f"Saving to {os.path.basename(save_filepath)}...", color='blue')
    plt.draw()

    try:
        epsilon = 1e-9
        indices_to_save = np.where(
            (full_timestamps >= selected_start_time - epsilon) &
            (full_timestamps <= selected_end_time + epsilon)
        )[0]

        if len(indices_to_save) == 0:
            print("No data points in the selected interval to save.")
            if fig: fig.suptitle("No data in selected interval to save.", color='orange')
            plt.draw()
            return

        data_to_save = full_trajectory_data[indices_to_save, :]

        with open(save_filepath, 'w') as f:
            f.write(f"# Trajectory segment from original file: {os.path.basename(loaded_filename)}\n")
            f.write(f"# Selected time interval (absolute): {selected_start_time:.6f} to {selected_end_time:.6f}\n")
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            np.savetxt(f, data_to_save, fmt='%.6f', delimiter=' ')
        
        print(f"Successfully saved {len(data_to_save)} points to {save_filepath}")
        if fig: fig.suptitle(f"Saved to {os.path.basename(save_filepath)}", color='green')

    except Exception as e:
        print(f"Error saving file: {e}")
        if fig: fig.suptitle(f"Error saving: {e}", color='red')
    plt.draw()


def main_gui():
    global fig, ax_3d, slider_start_time, slider_end_time, text_start_time, text_end_time, button_save

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.30, right=0.95, top=0.9)

    ax_3d = fig.add_subplot(111, projection='3d')
    ax_3d.set_xlabel('X Position')
    ax_3d.set_ylabel('Y Position')
    ax_3d.set_zlabel('Z Position')
    ax_3d.set_title("Load a trajectory file (gt.txt)")
    ax_3d.grid(True)

    ax_slider_start = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_start_time = Slider(
        ax=ax_slider_start,
        label='Start Time (s)',
        valmin=0.0,
        valmax=1.0,
        valinit=0.0,
        valfmt='%.3f'
    )
    slider_start_time.on_changed(update_plot)

    ax_slider_end = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_end_time = Slider(
        ax=ax_slider_end,
        label='End Time (s)',
        valmin=0.0,
        valmax=1.0,
        valinit=1.0,
        valfmt='%.3f'
    )
    slider_end_time.on_changed(update_plot)
    
    ax_button_load = plt.axes([0.2, 0.025, 0.25, 0.04])
    button_load = Button(ax_button_load, 'Load Trajectory (gt.txt)', color='lightcyan', hovercolor='cyan')
    button_load.on_clicked(lambda event: load_trajectory_file())

    ax_button_save = plt.axes([0.55, 0.025, 0.25, 0.04])
    button_save = Button(ax_button_save, 'Save Selected Interval', color='lightgreen', hovercolor='green')
    button_save.on_clicked(save_selected_interval)
    button_save.set_active(False)

    update_plot(reset_view=True) # Initial call with reset_view
    plt.show()

if __name__ == '__main__':
    main_gui()