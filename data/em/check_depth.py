import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Slider, Button

def plot_depth_3d_frame(ax, depth, stride=4, cmap='Spectral_r'):
    ax.clear()
    H, W = depth.shape
    x = np.arange(0, W, stride)
    y = np.arange(0, H, stride)
    X, Y = np.meshgrid(x, y)
    Z = depth[::stride, ::stride]
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    ax.set_zlabel('Depth')
    return surf

def interactive_depth_3d(npy_path, stride=4, cmap='Spectral_r'):
    depth = np.load(npy_path)
    if depth.ndim == 2:
        depth = depth[None, ...]
    n_frames = depth.shape[0]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    surf = plot_depth_3d_frame(ax, depth[0], stride, cmap)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Depth')
    ax.set_title(f"3D Depth Visualization (frame 0)")

    # Slider
    axframe = plt.axes([0.2, 0.1, 0.65, 0.03])
    frame_slider = Slider(axframe, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)

    # Button
    axbutton = plt.axes([0.85, 0.025, 0.1, 0.04])
    play_button = Button(axbutton, 'Play All')

    def update(val):
        idx = int(frame_slider.val)
        ax.clear()
        surf = plot_depth_3d_frame(ax, depth[idx], stride, cmap)
        ax.set_title(f"3D Depth Visualization (frame {idx})")
        fig.canvas.draw_idle()

    def play(event):
        for idx in range(n_frames):
            frame_slider.set_val(idx)
            plt.pause(0.1)  # Adjust for playback speed

    frame_slider.on_changed(update)
    play_button.on_clicked(play)

    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive 3D depth visualization with slider and play button")
    parser.add_argument("npy_path", type=str, help="Path to .npy file (depth map)")
    parser.add_argument("--stride", type=int, default=4, help="Stride for downsampling grid")
    parser.add_argument("--cmap", type=str, default="Spectral_r", help="Colormap")
    args = parser.parse_args()
    interactive_depth_3d(args.npy_path, stride=args.stride, cmap=args.cmap)