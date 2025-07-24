import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

# Load data
test_coords = np.load('test_coords.npy')
train_coords = np.load('train_coords.npy')
yrefs = np.load('yrefs.npy')  # shape (192, 288)
influences_all = np.load('index_15533_influences.npy', allow_pickle=True)  # shape (1024, 2765)

# --- Interactive plotting function ---
def plot_influence_over_time(yrefs, test_coords, train_coords, influences_all):
    num_timestamps = influences_all.shape[0]
    n_train = train_coords.shape[0]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'height_ratios': [1]})
    plt.subplots_adjust(bottom=0.13)
    cmap = plt.get_cmap("RdBu_r")

    im0 = ax0.imshow(yrefs, cmap=cmap)
    im1 = ax1.imshow(yrefs, cmap=cmap)
    fig.colorbar(im0, ax=ax0)
    ax0.set_title("Reference: Test index 15533 (yellow)")
    ax1.set_title("Influence: Top N% highlighted")
    ax0.set_xticks([]); ax0.set_yticks([])
    ax1.set_xticks([]); ax1.set_yticks([])
    ax0.set_xlim([0, yrefs.shape[1]])
    ax0.set_ylim([yrefs.shape[0], 0])
    ax1.set_xlim([0, yrefs.shape[1]])
    ax1.set_ylim([yrefs.shape[0], 0])

    marker = [None]
    influence_scatter = [None]

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_timestamp = plt.axes([0.15, 0.08, 0.7, 0.03], facecolor=axcolor)
    ax_percent = plt.axes([0.15, 0.03, 0.7, 0.03], facecolor=axcolor)
    slider_timestamp = widgets.Slider(ax_timestamp, 'Timestamp', 0, num_timestamps-1, valinit=0, valstep=1)
    slider_percent = widgets.Slider(ax_percent, 'Top % Influential Points', 10, 100, valinit=10, valstep=10)

    def update_plots(timestamp=None, percent=None):
        timestamp = int(slider_timestamp.val) if timestamp is None else int(timestamp)
        percent = slider_percent.val if percent is None else percent

        # Update left plot (reference field)
        im0.set_data(yrefs)
        # Find the test point location for index 15533
        loc = test_coords[15533]
        if marker[0] is not None:
            marker[0].remove()
        marker[0] = ax0.scatter(loc[1] * yrefs.shape[1], loc[0] * yrefs.shape[0],
                                s=100, edgecolor='yellow', facecolor='none', linewidth=2)

        # Update right plot (influence)
        im1.set_data(yrefs)
        if influence_scatter[0] is not None:
            influence_scatter[0].remove()
            influence_scatter[0] = None
        influence_np = influences_all[timestamp]
        if influence_np is None:
            print(f"No influence data for timestamp {timestamp}")
            fig.canvas.draw_idle()
            return
        influence_sorted_indices = np.argsort(influence_np)
        n_top = max(1, int((percent / 100.0) * len(influence_np)))
        top_idx = influence_sorted_indices[-n_top:]
        xs = train_coords[top_idx, 1] * yrefs.shape[1]
        ys = train_coords[top_idx, 0] * yrefs.shape[0]
        top_influences = influence_np[top_idx]
        norm = plt.Normalize(vmin=np.min(top_influences), vmax=np.max(top_influences))
        cmap_influence = plt.get_cmap('PRGn')
        influence_scatter[0] = ax1.scatter(xs, ys, s=20, c=top_influences, cmap='PRGn', norm=norm, edgecolor='none')
        # Add or update colorbar for influence
        if not hasattr(update_plots, 'cbar') or update_plots.cbar is None:
            update_plots.cbar = fig.colorbar(influence_scatter[0], ax=ax1, orientation='vertical')
            update_plots.cbar.set_label(f'Influence (Top {percent:.0f}%)')
        else:
            update_plots.cbar.mappable.set_clim(np.min(top_influences), np.max(top_influences))
            update_plots.cbar.set_label(f'Influence (Top {percent:.0f}%)')
            update_plots.cbar.update_normal(influence_scatter[0])
        fig.canvas.draw_idle()

    def slider_update(val):
        update_plots()

    slider_timestamp.on_changed(slider_update)
    slider_percent.on_changed(slider_update)

    # Initialize
    update_plots()
    plt.show()

# --- Usage example ---
if __name__ == '__main__':
    plot_influence_over_time(yrefs, test_coords, train_coords, influences_all) 