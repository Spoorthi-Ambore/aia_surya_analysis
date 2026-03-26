# aia_surya_analysis
This repository is created to save the python codes used for the UAH SPARC Project - on the Surya AI Foundation model and AIA.

# 26 march 2026, Thursday
# Python code to display an interactive AIA image plot, and select any region and obtain the output as: 
#!/usr/bin/env python3
# for original l1.5 images.

\`\`\`python 
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from astropy.io import fits
from matplotlib.colors import LogNorm
import sunpy.visualization.colormaps as cm

%matplotlib qt
\`\`\`

# ---------------- CONFIG ----------------
\`\`\`python 
AIA_DIR = "/home/spoorthi/Surya/aaa_right_files/ok/AIA_reduced_256"
SURYA_DIR = "/home/spoorthi/Surya/aaa_right_files/ok/Surya_reduced_256"
CHANNELS = ['94', '131', '171', '193', '211', '335']
BINS = 100
EPS = 1e-9
\`\`\`

# ---------- SAFE LOGNORM ----------
\`\`\`python 
def safe_lognorm(data):
    data = data[np.isfinite(data)]
    data = data[data > 0]

    if len(data) == 0:
        return None

    vmin = np.percentile(data, 1)
    vmax = np.percentile(data, 99)

    if vmin <= 0 or vmax <= 0 or vmin >= vmax:
        return None

    return LogNorm(vmin=vmin, vmax=vmax)
\`\`\`
# ---------- LOAD FITS ----------
\`\`\`python 
def load_fits(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(float)

def find_aia_by_channel(directory, channel):
    for f in sorted(os.listdir(directory)):
        if f"_{channel}A_" in f and f.endswith(".fits"):
            return os.path.join(directory, f)
    return None

def find_surya_by_channel(directory, channel):
    for f in sorted(os.listdir(directory)):
        if f"_{channel}A_" in f and f.endswith(".fits"):
            return os.path.join(directory, f)
    return None
\`\`\`
# ---------- LOAD DATA ----------
\`\`\`python 
aia_data = {}
surya_data = {}

for ch in CHANNELS:
    aia_path = find_aia_by_channel(AIA_DIR, ch)
    surya_path = find_surya_by_channel(SURYA_DIR, ch)

    if aia_path is None or surya_path is None:
        raise RuntimeError(f"Missing file for channel {ch}")

    aia_data[ch] = load_fits(aia_path)
    surya_data[ch] = load_fits(surya_path)
\`\`\`
# ---------- SUNPY COLORMAP ----------
\`\`\`python 
def get_cmap(ch):
    return cm.cmlist.get(f"sdoaia{ch}")
\`\`\`
# ---------- CROP ----------
def crop(img, coords):
    return img[
        coords['ymin']:coords['ymax'],
        coords['xmin']:coords['xmax']
    ]

# ---------- GLOBAL STORE ----------
pixel_map = {}

# ---------- MAIN PLOT ----------
def make_plots(coords):

    global pixel_map
    pixel_map = {}

    fig, axes = plt.subplots(5, 6, figsize=(32, 24), constrained_layout=True)

    for i, ch in enumerate(CHANNELS):

        aia_crop = crop(aia_data[ch], coords)
        surya_crop = crop(surya_data[ch], coords)

        # Flatten
        aia_flat = aia_crop.flatten()
        surya_flat = surya_crop.flatten()

        y_idx, x_idx = np.indices(aia_crop.shape)
        x_flat = x_idx.flatten()
        y_flat = y_idx.flatten()

        mask = np.isfinite(aia_flat) & np.isfinite(surya_flat)
        aia_flat = aia_flat[mask]
        surya_flat = surya_flat[mask]
        x_flat = x_flat[mask]
        y_flat = y_flat[mask]

        pixel_map[ch] = (aia_flat, surya_flat, x_flat, y_flat)

        cmap = get_cmap(ch)

        # ---------- ROW 1: AIA ----------
        ax = axes[0, i]
        norm = safe_lognorm(aia_crop)
        im = ax.imshow(aia_crop, origin='lower',
                       cmap=cmap,
                       norm=norm if norm else None)
        ax.set_title(f"AIA {ch}")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        # ---------- ROW 2: SURYA ----------
        ax = axes[1, i]
        norm = safe_lognorm(surya_crop)
        im = ax.imshow(surya_crop, origin='lower',
                       cmap=cmap,
                       norm=norm if norm else None)
        ax.set_title(f"Surya {ch}")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        # ---------- ROW 3: HIST ----------
        ax = axes[2, i]
        ax.hist(aia_flat, bins=BINS, alpha=0.5, label="AIA")
        ax.hist(surya_flat, bins=BINS, alpha=0.4, label="Surya")
        ax.legend(fontsize=8)
        ax.set_title(f"Overlap {ch}")

        # ---------- ROW 4: DIFFERENCE IMAGE ----------
        diff_img = aia_crop - surya_crop
        ax = axes[3, i]
        vmax = np.max(np.abs(diff_img))
        im = ax.imshow(diff_img, origin='lower',
                       cmap='bwr',
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"Diff (AIA - Surya) {ch}")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        # ---------- ROW 5: SCATTER ----------
        ax = axes[4, i]
        ax.scatter(surya_flat, aia_flat, s=1, alpha=0.3)

        minv = min(aia_flat.min(), surya_flat.min())
        maxv = max(aia_flat.max(), surya_flat.max())

        ax.plot([minv, maxv], [minv, maxv], 'r--')
        ax.set_xlabel("Surya Intensity")
        ax.set_ylabel("AIA Intensity")
        ax.set_title(f"Scatter {ch}")

    # ---------- CLICK INTERACTION ----------
    def on_click(event):
        if event.inaxes is None:
            return

        for i, ch in enumerate(CHANNELS):

            # Only histogram row
            if event.inaxes == axes[2, i]:

                aia_flat, surya_flat, x_flat, y_flat = pixel_map[ch]

                clicked_val = event.xdata
                if clicked_val is None:
                    return

                idx = np.argmin(np.abs(aia_flat - clicked_val))

                x = int(x_flat[idx])
                y = int(y_flat[idx])

                # plot marker
                axes[0, i].plot(x, y, 'go', markersize=6)
                axes[1, i].plot(x, y, 'go', markersize=6)

                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

# ---------- FULL DISK ----------
aia_example_path = find_aia_by_channel(AIA_DIR, CHANNELS[2])
aia_full = load_fits(aia_example_path)

coords = {'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None}

def onselect(eclick, erelease):
    if eclick.xdata is None or erelease.xdata is None:
        return

    coords['xmin'], coords['xmax'] = sorted([int(eclick.xdata), int(erelease.xdata)])
    coords['ymin'], coords['ymax'] = sorted([int(eclick.ydata), int(erelease.ydata)])

    print(f"\nSelected region: {coords}")

    make_plots(coords)

fig, ax = plt.subplots(figsize=(15, 15))

#norm = safe_lognorm(aia_full)

#im = ax.imshow(aia_full, origin='lower', cmap=get_cmap(CHANNELS[1]),norm=norm if norm else None)
im = ax.imshow(aia_full, origin='lower',
               cmap=get_cmap(CHANNELS[1]),
               vmin = 100, vmax = 1000)
plt.colorbar(im, ax=ax)

ax.set_title(f"AIA Full Disk ({CHANNELS[1]})\nSelect region")

rs = RectangleSelector(
    ax, onselect,
    useblit=True,
    button=[1],
    minspanx=5, minspany=5,
    spancoords='pixels',
    interactive=True
)
plt.show()
