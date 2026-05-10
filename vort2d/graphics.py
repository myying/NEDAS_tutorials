import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from .diagnostics import rmse, sprd, err_spec, sprd_spec
from .utils import get_times

# set global plotting parameters
# color map and range of vorticity to show (1/s)
import cmocean
vort_cmap = getattr(cmocean.cm, 'curl')
vort_min = -5e-3
vort_max = 5e-3
vort_intv = 1e-3

# compute vorticity (1/s) from velocity (m/s)
from NEDAS.utils.spatial_operation import gradx, grady
def uv2zeta(grid, vel):
    u, v = vel[0], vel[1]
    zeta = gradx(v, grid.dx, grid.cyclic_dim) - grady(u, grid.dy, grid.cyclic_dim)
    return zeta

def set_map_axis(ax, grid):
    cticks = np.array([5, 15, 25, 35, 45])
    ax.set_xticks(grid.x[0,cticks])
    ax.set_xticklabels(((grid.x[0,cticks] - grid.Lx/2)/1e3).astype(int))
    ax.set_yticks(grid.y[cticks,0])
    ax.set_yticklabels(((grid.y[cticks,0] - grid.Ly/2)/1e3).astype(int))
    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$y$ (km)')

# true vorticity map, highlighted contour in black, and ensemble members in colors
def plot_vorticity_map(ax, c, n, hours, Xt, Xens):
    ax.clear()
    vort_truth = uv2zeta(c.grid, Xt[n])
    vort_truth[np.where(vort_truth>vort_max)] = vort_max
    vort_truth[np.where(vort_truth<vort_min)] = vort_min
    ax.contourf(c.grid.x, c.grid.y, vort_truth, np.arange(vort_min, vort_max+vort_intv, vort_intv), cmap=vort_cmap)
    clvs = [-1e-3, 1e-3]
    cmap = [getattr(plt.cm, 'jet')(x) for x in np.linspace(0, 1, c.config.nens)]
    for m in range(c.config.nens):
        vort_mem = uv2zeta(c.grid, Xens[n,m])
        ax.contour(c.grid.x, c.grid.y, vort_mem, clvs, colors=[cmap[m][0:3]], linewidths=1)
    ax.contour(c.grid.x, c.grid.y, vort_truth, clvs, colors='k', linewidths=2)
    ax.set_title(fr"vorticity $t$={hours[n]:02}h")
    set_map_axis(ax, c.grid)

# power spectra of error and ensemble spread
from NEDAS.diag.metrics.spectral import pwrspec2d
def plot_spectrum(ax, grid, n, hours, Xt, Xens):
    ax.clear()
    wn, pwr = pwrspec2d(Xt[n])
    ax.loglog(wn, np.mean(pwr, axis=0), color='k', linewidth=2, label='truth')
    wn, err_pwr = err_spec(Xens[n], Xt[n])
    ax.loglog(wn, np.mean(err_pwr, axis=0), color='g', linewidth=3, label='rmse')
    wn, sprd_pwr = sprd_spec(Xens[n])
    ax.loglog(wn, np.mean(sprd_pwr, axis=0), color='r', linestyle=':', linewidth=2, label='sprd')
    ax.legend()
    ax.set_title(fr'spectrum $t$={hours[n]:02}h')
    ax.set_xlim([0.8, 12])
    ax.set_ylim([1e-5, 1e2])
    lengths = np.array([500, 200, 100, 50, 20])
    ax.set_xticks(grid.Lx / 1e3 / lengths)
    ax.set_xticklabels(lengths)
    ax.set_xlabel('wavelength (km)')
    ax.set_ylabel(r'$m^2/s^2$')
    ax.grid()

# Sawtooth time series of error and ensemble spread
def plot_sawtooth_ts(ax, n, hours, rmse_ts, sprd_ts):
    ax.clear()
    ax.plot(hours[0:n+1], rmse_ts[0:n+1], color='g', linewidth=3)
    ax.plot(hours[0:n+1], sprd_ts[0:n+1], color='r', linestyle=':', linewidth=2)
    ax.plot(hours[n], rmse_ts[n], color='k', marker='+', markersize=10)
    ax.set_title('domain-avg rmse,sprd')
    ax.set_xlim([-1, np.max(hours)+1])
    ax.set_xticks([0, 12, 24, 36, 48])
    ax.set_ylim([0, 20])
    ax.set_xlabel(r'forecast $t$ (h)')
    ax.set_ylabel(r'$m/s$')
    ax.grid()

import base64
import ipywidgets as widgets
from PIL import Image
from IPython.display import HTML, display, clear_output
from NEDAS.utils.graphics import add_colorbar, adjust_ax_size

def get_hours(c):
    times = get_times(c)
    dt = c.models['vort2d'].restart_dt
    return [int((t - c.config.time_start) / (timedelta(hours=1)*dt)) for t in times]

def get_time_id_for_plot(c):
    times = get_times(c)
    if c.config.run_analysis:
        return [i for i,_ in enumerate(times)]
    else:
        return [times.index(t) for t in np.unique(times)]

def plot_ens_error_sprd(casename, c, truth_state, ens_state):
    # compute rmse and ensemble spread
    Xt = np.array(truth_state)
    Xens = np.array(ens_state)
    rmse_ts = rmse(Xens, Xt)
    sprd_ts = sprd(Xens)

    # loop over time index and make plot
    hours = get_hours(c)
    t_ids = get_time_id_for_plot(c)
    for i, n in enumerate(t_ids):
        fig, ax = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
        plot_vorticity_map(ax[0], c, n, hours, Xt, Xens)
        plot_spectrum(ax[1], c.grid, n, hours, Xt, Xens)
        plot_sawtooth_ts(ax[2], n, hours, rmse_ts, sprd_ts)
        plt.savefig(f"vort2d/work/plots/{casename}_diag_{i+1:02}.png")
        plt.close()

def make_animation(casename, c):
    t_ids = get_time_id_for_plot(c)
    frames = []
    for i,_ in enumerate(t_ids):
        path = f"vort2d/work/plots/{casename}_diag_{i+1:02d}.png"
        frames.append(Image.open(path))
    
    # Save as GIF
    frames[-1].save(f'vort2d/{casename}_diag_animation.gif',
                   save_all=True,
                   append_images=frames[0:],
                   optimize=False,
                   duration=200, # ms per frame
                   loop=0)

def interactive_animator(casename, c):
    t_ids = get_time_id_for_plot(c)

    # 1. Create the Slider
    slider = widgets.IntSlider(
        value=1, min=1, max=len(t_ids),
        description='Frame:',
        continuous_update=True,
        layout=widgets.Layout(width='500px')
    )

    # 2. Create the Output area
    out = widgets.Output()

    def update_frame(change):
        i = change['new']
        # Construct the path we know exists
        img_path = f"vort2d/work/plots/{casename}_diag_{i:02d}.png"

        with out:
            clear_output(wait=True)
            try:
                # Read image and convert to Base64
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")

                # Display via HTML string (This bypasses the 404 path error)
                display(HTML(f'<img src="data:image/png;base64,{encoded}" style="width:100%; max-width:800px;">'))
            except Exception as e:
                print(f"Error loading frame {i}: {e}")

    # 3. Link and Display
    slider.observe(update_frame, names='value')

    # Show initial frame
    with out:
        update_frame({'new': 1})

    return widgets.VBox([slider, out])
