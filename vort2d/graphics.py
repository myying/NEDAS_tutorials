import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from NEDAS.utils.graphics import add_colorbar, adjust_ax_size

from .diagnostics import rmse, sprd, pwrspec2d, variance_spec
from .utils import get_times

# set global plotting parameters
# color map and range of vorticity to show (1/s)
import cmocean
vort_cmap = getattr(cmocean.cm, 'curl')
vort_min = -5e-3
vort_max = 5e-3
vort_intv = 1e-3
vort_highlights = [-1e-3, 1e-3]

# compute vorticity (1/s) from velocity (m/s)
from NEDAS.utils.spatial_operation import gradx, grady
def uv2zeta(grid, vel):
    u, v = vel[0], vel[1]
    zeta = gradx(v, grid.dx, grid.cyclic_dim) - grady(u, grid.dy, grid.cyclic_dim)
    zeta[np.where(zeta>vort_max)] = vort_max
    zeta[np.where(zeta<vort_min)] = vort_min
    return zeta

def set_map_axis(ax, grid):
    cticks = np.array([5, 15, 25, 35, 45])
    ax.set_xticks(grid.x[0,cticks])
    ax.set_xticklabels(((grid.x[0,cticks] - grid.Lx/2)/1e3).astype(int))
    ax.set_yticks(grid.y[cticks,0])
    ax.set_yticklabels(((grid.y[cticks,0] - grid.Ly/2)/1e3).astype(int))
    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$y$ (km)')
    ax.set_aspect('equal')

def plot_ens_cov(ax, c, n, hours, Xt, Xens):
    vort_truth = uv2zeta(c.grid, Xt[n])
    vort_ens_mean = uv2zeta(c.grid, np.mean(Xens[n], axis=0))
    ens_cov = (vort_ens_mean - vort_truth)**2
    im = ax.contourf(c.grid.x, c.grid.y, ens_cov, cmap='Reds')
    ax.set_title(f"ensemble covariance $t$={hours[n]:02}h")
    set_map_axis(ax, c.grid)
    plt.colorbar(im, ax=ax, label=r'$(\overline{\zeta_{ens}} - \zeta_{truth})^2$')

def plot_velocity_map(fig, ax, c, hour, state, vmax=20, color=[.7,.7,.7], showref=True):
    L = c.grid.Lx/10
    c.grid.plot_vectors(ax, state, V=vmax, L=L, linecolor=color, num_steps=5, showref=showref)
    ax.set_title(f"Velocity $t$={hour:02}h")
    set_map_axis(ax, c.grid)
    adjust_ax_size(ax,0.8,0.8,0.05)

def plot_vorticity_map(fig, ax, c, hour, state, colorbar=False):
    vort = uv2zeta(c.grid, state)
    ax.contourf(c.grid.x, c.grid.y, vort, np.arange(vort_min, vort_max+vort_intv, vort_intv), cmap=vort_cmap)
    ax.contour(c.grid.x, c.grid.y, vort, vort_highlights, colors='k', linewidths=2)
    ax.set_title(f"Vorticity $t$={hour:02}h")
    set_map_axis(ax, c.grid)
    nlevels = int((vort_max - vort_min) / vort_intv)
    units = f'{vort_intv}'+ r' $\mathregular{s^{-1}}$'
    if colorbar:
        adjust_ax_size(ax,0.8,0.8,0.05)
        add_colorbar(fig, ax, vort_cmap, vort_min/vort_intv, vort_max/vort_intv, nlevels, units=units, fontsize=10)

# true vorticity map, highlighted contour in black, and ensemble members in colors
def plot_vorticity_spaghetti(fig, ax, c, hour, ref_state, ens_states):
    vort_ref = uv2zeta(c.grid, ref_state)

    # highlighted contours for ref and ensemble members
    cmap = [getattr(plt.cm, 'jet')(x) for x in np.linspace(0, 1, c.config.nens)]
    for m in range(c.config.nens):
        vort_mem = uv2zeta(c.grid, ens_states[m,...])
        ax.contour(c.grid.x, c.grid.y, vort_mem, vort_highlights, colors=[cmap[m][0:3]], linewidths=1)
    ax.contour(c.grid.x, c.grid.y, vort_ref, vort_highlights, colors='k', linewidths=2)
    ax.set_title(f"Vorticity, $t$={hour:02}h\nTruth(black),ensemble(colors)")
    set_map_axis(ax, c.grid)
    adjust_ax_size(ax,0.8,0.8,0.05)

# def plot_obs(c, obs_rec_id):
#     obs_seq = c.obs.obs_seq[obs_rec_id]
#     obs_val = obs_seq['obs']
#     obs_x = obs_seq['x']
#     obs_y = obs_seq['y']
#     grid.plot_scatter(ax[0], obs_val, vmax=vmax, x=obs_x, y=obs_y, is_vector=True, linecolor='k', linewidth=1)
#     ax[0].plot(obs_x, obs_y, color='k', marker='o', markersize=3, zorder=10)
#     ax[0].plot(state_x, state_y, color='k', marker='+', markersize=10, zorder=10)

def plot_spectrum(ax, wn, spec, color, style, width, label):
    ax.loglog(wn, spec, color=color, linestyle=style, linewidth=width, label=label)

def adjust_spec_ax(ax, Lmax, hour):
    ax.set_title(f'Spectrum $t$={hour:02}h')
    ax.set_xlim([0.8, 12])
    ax.set_ylim([1e-5, 1e2])
    lengths = np.array([500, 200, 100, 50, 20])
    ax.set_xticks(Lmax / 1e3 / lengths)
    ax.set_xticklabels(lengths)
    ax.set_xlabel('wavelength (km)')
    ax.set_ylabel(r'$m^2/s^2$')
    ax.grid()
    adjust_ax_size(ax,0.9,0.8,0.05)

# Sawtooth time series of error and ensemble spread
def plot_ts(ax, n, hours, ts, color, style, width, label='', current_marker=False):
    ax.plot(hours[0:n+1], ts[0:n+1], color=color, linestyle=style, linewidth=width, label=label)
    if current_marker:
        ax.plot(hours[n], ts[n], color='k', marker='+', markersize=10)

def adjust_ts_ax(ax, hours, vmax=20):
    ax.set_title('Domain-avg RMSE,spread')
    ax.set_xlim([-1, hours[-1]+1])
    ax.set_xticks([h for h in range(0, hours[-1], 12)])
    ax.set_ylim([0, vmax])
    ax.set_xlabel(r'Forecast $t$ (h)')
    ax.set_ylabel(r'$m/s$')
    ax.grid()
    adjust_ax_size(ax,0.9,0.8,0.05)

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

def make_animation(c, casename, plotname):
    from PIL import Image
    t_ids = get_time_id_for_plot(c)
    frames = []
    for i,_ in enumerate(t_ids):
        path = f"vort2d/work/plots/{casename}_{plotname}_{i+1:02d}.png"
        frames.append(Image.open(path))

    # Save as GIF
    frames[-1].save(f'vort2d/{casename}_{plotname}_animation.gif',
                   save_all=True,
                   append_images=frames[0:],
                   optimize=False,
                   duration=200, # ms per frame
                   loop=0)

def animation_ui(c, casename, plotname):
    import base64
    import ipywidgets as widgets
    from IPython.display import HTML, display, clear_output
    t_ids = get_time_id_for_plot(c)

    # 1. Create the Slider
    slider = widgets.IntSlider(
        value=1, min=1, max=len(t_ids),
        description=f'Frame:',
        continuous_update=True,
        layout=widgets.Layout(width='500px')
    )

    # 2. Create the Output area
    out = widgets.Output()

    def update_frame(change):
        i = change['new']
        # Construct the path we know exists
        img_path = f"vort2d/work/plots/{casename}_{plotname}_{i:02d}.png"

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
