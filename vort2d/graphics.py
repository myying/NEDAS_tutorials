import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# velocity plotting parameters
vel_scale = 30
len_scale = 5e4
# correlation
corr_cmap = getattr(cmocean.cm, 'balance')

from NEDAS.utils.spatial_operation import gradx, grady
def uv2zeta(grid, vel):
    # compute vorticity (1/s) from velocity (m/s)
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

def plot_velocity_map(ax, c, hour, state, color=[.7,.7,.7], showref=True):
    c.grid.plot_vectors(ax, state, V=vel_scale, L=len_scale, linecolor=color, num_steps=5, showref=showref, ref_units='m/s')
    ax.set_title(f"Velocity $t$={hour:02}h")
    set_map_axis(ax, c.grid)
    adjust_ax_size(ax,0.8,0.8,0.05)

def add_obs_marker(ax, obs_x, obs_y):
    # show marker at obs location
    ax.plot(obs_x, obs_y, color='r', marker='o', markersize=6, markerfacecolor='w', markeredgewidth=1.5)

def add_state_marker(ax, grid, i, j):
    x, y = grid.x[j,i], grid.y[j,i]
    ax.plot(x, y, color='k', marker='s', markersize=4, markerfacecolor='w', markeredgewidth=1.5)

def plot_velocity_obs(ax, grid, obs_val, obs_x, obs_y):
    grid.plot_scatter(ax, obs_val, is_vector=True, linecolor='r', x=obs_x, y=obs_y, vmax=vel_scale, L=len_scale, units='m/s')

def plot_var_on_map(fig, ax, c, title, var, var_min, var_max, var_intv, var_cmap, var_units):
    c.grid.plot_field(ax, var, vmin=var_min, vmax=var_max, cmap=var_cmap)
    ax.set_title(title)
    set_map_axis(ax, c.grid)
    adjust_ax_size(ax,0.8,0.8,0.05)
    add_colorbar(fig, ax, var_cmap, var_min, var_max, int((var_max - var_min) / var_intv), units=var_units, fontsize=8)

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
        add_colorbar(fig, ax, vort_cmap, vort_min/vort_intv, vort_max/vort_intv, nlevels, units=units, fontsize=8)

def plot_vorticity_spaghetti(fig, ax, c, hour, ref_state, ens_states):
    vort_ref = uv2zeta(c.grid, ref_state)

    # highlighted contours for ref and ensemble members
    cmap = [getattr(plt.cm, 'jet')(x) for x in np.linspace(0, 1, c.config.nens)]
    for m in range(c.config.nens):
        vort_mem = uv2zeta(c.grid, ens_states[m,...])
        ax.contour(c.grid.x, c.grid.y, vort_mem, vort_highlights, colors=[cmap[m][0:3]], linewidths=1)
    ax.contour(c.grid.x, c.grid.y, vort_ref, vort_highlights, colors='k', linewidths=2)
    ax.set_title(f"Vorticity, $t$={hour:02}h\nTruth(black); ensemble(colors)")
    set_map_axis(ax, c.grid)
    adjust_ax_size(ax,0.8,0.8,0.05)

def plot_histogram(ax, data, bincolor, bin_wd=4, alpha=1.0, orientation='vertical', label=''):
    nbins = int((max(data) - min(data)) / bin_wd)
    ax.hist(data, bins=nbins, color=bincolor, orientation=orientation, density=True, alpha=alpha, label=label) 

def plot_bivariate_scatter(state_ens_0, state_ens_1, obs_ens_0, obs_ens_1, obs_val, obs_err, nens, truth_state, i, j):
    fig = plt.figure(figsize=(10,5))
    vnames = ['u', 'v']
    gs = gridspec.GridSpec(3,5)
    ax_sc = []
    for v in range(2):
        ax_sc.append(fig.add_subplot(gs[1:3,v*2:(v+1)*2]))
    ax_histy = fig.add_subplot(gs[1:3,4], sharey=ax_sc[1])
    ax_histx = []
    ax_histx.append(fig.add_subplot(gs[0,0:2], sharex=ax_sc[0]))
    ax_histx.append(fig.add_subplot(gs[0,2:4], sharex=ax_sc[1], sharey=ax_histx[0]))
    for v in range(2):
        plt.setp(ax_histx[v].get_xticklabels(), visible=False)
    plt.setp(ax_sc[1].get_yticklabels(), visible=False)
    plt.setp(ax_histx[1].get_yticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    
    prior_bin_color = 'c'
    post_bin_color = [.7, .3, .3]
    obs_bin_color = 'y'

    # Scatter plot
    for v in range(2):
        ax_sc[v].scatter(state_ens_0[:,v,j,i], obs_ens_0, color=prior_bin_color, s=15)
        ax_sc[v].plot([-50, 50], [obs_val, obs_val], 'y', linewidth=3, alpha=0.5)
        ax_sc[v].scatter(state_ens_1[:,v,j,i], obs_ens_1, color=post_bin_color, alpha=0.7, s=15)
        ax_sc[v].plot(np.array([state_ens_0[:,v,j,i], state_ens_1[:,v,j,i]]), np.array([obs_ens_0, obs_ens_1]), 'k', linewidth=0.5, alpha=0.3)
        ax_sc[v].plot([truth_state[0][v,j,i], truth_state[0][v,j,i]], [-50, 50], 'k-', linewidth=0.5)
        ax_sc[v].set_xlabel(f"State ${vnames[v]}$ (m/s)")
        ax_sc[v].set_xlim([-50, 50])
        ax_sc[v].set_ylim([-50, 50])
    ax_sc[0].set_ylabel("Obs $u$ (m/s)")
    
    # Histogram state u
    for v in range(2):
        plot_histogram(ax_histx[v], state_ens_0[:,v,j,i], prior_bin_color)
        plot_histogram(ax_histx[v], state_ens_1[:,v,j,i], post_bin_color, alpha=0.7)
    
    # Histogram obs u
    plot_histogram(ax_histy, obs_ens_0, prior_bin_color, orientation='horizontal', label='Prior')
    # simulate obs likelihood with random sampling
    obs_sample = obs_val + obs_err * np.random.normal(0, 1, nens)
    plot_histogram(ax_histy, obs_sample, obs_bin_color, orientation='horizontal', alpha=0.5, label='Obs likelihood')
    plot_histogram(ax_histy, obs_ens_1, post_bin_color, orientation='horizontal', alpha=0.7, label='Posterior')
    ax_histy.legend(bbox_to_anchor=(1, 1))
    
    plt.tight_layout()

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
