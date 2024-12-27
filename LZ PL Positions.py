# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 03:10:15 2024

@author: sugnu
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

# Constants and functions
μ       = 379312077e8            # m^3/s^2 
R       = 60268e3                # metre
g_10    = 21141e-9
Ω       = 1.74e-4                # rad/s
j_2     = 1.629071e-2    # J2 perturbation coefficient

# charge by mass ratio
ρ       = 1e3                    # 1gm/cm^3 Headmen 2021= 10^3 kg/m^3
#b = 1e-6                  # = 1e-3 micro
V       = 10                     # Volt
ε       = 8.85e-12               # Farad/metre
μ_0     = 4*np.pi*1e-7

# Initial parameters
r0      = 67628e3               # Initial radial distance

# Initial conditions for the plasma ions
r_i     = 657e5   #1.12 * R
x_i     = 0.0
θ_i     = np.pi /2
y_i     = 0.0
z_i     = (μ / r_i**3) ** 0.5

# Initial conditions of the charged particle
x0      = 0.0
θ0      = np.pi / 2
y0      = 0.0
z_0     = (μ / r0**3)**0.5
n_i     = 1e-3                 # Plasma ion density

# Define relative velocity to use in the ODEs
w_relative = np.sqrt((x_i - x0)**2 + (r_i*y_i - r0*y0)**2 + (r_i*z_i*np.sin(θ_i) - r0*z_0*np.sin(θ0))**2)

z_kep      = (μ/(r0**3))**0.5
Co_y_kep   = z_kep/r0

b_values = np.array([1e-9,5e-9, 1e-8, 1e-7,1e-6,5e-6,10e-6])
#b_values   = np.array([1e-7,1e-6])

def event_func_r(t, p):
    return p[0] - 60968e3 # R+700km kollman et al. 2018
event_func_r.terminal  = True
event_func_r.direction = -1

def event_func_outer(t, p):
    return p[0] - 7156e4
event_func_outer.terminal  = True
event_func_outer.direction = +1

trajectories = {}

# Iterate over different values of b
for i, b_value in enumerate(b_values):
    # Update parameters based on the current value of b
    m = (4/3) * pi * (b_value**3) * ρ
    q = 4 * pi * ε * b_value * V
    β = q / m
    
    a_plasma = np.pi * b_value ** 2 * n_i * w_relative
    
    def odes(t, p):
        # Extracting variables from the array
        r, x, θ, y, ϕ, z = p

        B_θ = μ_0*(R/r)**3*g_10*np.sin(θ)
        B_r = μ_0*(R/r)**3*g_10*np.cos(θ)

        # defining the ODEs
        drdt = x
        dxdt = r * ((y**2 + ((z + Ω)**2) * (np.sin(θ)**2)) - β*z*np.sin(θ)*B_θ) - a_plasma * x
    
        dθdt = y
        dydt = (-2*x*y + r*((z + Ω)**2) * np.sin(θ)*np.cos(θ) + (β*r*z*np.sin(θ)*B_r))/r - a_plasma * y
                    
        dϕdt = z
        dzdt = (-2*(z + Ω)*(x*np.sin(θ) + r*y*np.cos(θ)) + β*(x*B_θ - r*y*B_r))/(r*np.sin(θ)+1e-15) - a_plasma * z
    
        return np.array([drdt,dxdt,dθdt,dydt,dϕdt,dzdt])

        # time window
    t_span = (0, 86400)
    #t = np.linspace(t_span[0],t_span[1],3600)

    dt_desired = 1 # Desired time step in seconds
    num_points = int((t_span[1] - t_span[0]) / dt_desired) + 1
    t = np.linspace(t_span[0], t_span[1], num_points)
    
    # initial conditions
    p0 = np.array([r0,0, 90.0*(pi/180), 0, 0.0*(pi/180),1.74e-4])
    # 0 -1000

    # Solve IVP 
    sol = solve_ivp(odes, t_span, p0, t_eval=t, events=[event_func_r, event_func_outer], 
                    method="BDF", rtol=1e-9, atol=1e-11)
    
    # Extract final values and end time
    final_values = sol.y[:, -1]
    end_time = sol.t[-1]
    
   # Print or use final values for the next part of integration
    print(f"For b = {b_value}:")
    print("Final Values:", final_values)
    print("End Time:", end_time)

    # Check if any events occurred
    if sol.status == 0 and len(sol.t_events[0]) > 0:
        for i, event_time in enumerate(sol.t_events[0]):
            event_idx = np.where(t >= event_time)[0][0]
            t_event = sol.t_events[0][i]
            r_event = sol.y_events[0][i, 0]
            print(f"Event {i + 1} occurred at time {t_event:.2f} seconds.Radial distance:{r_event:.2f}")
        else:
            event_idx = len(t) - 1  # Plot until the end if no event occurred
            print("Integration completed without reaching the specified event condition.")
    
    print("Integration Status:", sol.message)
    print("Number of Events Detected:", len(sol.t_events[0]))
    # Print minimum values of sol.y for each component
    min_values = np.min(sol.y, axis=1)
    min_indices = np.argmin(sol.y, axis=1)
    min_times = sol.t[min_indices]

    print(f"For b = {b_value}:")
    print("Minimum Values of sol.y:", min_values)
    print("Corresponding Time Points:", min_times)
    trajectories[b_value] = sol

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                                        VISUALISATION
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
from mpl_toolkits.mplot3d import Axes3D  # Import 3D toolkit if not already
from collections import OrderedDict  # For legend handling
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd

from matplotlib import cm
rainbow_r_cmap = cm.rainbow_r
colors = [rainbow_r_cmap(i) for i in np.linspace(0, 1, len(b_values))]

from matplotlib.ticker import ScalarFormatter, AutoLocator, FuncFormatter

b_zorders = {1e-9: 710, 5e-9: 9, 10e-9: 5, 1e-7: 4, 1e-6: 3, 5e-6: 8, 10e-6: 2}
#-----------------------------------------------------------------------------------------------
#                                   Time vs Radial vs Phi vs Theta speeds
#---------------------------------------------------------------------------------------------
# Create the main figure and axis
fig, ax_r = plt.subplots(figsize=(12, 8), dpi=270)
ax_r.grid(color='grey', linestyle='dotted', linewidth=0.6, alpha=1)

# Create a secondary x-axis for time
ax_time = ax_r.twiny()

# Create a secondary y-axis for colatitude
ax_theta = ax_r.twinx()

# Plot on the main axis for each b_value in `trajectories`
for i, (b_value, sol) in enumerate(trajectories.items()):
    #z_order = b_zorders.get(b_value, 2)
    
    r_sol      = sol.y[0, :]
    phi_sol    = sol.y[4, :] *(180 /np.pi)  # Azimuthal angle in degrees
    theta_sol  = sol.y[2, :] *(180 /np.pi)  # Convert to degrees for better readability
    
    color = colors[i]  # Assign color based on the index
    
    # Plot radial distance vs azimuthal angle on the main axis
    ax_r.plot(phi_sol, r_sol, linewidth=2.5, linestyle='solid', alpha=1, color=color, label=f'{b_value:.1e}m')#,zorder=z_order)
    ax_r.plot(phi_sol[-1], r_sol[-1], 'o', color=color, markeredgecolor='white', markersize=7)#,zorder=z_order)
    
    # Plot colatitude vs azimuthal angle on the secondary y-axis
    ax_theta.plot(phi_sol, theta_sol, linewidth=2.5, linestyle='dotted', alpha=1, color=color, label=f'{b_value:.1e}m')#,zorder=z_order)
    ax_theta.plot(phi_sol[-1], theta_sol[-1], 'o', color=color, markeredgecolor='white', markersize=7)#,zorder=z_order)
    
    # Plot time on the secondary x-axis
    ax_time.plot(sol.t, sol.y[0, :], linewidth=0, linestyle='solid', alpha=0)  # Transparent line to maintain x-axis scaling

# Set labels for the axes
ax_r.set_xlabel('Azimuth (deg)', fontsize=18)
ax_r.set_ylabel('Radial Distance (m)', fontsize=18)
ax_theta.set_ylabel('Colatitude (deg)', fontsize=18)
ax_time.set_xlabel('Time (s)', fontsize=18)
ax_time.invert_xaxis()

# Custom formatter to display ticks in exponential form with a single scale at the corner
scalar_formatter = ScalarFormatter(useMathText=True)
scalar_formatter.set_powerlimits((0, 0))
scalar_formatter.set_scientific(True)
scalar_formatter.set_useOffset(False)

# Apply formatter to the axes
ax_r.xaxis.set_major_formatter(scalar_formatter)
ax_time.xaxis.set_major_formatter(scalar_formatter)

# Customize tick parameters
ax_r.tick_params(axis='y', labelsize=18)
ax_r.tick_params(axis='x', labelsize=18)
ax_theta.tick_params(axis='y', labelsize=18)
ax_time.tick_params(axis='x', labelsize=18)

# Customize grid lines
ax_theta.grid(color='grey', linestyle='dotted', linewidth=0.6, alpha=1)
ax_time.grid(color='grey', linestyle='dotted', linewidth=0.6, alpha=1)

"""
# Add legends for radial distance
lines_r, labels_r = ax_r.get_legend_handles_labels()
radial_legend = ax_r.legend(lines_r, labels_r, loc='upper left', bbox_to_anchor=(1.3, 0.95), 
                            borderaxespad=0., title="b_values for r", fontsize=11, title_fontsize=11, ncol=1)
ax_r.add_artist(radial_legend)  # Add the radial legend manually to the plot

# Add legends for colatitude
lines_theta, labels_theta = ax_theta.get_legend_handles_labels()
colatitude_legend = ax_theta.legend(lines_theta, labels_theta, loc='upper left', bbox_to_anchor=(1.8, 0.95), 
                                    borderaxespad=0., title="b_values for θ", fontsize=11, title_fontsize=11, ncol=1)
ax_theta.add_artist(colatitude_legend)  # Add the colatitude legend manually to the plot
"""
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#        Time vs Radial speed and Colatitudinal speed'
b_zorders = {1e-9: 10, 5e-9: 9, 10e-9: 7, 1e-7: 4, 1e-6: 3, 5e-6: 8, 10e-6: 2}
# Create the main figure and axis
fig, ax_left = plt.subplots(figsize=(12, 8), dpi=270)

# Secondary x-axis for time
#ax_time = ax_left.twiny()

# Secondary y-axis for colatitudinal speed
ax_theta = ax_left.twinx()
ax_theta.set_ylabel('dθ/dt (deg/s)', fontsize=18)

# Plot each b_value trajectory
for i, (b_value, color) in enumerate(zip(b_values, colors)):
    #z_order = b_zorders.get(b_value, 2)
    
    sol     = trajectories[b_value]
    d_r     = sol.y[1, :]  # Radial speed
    d_theta = sol.y[3, :] * (180 /np.pi)  # Colatitude speed in degrees/s
    d_phi   = sol.y[5, :] * (180 /np.pi)  # Azimuthal speed in degrees/s
    
    # Plot d_phi vs d_r/dt on the main y-axis
    ax_left.plot(d_phi, d_r, linestyle='solid', linewidth=2.5, color=color, label=f'{b_value:.1e} m')#,zorder=z_order)
    ax_left.plot(d_phi[-1], d_r[-1], 'o', color=color, markeredgecolor='white', markersize=7)#,zorder=z_order)
    
    # Plot d_phi vs d_theta/dt on the secondary y-axis
    ax_theta.plot(d_phi, d_theta, linestyle='dotted', linewidth=2.5, color=color)#,zorder=z_order)
    ax_theta.plot(d_phi[-1], d_theta[-1], 'o', color=color, markeredgecolor='white', markersize=7)#,zorder=z_order)
    
    # Plot time on the secondary x-axis
    #ax_time.plot(sol.t, d_phi, linewidth=0, alpha=0)  # Transparent plot for time scaling

# x-axis formatter for scientific notation with a single scale factor
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))  # Use scientific notation for all numbers
formatter.set_scientific(True)
formatter.set_useOffset(False)

# Apply formatter to the main x-axis and y-axis
ax_left.xaxis.set_major_formatter(formatter)
ax_left.yaxis.set_major_formatter(formatter)

# Apply formatter to the secondary x-axis
#ax_time.xaxis.set_major_formatter(scalar_formatter)

# Add the scale factor as a label at the corner of the x-axis
ax_left.ticklabel_format(axis='x', style='scientific', scilimits=(-2, 2))
ax_left.xaxis.get_offset_text().set_fontsize(18)
ax_left.xaxis.get_offset_text().set_color("black")
ax_left.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

#ax_time.xaxis.get_offset_text().set_fontsize(18)
#ax_time.xaxis.get_offset_text().set_color("black")

# Customize the secondary x-axis (time)
##ax_time.set_xlim(0, max([max(trajectories[b_val].t) for b_val in b_values]))
#ax_time.set_xlabel('Time (s)', fontsize=18)
#ax_time.tick_params(axis='x', labelsize=18)

# Rotate x-axis ticks at 45 degrees
for label in ax_left.get_xticklabels():
    label.set_rotation(45)
for label in ax_left.get_yticklabels():
    label.set_rotation(45)
    
# Customize the main and secondary y-axes
ax_left.set_xlabel('dϕ/dt (deg/s)', fontsize=18)
ax_left.set_ylabel('dr/dt (m/s)', fontsize=18)
ax_left.tick_params(axis='x', labelsize=18)
ax_left.tick_params(axis='y', labelsize=18)
ax_theta.tick_params(axis='y', labelsize=18)

# Add grids for all axes
ax_left.grid(color='grey', linestyle='dashed', linewidth=0.6, alpha=0.9)
#ax_time.grid(color='grey', linestyle='dotted', linewidth=0.6, alpha=0.5)
ax_theta.grid(color='grey', linestyle='dotted', linewidth=0.6, alpha=0.5)

"""
# Create insets for radial and colatitude plots
ax_inset_r = inset_axes(ax_r, width="60%", height="60%", loc='lower left', bbox_to_anchor=(0.1, 0.5, 0.6, 0.7), bbox_transform=ax_r.transAxes)
ax_inset_theta = inset_axes(ax_r, width="60%", height="60%", loc='lower left', bbox_to_anchor=(0.7, 0.1, 0.6, 0.7), bbox_transform=ax_r.transAxes)

# Customize insets
ax_inset_r.set_facecolor('white')
ax_inset_theta.set_facecolor('white')
ax_inset_r.grid(color='grey', linestyle='dotted', linewidth=0.6, alpha=1)
ax_inset_theta.grid(color='grey', linestyle='dotted', linewidth=0.6, alpha=1)

# Adjust axis limits for insets
ax_inset_r.set_xlim(12, 13)  # Example limit for x-axis of radial inset (azimuth angle)
ax_inset_r.set_ylim(7.10e7, 7.12e7)  # Example limit for y-axis of radial inset (radial distance)
ax_inset_theta.set_xlim(12.4, 12.6)  # Example limit for x-axis of colatitude inset (azimuth angle)
ax_inset_theta.set_ylim(89.90, 90.01)  # Example limit for y-axis of colatitude inset (colatitude angle in degrees)

# Plot on the insets
for i, (b_value, sol) in enumerate(trajectories.items()):
    r_sol = sol.y[0, :]
    phi_sol = sol.y[4, :] * (180 / np.pi)  # Azimuthal angle in degrees
    theta_sol = sol.y[2, :] * (180 / np.pi)  # Convert to degrees for better readability
    
    color = colors[i]  # Assign color based on the index
    
    # Plot radial distance vs azimuthal angle on the inset for radial plot
    ax_inset_r.plot(phi_sol, r_sol, linewidth=1.3, linestyle='solid', alpha=1, color=color, label=f'{b_value:.1e}m')
    ax_inset_r.plot(phi_sol[-1], r_sol[-1], 'o', color=color, markeredgecolor='white', markersize=6)
    
    # Plot colatitude vs azimuthal angle on the inset for colatitude plot
    ax_inset_theta.plot(phi_sol, theta_sol, linewidth=1.3, linestyle='dotted', alpha=1, color=color, label=f'{b_value:.1e}m')
    ax_inset_theta.plot(phi_sol[-1], theta_sol[-1], 'o', color=color, markeredgecolor='white', markersize=7)

"""
plt.tight_layout()
plt.show()

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#               POSTIONALA VARIABLES
#-----------------------------------------------------------------------------------------
"""
# Create separate figures for each positional variable
fig_r, ax_r = plt.subplots(dpi=260)
ax_r.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8)

fig_x, ax_x = plt.subplots(dpi=260)
ax_x.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8)

fig_theta, ax_theta = plt.subplots(dpi=260)
ax_theta.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8)

fig_y, ax_y = plt.subplots(dpi=260)
ax_y.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8)

fig_phi, ax_phi = plt.subplots(dpi=260)
ax_phi.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8)

fig_z, ax_z = plt.subplots(dpi=260)
ax_z.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8)
#..............................................................................
# Plot on individual plots for each b_value in `trajectories`
for i, (b_value, sol) in enumerate(trajectories.items()):
    color = colors[i]  # Assign color based on the index

    # Plot radial distance vs time
    ax_r.plot(sol.t, sol.y[0, :], linewidth=2.5, linestyle='solid', alpha=1, color=color, label=f'{b_value:.1e} m')
    ax_r.plot(sol.t[-1], sol.y[0, -1], 'o', color=color, markeredgecolor='white', markersize=6)
    ax_r.text(sol.t[-1], sol.y[0, -1], f'{b_value:.1e} m', fontsize=10, color=color, ha='right', va='bottom')

    # Plot radial speed vs time
    ax_x.plot(sol.t, sol.y[1, :], linewidth=2.5, linestyle='solid', alpha=1, color=color, label=f'{b_value:.1e} m')
    ax_x.plot(sol.t[-1], sol.y[1, -1], 'o', color=color, markeredgecolor='white', markersize=6)
    ax_x.text(sol.t[-1], sol.y[1, -1], f'{b_value:.1e} m', fontsize=10, color=color, ha='right', va='bottom')

    # Plot co-latitude vs time
    ax_theta.plot(sol.t, sol.y[2, :] * (180 / np.pi), linewidth=2.5, linestyle='solid', alpha=1, color=color, label=f'{b_value:.1e} m')
    ax_theta.plot(sol.t[-1], sol.y[2, -1] *(180 /np.pi), 'o', color=color, markeredgecolor='white', markersize=6)
    ax_theta.text(sol.t[-1], sol.y[2, -1] *(180 /np.pi), f'{b_value:.1e} m', fontsize=10, color=color, ha='right', va='bottom')

    # Plot co-latitude speed vs time
    ax_y.plot(sol.t, sol.y[3, :] * (180 / np.pi), linewidth=2.5, linestyle='solid', alpha=1, color=color, label=f'{b_value:.1e} m')
    ax_y.plot(sol.t[-1], sol.y[3, -1] *(180 /np.pi), 'o', color=color, markeredgecolor='white', markersize=6)
    ax_y.text(sol.t[-1], sol.y[3, -1] *(180 /np.pi), f'{b_value:.1e} m', fontsize=10, color=color, ha='right', va='bottom')

    # Plot azimuthal angle vs time
    ax_phi.plot(sol.t, sol.y[4, :] * (180 / np.pi), linewidth=2.5, linestyle='solid', alpha=1, color=color, label=f'{b_value:.1e} m')
    ax_phi.plot(sol.t[-1], sol.y[4, -1] * (180 / np.pi), 'o', color=color, markeredgecolor='white', markersize=6)
    ax_phi.text(sol.t[-1], sol.y[4, -1] * (180 / np.pi), f'{b_value:.1e} m', fontsize=10, color=color, ha='right', va='bottom')

    # Plot azimuthal speed vs time
    ax_z.plot(sol.t, sol.y[5, :] * (180 / np.pi), linewidth=2.5, linestyle='solid', alpha=1, color=color, label=f'{b_value:.1e} m')
    ax_z.plot(sol.t[-1], sol.y[5, -1] * (180 / np.pi), 'o', color=color, markeredgecolor='white', markersize=6)
    ax_z.text(sol.t[-1], sol.y[5, -1] * (180 / np.pi), f'{b_value:.1e} m', fontsize=10, color=color, ha='right', va='bottom')

#...................................................................................................................................
# Set common labels for each axis

ax_r.set_xlabel('Time (sec)', fontsize=18)
ax_x.set_xlabel('Time (sec)', fontsize=18)
ax_theta.set_xlabel('Time (sec)', fontsize=18)
ax_y.set_xlabel('Time (sec)', fontsize=18)
ax_phi.set_xlabel('Time (sec)', fontsize=18)
ax_z.set_xlabel('Time (sec)', fontsize=18)

ax_r.set_ylabel('Radial Distance (m)', fontsize=18)
ax_x.set_ylabel('Radial Speed (m/s)', fontsize=18)
ax_theta.set_ylabel('Co-latitude (deg)', fontsize=18)
ax_y.set_ylabel('Co-latitude Speed (deg/s)', fontsize=18)
ax_phi.set_ylabel('Azimuthal Angle (deg)', fontsize=18)
ax_z.set_ylabel('Azimuthal Speed (deg/s)', fontsize=18)

# Set tick parameters for better readability
ax_r.tick_params(axis='both', which='major', labelsize=18)
ax_x.tick_params(axis='both', which='major', labelsize=18)
ax_theta.tick_params(axis='both', which='major', labelsize=18)
ax_y.tick_params(axis='both', which='major', labelsize=18)
ax_phi.tick_params(axis='both', which='major', labelsize=18)
ax_z.tick_params(axis='both', which='major', labelsize=18)

# Apply tight layout to each figure to prevent overlap
fig_r.tight_layout()
fig_x.tight_layout()
fig_theta.tight_layout()
fig_y.tight_layout()
fig_phi.tight_layout()
fig_z.tight_layout()

# Show the figures
plt.show()
"""