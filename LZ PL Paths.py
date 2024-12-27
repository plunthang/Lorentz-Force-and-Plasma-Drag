# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 06:38:37 2024

@author: sugnu
"""
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import OrderedDict  # For legend handling
from mpl_toolkits.mplot3d import Axes3D  # Import 3D toolkit if not already
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

# Constants and functions
μ = 379312077e8            # m^3/s^2
R = 60268e3                # metre
g_10 = 21141e-9
Ω = 1.74e-4                # rad/s
j_2 = 1.629071e-2    # J2 perturbation coefficient

# charge by mass ratio
ρ = 1e3                    # 1gm/cm^3 Headmen 2021= 10^3 kg/m^3
# b = 1e-6                  # = 1e-3 micro
V = 10                     # Volt
ε = 8.85e-12               # Farad/metre
μ_0 = 4*np.pi*1e-7

# Initial parameters
r0 = 67628e3               # Initial radial distance

# Initial conditions for the plasma ions
r_i = 657e5  # 1.12 * R
x_i = 0.0
θ_i = np.pi / 2
y_i = 0.0
z_i = (μ /r_i**3) ** 0.5

# Initial conditions of the charged particle
x0 = 0.0
θ0 = np.pi / 2
y0 = 0.0
z_0 = (μ /r0**3)**0.5
n_i = 1e-3                 # Plasma ion density

# Define relative velocity to use in the ODEs
w_relative = np.sqrt((x_i - x0)**2 + (r_i*y_i - r0*y0) **
                     2 + (r_i*z_i*np.sin(θ_i) - r0*z_0*np.sin(θ0))**2)

z_kep    = (μ/(r0**3))**0.5
Co_y_kep = z_kep/r0

b_values = np.array([1e-9,5e-9, 1e-8, 1e-7,1e-6,5e-6,10e-6])
#b_values  = np.array([1e-7,1e-6])

def event_func_r(t, p):
    return p[0] - 60968e3  # R+700km kollman et al. 2018
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
        dxdt = r * ((y**2 + ((z + Ω)**2) * (np.sin(θ)**2)) -
                    β*z*np.sin(θ)*B_θ) - a_plasma * x

        dθdt = y
        dydt = (-2*x*y + r*((z + Ω)**2) * np.sin(θ)*np.cos(θ) +
                (β*r*z*np.sin(θ)*B_r))/r - a_plasma * y

        dϕdt = z
        dzdt = (-2*(z + Ω)*(x*np.sin(θ) + r*y*np.cos(θ)) + β *
                (x*B_θ - r*y*B_r)) /(r*np.sin(θ) + 1e-15) - a_plasma * z

        return np.array([drdt, dxdt, dθdt, dydt, dϕdt, dzdt])

        # time window
    t_span = (0, 86400)
    # t = np.linspace(t_span[0],t_span[1],3600)

    dt_desired = 1  # Desired time step in seconds
    num_points = int((t_span[1] - t_span[0]) /dt_desired) + 1
    t = np.linspace(t_span[0], t_span[1], num_points)

    # initial conditions
    p0 = np.array([r0,0, 90.0*(pi/180), 0, 0.0*(pi/180), 1.74e-4])

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
            print(f"Event {
                  i + 1} occurred at time {t_event:.2f} seconds.Radial distance:{r_event:.2f}")
        else:
            event_idx = len(t) - 1  # Plot until the end if no event occurred
            print("Integration completed without reaching the specified event condition.")

    print("Integration Status:", sol.message)
    print("Number of Events Detected:", len(sol.t_events[0]))
    """
    # Print minimum values of sol.y for each component
    min_values = np.min(sol.y, axis=1)
    min_indices = np.argmin(sol.y, axis=1)
    min_times = sol.t[min_indices]

    print(f"For b = {b_value}:")
    print("Minimum Values of sol.y:", min_values)
    print("Corresponding Time Points:", min_times)
    """

    trajectories[b_value] = sol

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                                        VISUALISATION
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

rainbow_r_cmap = cm.rainbow_r
colors = [rainbow_r_cmap(i) for i in np.linspace(0, 1, len(b_values))]

# zorders for each b_value
b_zorders = {1e-9: 710, 5e-9: 9, 10e-9: 5, 1e-7: 4, 1e-6: 3, 5e-6: 8, 10e-6: 2}

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                               Plot all trajectories in polar plot
# ---------------------------------------------------------------------------------------------
"""
fig_polar, axs_polar = plt.subplots(
    subplot_kw={'projection': 'polar'}, dpi=250)
axs_polar.set_facecolor('black')

# Planet representation
planet = plt.Circle(
    (0, 0), R, transform=axs_polar.transData._b, color='#EE9A49', alpha=0.8)
axs_polar.add_patch(planet)
for idx, (b_value, sol) in enumerate(trajectories.items()):
    r_sol = sol.y[0, :]
    phi_sol = sol.y[4, :]
    axs_polar.plot(phi_sol, r_sol, linewidth=1.5, linestyle='solid', alpha=1, color=colors[idx],
                   label=f'{b_value:.1e} m', zorder=10)
    axs_polar.plot(phi_sol[-1], r_sol[-1], 'o',
                   color=colors[idx], markersize=3, markeredgecolor='white')

# Configure polar plot
axs_polar.set_rlim(0, 7.93e7)
axs_polar.set_rticks([R, 60968e3, 676e5, 7156e4, 7327e4, 7449e4])
axs_polar.set_yticklabels([])
axs_polar.set_rlabel_position(30)
axs_polar.grid(color='white', linestyle='dotted', linewidth=0.4, alpha=1)
axs_polar.tick_params(axis='x', labelsize=16, pad=7)

# Add legend
axs_polar.legend(loc='upper left', bbox_to_anchor=(1.2, 0.7), borderaxespad=0.,
                 title="Legend", fontsize=10, title_fontsize=10, ncol=1)
"""
# ..................................................................................
"""
# Radial and time limits for the inset view
r_min, r_max = 6e7, 7.2e7  # Radial range in meters
t_min, t_max = 0, 36000       # Time range in seconds

# Define the radii for the concentric circles (rings) ,labels and text angles
ring_info = [
    (60968e3, "ATM", 355, 0),
    (676e5, "D68", 0, -1000),
    (7156e4, "D72", 344, -0.6e6),
    (7327e4, "D73", 60, -2e6),
    (7449e4, "C", 85, -5e6)]

# Loop through each item and add labels with specific position adjustments
for radius, label, angle_deg, radius_offset in ring_info:
    angle_rad = np.radians(angle_deg)  # Convert angle to radians
    adjusted_radius = radius + radius_offset  # Apply radial offset

    # Customize alignment and position
    axs_polar.text(
        angle_rad, adjusted_radius, label,
        ha='right', va='baseline', color='white', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.2', edgecolor='none', facecolor='none', alpha=1))
"""
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# INSET AXES for zoomed-in view
"""
ax_inset = inset_axes(axs_polar, width="68%", height="68%", loc='lower left',
                      bbox_to_anchor=(0.07, 0.16, 0.9, 0.99), bbox_transform=axs_polar.transAxes)
ax_inset.set_facecolor('black')
ax_inset.tick_params(axis='x', labelsize=14,labelcolor='skyblue') 
ax_inset.tick_params(axis='y', labelsize=14,labelcolor='skyblue')
ax_inset.grid(color='gray', linestyle='--', linewidth=0.7, alpha=1)
# Color the inset border 
ax_inset.patch.set_edgecolor('white') 
ax_inset.patch.set_linewidth(2)
# Loop through trajectories and plot only the specified range in the INSET
for idx, (b_value, sol) in enumerate(trajectories.items()):
    r_sol = sol.y[0, :]        # Radial distance
    phi_sol = sol.y[4, :] * (180 /np.pi)     # Azimuthal angle
    time_sol = sol.t           # Time array

    # Apply radial range filter
    mask_radial = (r_sol >= r_min) & (r_sol <= r_max)
    
    # Apply time range filter
    mask_time = (time_sol >= t_min) & (time_sol <= t_max)
    
    # Combine radial and time filters
    combined_mask = mask_radial & mask_time
    
    # Filtered data for radial distance, azimuth, and time
    r_filtered = r_sol[combined_mask]
    phi_filtered = phi_sol[combined_mask]
    
    # Plot the filtered data in the inset
    ax_inset.plot(r_filtered, phi_filtered, linewidth=1.0, linestyle='solid', color=colors[idx], label=f'{b_value:.1e} m')
    #ax_inset.plot(r_filtered[-1], phi_filtered[-1], 'o', color=colors[idx], markersize=5)
"""
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                                    POLAR NO CENTRE
# -----------------------------------------------------------------------------------------------
"""
fig_polar_no_center, axs_polar_no_center = plt.subplots(
    subplot_kw={'projection': 'polar'}, dpi=300)
axs_polar_no_center.set_facecolor('black')

# Plot each trajectory
for idx, (b_value, sol) in enumerate(trajectories.items()):
    r_sol   = sol.y[0, :]
    phi_sol = sol.y[4, :]

    # Plot the line with markers
    axs_polar_no_center.plot(phi_sol, r_sol, linewidth=0.5, linestyle='solid',
                             label=f'{b_value:.1e} m', alpha=0.9, color=colors[idx], zorder=10)

    # Mark the end point with a filled circle
    axs_polar_no_center.plot(phi_sol[-1], r_sol[-1], 'o', color=colors[idx],
                             markeredgecolor='white', markersize=6, zorder=10)

# Set radial limits and configure ticks
axs_polar_no_center.set_rlim(R, 7.34e7)
axs_polar_no_center.set_rticks([R, 60968e3, 676e5, 7.15e7, 7.32e7])
axs_polar_no_center.set_yticklabels([])  # hide labels

# Move radial ticks outward and change size/color for clarity
axs_polar_no_center.tick_params(axis='y', labelsize=15, pad=5, colors='black')

# Customize labels and grid
axs_polar_no_center.set_rlabel_position(30)
# color='gray', linestyle='--', linewidth=1, alpha=1)
axs_polar_no_center.grid(False)
axs_polar_no_center.tick_params(axis='x', labelsize=15, pad=5)

# The radii for the concentric circles (rings) and their labels
ring_info = [
    (60968e3, "Atm", 5, 0),  # Tuple: (radius, label, angle in degrees, radius offset)
    (676e5, "D68", 0, -1000),
    (7156e4, "D72", 90, 0)]

# Draw concentric circles
for radius, _, _, _ in ring_info:
    theta = np.linspace(0, 2 * np.pi, 100)
    axs_polar_no_center.plot(theta, [radius]*len(theta), color='dimgray', linestyle='dotted',
                             linewidth=3, alpha=0.8, zorder=1)

# Loop through each item and add labels with specific position adjustments
for radius, label, angle_deg, radius_offset in ring_info:
    angle_rad = np.radians(angle_deg)            # Convert angle to radians
    adjusted_radius = radius + radius_offset     # Apply radial offset

    # Customize alignment and position as needed
    axs_polar_no_center.text(
        angle_rad, adjusted_radius, label,
        ha='right', va='baseline', color='white', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.2', edgecolor='none', facecolor='none', alpha=1))
"""
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                                   3D cartesian
# ----------------------------------------------------------------------------------

# Create the first 3D figure
fig_3d_spherical = plt.figure(dpi=330)
ax_3d_spherical = fig_3d_spherical.add_subplot(111, projection="3d")

# Pane color (black)
pane_color = (0.0, 0.0, 0.0, 1.0)
ax_3d_spherical.xaxis.set_pane_color(pane_color)
ax_3d_spherical.yaxis.set_pane_color(pane_color)
ax_3d_spherical.zaxis.set_pane_color(pane_color)

# Set labels for spherical coordinates
ax_3d_spherical.set_xlabel('X', fontsize=12, labelpad=5)
ax_3d_spherical.set_ylabel('Y', fontsize=12, labelpad=5)
ax_3d_spherical.set_zlabel('Z', fontsize=12, labelpad=5)

# Customize tick colors, font size, and tick font size
ax_3d_spherical.tick_params(axis='x', labelsize=12, pad=3)
ax_3d_spherical.tick_params(axis='y', labelsize=12, pad=3)
ax_3d_spherical.tick_params(axis='z', labelsize=12, pad=3)

# Customize grid line colors and size
ax_3d_spherical.grid(True, color='gray', linestyle='-',
                     linewidth=0.3, alpha=0.5)
# --------------------------------------------------------------------------------

# Create the second 3D figure
fig_3d_with_sphere = plt.figure(dpi=330)
ax_3d_with_sphere = fig_3d_with_sphere.add_subplot(111, projection="3d")

ax_3d_with_sphere.xaxis.set_pane_color(pane_color)
ax_3d_with_sphere.yaxis.set_pane_color(pane_color)
ax_3d_with_sphere.zaxis.set_pane_color(pane_color)

# Set labels for spherical coordinates
ax_3d_with_sphere.set_xlabel('X', fontsize=15, labelpad=6)
ax_3d_with_sphere.set_ylabel('Y', fontsize=15, labelpad=6)
ax_3d_with_sphere.set_zlabel('Z', fontsize=15, labelpad=8)

# Customize tick colors, font size, and tick font size
ax_3d_with_sphere.tick_params(axis='x', labelsize=15, pad=5)
ax_3d_with_sphere.tick_params(axis='y', labelsize=15, pad=5)
ax_3d_with_sphere.tick_params(axis='z', labelsize=15, pad=7)

# Customize grid line colors and size
ax_3d_with_sphere.grid(True, color='gray', linestyle='-', linewidth=0.3)

# Plot trajectories in spherical polar coordinates
for i, (b_value, sol) in enumerate(trajectories.items()):
    color = colors[i]  # Assign color based on the index

    # zorder based on the custom mapping
    # z_order = b_zorders.get(b_value, 1)

    # 3D Trajectories around the sphere
    # Convert spherical coordinates to Cartesian coordinates
    r_sol = sol.y[0, :]
    theta_sol = sol.y[2, :]
    phi_sol = sol.y[4, :]
    x_traj = r_sol * np.sin(theta_sol) * np.cos(phi_sol)
    y_traj = r_sol * np.sin(theta_sol) * np.sin(phi_sol)
    z_traj = r_sol * np.cos(theta_sol)

    # Plot the trajectories in Cartesian coordinates (x, y, z)
    ax_3d_spherical.plot3D(x_traj, y_traj, z_traj, label=f'{b_value:.1e}',
                           linewidth=2.5, linestyle='solid', color=color, alpha=1, zorder=7)  # ,zorder=z_order)

    ax_3d_spherical.plot3D(x_traj[-1], y_traj[-1], z_traj[-1], marker='o',
                           color=color, markeredgecolor='white', markersize=6, zorder=7)  # ,zorder=z_order)

    ax_3d_with_sphere.plot3D(x_traj, y_traj, z_traj, label=f'{b_value:.1e}',
                             linewidth=2.5, linestyle='solid', color=color, alpha=1, zorder=7)  # ,zorder=z_order)

    ax_3d_with_sphere.plot3D(x_traj[-1], y_traj[-1], z_traj[-1], marker='o',
                             color=color, markeredgecolor='white', markersize=6, zorder=7)  # ,zorder=z_order)

# Add a legend for the first subplot
#ax_3d_spherical.legend(loc='upper left', bbox_to_anchor=(1.2, 0.8), borderaxespad=0.,
                       #title="Legend", fontsize=12, title_fontsize=12, ncol=1)
# ------------------------------------------------------------------------------------------

# Plot the sphere at the center in the second plot
phi_sphere   = np.linspace(0, 2 * np.pi, 100)
theta_sphere = np.linspace(0, np.pi, 100)
xm_sphere    = R * np.outer(np.cos(phi_sphere), np.sin(theta_sphere))
ym_sphere    = R * np.outer(np.sin(phi_sphere), np.sin(theta_sphere))
zm_sphere    = R * np.outer(np.ones(np.size(phi_sphere)), np.cos(theta_sphere))

ax_3d_with_sphere.plot_surface(xm_sphere, ym_sphere, zm_sphere, rstride=1, cstride=1,
                               cmap=plt.get_cmap('copper'), linewidth=1, antialiased=False, alpha=0.8)

# Radii for the concentric circles (rings) and their labels
ring_info   = [(60968e3, "ATM"), (676e5, "D68"), (7156e4, "D72")]

# Define the angles for text placement (in radians), spaced at least 10 degrees apart
text_angles = [np.radians(1), np.radians(320), np.radians(300)]

# Define the offset distance from the center
text_offset = 0.5e6

# Plot the concentric circles and add text annotations
for index, (radius, label) in enumerate(ring_info):
    theta = np.linspace(0, 2 * np.pi, 100)

    # Calculate x, y coordinates for the circle in the xy-plane at z = 0
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    z_circle = np.zeros_like(theta)  # z = 0 = equatorial plane

    # Plot the circles
    ax_3d_with_sphere.plot(x_circle, y_circle, z_circle, color='gray', alpha=1, linestyle="solid",
                           linewidth=1, zorder=4)

    # Predefined angles for text placement
    angle = text_angles[index]  # angle in radians

    # Text annotation for each ring with an offset
    ax_3d_with_sphere.text(
        (radius + text_offset) * np.cos(angle),  # offset to radius for x
        (radius + text_offset) * np.sin(angle),  # offset to radius for y
        0, label, color='white', fontsize=10, ha='left', va='baseline', zorder=6)

# Add a single legend for the second subplot
ax_3d_with_sphere.legend(loc='upper left', bbox_to_anchor=(1.3, 0.8), borderaxespad=0.,
                         title="Legend", fontsize=13, title_fontsize=13, ncol=1)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                       CARTESIAN with centre XY plane
# ------------------------------------------------------------------------------
"""
# Create the main figure and axis
fig_cart, ax_cart = plt.subplots(dpi=360)
ax_cart.set_facecolor('black')

# Plot the trajectories
for idx, (b_value, sol) in enumerate(trajectories.items()):
    
    #z_order = b_zorders.get(b_value, 5)

    r_sol   = sol.y[0, :]
    phi_sol = sol.y[4, :]
    x_sol   = r_sol * np.cos(phi_sol)
    y_sol   = r_sol * np.sin(phi_sol)

    ax_cart.plot(x_sol, y_sol, linewidth=2.5, linestyle='solid', label=f'{b_value:.1e} m', color=colors[idx])#,zorder=z_order)
    ax_cart.plot(x_sol[-1], y_sol[-1], marker='o', color=colors[idx],
                 markeredgecolor='white', markersize=7 )#, zorder=z_order)

# Add filled circle at the center
R = 60268000  # Radius in meters
circle = plt.Circle((0, 0), R, color='orange', alpha=1, zorder=7)
ax_cart.add_artist(circle)

# Concentric circles and their labels
ring_info = [(60968e3, "ATM"), (676e5, "D68"),(7156e4, "D72")]#, (7327e4, "D73")]

# Define the angles for text placement (in radians), spaced apart
text_angles = [np.radians(8), np.radians(357), np.radians(41)]#, np.radians(60)]

# Add concentric circles and labels
for index, (radius, label) in enumerate(ring_info):
    # Create a circle
    circle = plt.Circle((0, 0), radius, color='dimgray',
                        linestyle='solid', linewidth=5, alpha=0.8, fill=False)
    ax_cart.add_artist(circle)

    # Calculate the position for the text label
    angle  = text_angles[index]
    x_text = radius * np.cos(angle)
    y_text = radius * np.sin(angle)

    # Add text label
    ax_cart.text(x_text, y_text, label, color='white',
                 fontsize=13, ha='right', va='baseline', zorder=10)

# Customize plot appearance
ax_cart.set_xlabel("X Distance (m)", fontsize=13)
ax_cart.set_ylabel("Y Distance (m)", fontsize=13)
ax_cart.tick_params(axis='x', labelsize=13)
ax_cart.tick_params(axis='y', labelsize=13)
ax_cart.set_xlim(-1.5 * R, 1.5 * R)
ax_cart.set_ylim(-1.5 * R, 1.5 * R)
ax_cart.grid(color='gray', linestyle='--', linewidth=0.6, alpha=1, zorder=1)

plt.tight_layout()
plt.show()

"""
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                               CARTESIAN with centre YZ plane
#------------------------------------------------------------------------------

fig_cart1, ax_cart1 = plt.subplots(dpi=250)
ax_cart1.set_facecolor('black')

for idx, (b_value, sol) in enumerate(trajectories.items()):
    r_sol = sol.y[0, :]
    phi_sol = sol.y[4, :]
    theta_sol = sol.y[2, :]
    y_sol = r_sol * np.sin(phi_sol)
    z_sol = r_sol * np.cos(theta_sol)
    
    ax_cart1.plot(y_sol, z_sol, linewidth=1.5, linestyle='solid', label=f'{b_value:.1e} m', 
                  color=colors[idx],zorder=10)
    ax_cart1.plot(y_sol[-1], z_sol[-1], 'o', color=colors[idx], markeredgecolor='white', markersize=6)

# Add legend
#ax_cart1.legend(loc='upper left', fontsize=12)
ax_cart1.set_xlabel("Y Distance (m)", fontsize=18)
ax_cart1.set_ylabel("Z Distance (m)", fontsize=18)
# ax_cart1.set_title("YZ Plane Trajectories", fontsize=15)

# Set x and y ticks with desired spacing
ax_cart1.tick_params(axis='x', labelsize=18) 
ax_cart1.invert_xaxis()
ax_cart1.tick_params(axis='y', labelsize=18)
# Add grid
ax_cart1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8, zorder=1)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
