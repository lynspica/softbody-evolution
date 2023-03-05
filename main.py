# Modelling the growth of a soft-body system from a single cell and temporal adaptation to the environment
# Written by: Ahmet Burak Yıldırım, 2017
#
# To run this program, please run the cells in order
# The order is given by the numbers in the explanation of each cell
#
# 1- The program should first import the necessary libraries
# 2- Then open the drawer GUI and ask for an input map
# 3- And then add the input map to the system
# 4- By running the next cell, the user will define the functions
# 5- The last part will open up a window and one will be able to see the dynamics

#%% 1
# ===============================================
# This part includes the libraries to be imported
# ===============================================

import numpy as np # For mathematical operations
from matplotlib import pyplot as plt # For plotting and dynamics animation
import random # For random choice from a list, used while apoptosis
import time # For the option to wait a couple of seconds to check the console
from tkinter import Tk, Canvas, OptionMenu, StringVar # For painter GUI
import tkinter # For painter GUI
from tkinter import messagebox as mbox # For painter GUI
from tkinter.ttk import * # For painter GUI
import PIL # For painter GUI
from PIL import ImageDraw # For painter GUI
import cv2 # For creating GIF
import imageio # For creating GIF
from PIL import ImageGrab # For creating GIF

#%% 2
# ======================================================
# This part includes the construction of the drawing GUI 
# ======================================================

# Here we define a window for the GUI, and a canvas for drawing on it
root = Tk()
root.geometry("600x600")
screen_width = 500
screen_height = 500
cn = Canvas(root, bg='white', height=50, width=50)
canvas = Canvas(root, bg='white', height=500, width=500)
root.title('Paint')

# Initialization of brush colors, one must select a color before drawing
hx = None
value_color = None
rgb = None
col = (255, 255, 255)

# Drawing a circle with the selected brush size on the clicked point
def draw(event):
    value = variable.get()
    if value and hx:
        canvas.create_oval((event.x - int(value) // 2, event.y - int(value) // 2),
                           (event.x + int(value) // 2, event.y + int(value) // 2), fill=hx, outline=hx)
        drawer.ellipse((event.x - int(value) // 2, event.y - int(value) // 2,
                  event.x + int(value) // 2, event.y + int(value) // 2), hx)
    else:
        mbox.showerror("Error", "Choose a color for painting!!")

# Set background color to white, which means desired region in the simulations
def whitebackground():
    global image, drawer
    pixels = image.load()
    x, y = image.size
    for i in range(x):
        for j in range(y):
            pixels[i, j] = (255, 255, 255)
    canvas.config(bg='#ffffff')
    canvas.delete("all")

# Set background color to red, which means undesired region in the simulations
# One may draw a shape using white on red background to imply the desired cell
# structure to be formed
def redbackground():
    global image, drawer
    pixels = image.load()
    x, y = image.size
    for i in range(x):
        for j in range(y):
            pixels[i, j] = (255, 0, 0)
    canvas.config(bg='#ff0000')
    canvas.delete("all")

# After clicking save button, the window closes and we are ready for the next 
# code cell to be ran
def save():
    x0 = root.winfo_rootx()
    y0 = root.winfo_rooty()
    x1 = x0 + root.winfo_width()
    y1 = y0 + root.winfo_height()
    ImageGrab.grab((x0, y0, x1, y1)).save('map.png')
    root.destroy()

# Setting the brush color to red, helpful for drawing undesirable shapes
def redcolor():
    global hx, rgb
    rgb = (255, 0, 0) 
    hx = '#ff0000'
    cn.config(bg=hx)

# Similarly, setting the brush color to red, helpful for drawing desirable shapes
def whitecolor():
    global hx, rgb
    rgb = (255, 255, 255)
    hx = '#ffffff'
    cn.config(bg=hx)

# Constructing the GUI
image = PIL.Image.new('RGB', (screen_width, screen_height), 'white')
drawer = ImageDraw.Draw(image)

# Function assignments for the buttons etc.
redcolorbutton = tkinter.ttk.Button(root, text='Color: Undesirable Region', command=redcolor)
whitecolorbutton = tkinter.ttk.Button(root, text='Color: Desirable Region', command=whitecolor)
redbutton = tkinter.ttk.Button(root, text='Fill all: Undesirable Region', command=redbackground)
whitebutton = tkinter.ttk.Button(root, text='Fill all: Desirable Region', command=whitebackground)
label = tkinter.ttk.Label(root, text='Brush size:')
savebutton = tkinter.ttk.Button(root, text='Save', command=save)
canvas.bind("<B1-Motion>", draw)
canvas.bind('<Button-1>', draw)
variable = StringVar(root)
shr = OptionMenu(root, variable, 50, 25, 50, 75, 100)

# Position assignments for the buttons etc.
label.grid(column=0, row=0, padx=5, pady=0)
shr.grid(column=0, row=1, padx=5, pady=0)
redcolorbutton.grid(column=6, row=0)
whitecolorbutton.grid(column=6, row=1)
redbutton.grid(column=8, row=0, padx = 5)
whitebutton.grid(column=8, row=1, padx = 5)
savebutton.grid(column=10, row=0, padx = 0)
cn.grid(column=10, row=1, padx = 5)
canvas.place(x=50, y=90)
root.mainloop()

#%%  3
# ==============================================================
# This part includes the image assignments to U, N, A, M letters
# I have drawn the letters one by one and saved each into the
# corresponding variable (U, N, A, M)
# Later I have used these images without drawing anything new
# while running simulations for different cell diameters
# ==============================================================

# 
#U = image
#N = image
#A = image
#M = image
#  
# Making of Figure 1 in the report
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
#ax1.imshow(U)
#ax2.imshow(N)
#ax3.imshow(A)
#ax4.imshow(M)

# 'Condition map' is mainly the undesired/desired map that the drawing implies
# According to this map, cells will die if they move into undesired regions
# For U, N, A, M letters, I have inserted the corresponding variables to 
# construct 'condition map'
#condition_map = np.array(U)[:,:,1]/255
#imgplot = plt.imshow(image)
condition_map = np.array(image)[:,:,1]/255 # Using current drawing
#%% 4
# ==============================================================
# This part includes the main functions that runs the whole cell
# simulation
# The soft-body dynamics were solved using spring interactions
# combined into velocity-Verlet algorithm
# There is also a function dedicated to plotting, which takes 
# the obtained solution for the current 'event' (iteration) and
# visualized it a function of time
# ==============================================================

# Initial parametrization of the soft-body dynamics and a few more
starting_cells = 1 # We start with a single cell
cell_diameter = 1 # The diameter of cell, I have investigated this in the report
spring_constant = 0.5 # Spring constant between the cells
cell_mass = 1 # Cell mass, actually k/m is the thing that is important but anyway..
timestep = 0.1 # Timestep for the velocity-Verlet solver, 0.1 works fine
totalnumberofallowedcells = 128 # If the system reaches 128 cells, we stop

# Here we are defining some parameters related to the interactions between the 
# cells
#
# # If a cell have more interactions than 'Allowed interactions', the cell no
# longer divides and also gets marked with red, unless the interaction is 1,
# the cell gets marked with yellow and for 1 it becomes green
# The spring interaction limit is 'max interaction' * 'cell diameter'
allowed_interactions = 3 
maxinteraction = 1.5 

# Here I have introduced two coefficients for the momentum
#
# 'Collision momentum' is for the coefficient in the following formula:
# mv1 + mv2 -> CM*m(-v1) + CM*m(-v2)
# So, if we give it as 1, the collision is purely elastic
# For values between 0 and 1, we introduce an energy loss per collision
#
# 'Division momentum' is for the coefficient in the following formula:
# mv1 -> DM*m(-v1/2) + DM*m(v1/2)
# So, when a cell divides, the mother cell goes to the reverse direction
# and the newborn cell goes to the mother cell's initial direction
# If we give 'Division momentum' as 1, there is no momentum loss per division
# For values between 0 and 1, we introduce an momentum loss per division
# Note that the 1/2 part is implemented into the equation itself, so the 
# coefficient must be 1, not 1/2
collision_momentum = 1 # Momentum transfer coefficient when two cells collide 
division_momentum = 1 # Momentum transfer coefficient when a cell divide

# To quicken the simulation, I am reassigning some values into the total number
# of steps to be taken
# So, while there are cells to be deleted, I am running for 10 steps and if 
# there are no problem with the system, I am running for 50 steps
total_number_of_steps = 50 # This is only for the first run

# In each 'mod' timesteps we update the spring interaction between the cells
# In each 'plot mod' steps we update the plot, to speed up this can be increased
mod = 10 
plotmod = 2 

first_run = 1 # An identifier to check whether it is the first run or not
new_position_cells = [] # Creating some initial lists
new_velocity_cells = [] # Creating some initial lists
howmanycellswerekilled = 0 # Recording the apoptosis count

# This function is dedicated to velocity-Verlet algorithm
# In each timestep, we run this function to mainly obtain the cell position 
# and velocities in the next timestep
# Also, we determine the spring interactions, the cells to be deleted due to
# overlaps or no-interaction
# We output these variables to delete the cells accordingly and show the 
# interactions in the plot
def velocity_verlet(first_run,position_cells,velocity_cells,starting_cells,cell_diameter,spring_constant,cell_mass,timestep,collision_momentum,total_number_of_steps):
    # If we are in the first run, we put the single cell in x=0, y=0 position
    # with Vx=0, Vy=0 velocity
    if first_run == 1:
        position_cells = np.zeros((2, starting_cells,total_number_of_steps))
        velocity_cells = np.zeros((2, starting_cells,total_number_of_steps))
        first_run = 0 # We are no longer in the first run
    # If not, we take cell positions and velocities that are given as an input 
    # to this function
    # Since the function itself outputs these variables, we simply introduce 
    # the determined values in each event (iteration) with modifications
    # Between two events, where the mitosis might have occured or a cell may be 
    # deleted, the modified version of these position and velocity variables
    # were given as input to this function since the number of cells will change
    else:
        starting_cells = position_cells.shape[1] # How many cells we currently have
        # Previous last step = Current first step
        old_position_cells = position_cells 
        old_velocity_cells = velocity_cells
        position_cells = np.zeros((2, starting_cells,total_number_of_steps))
        velocity_cells = np.zeros((2, starting_cells,total_number_of_steps))
        # We will fill the further timesteps with each calculation
        position_cells[:,:,0] = old_position_cells
        velocity_cells[:,:,0] = old_velocity_cells
    # Determining the current spring interactions by obtaining the distance
    # between each cell
    interaction_matrix = np.zeros((starting_cells,starting_cells))
    distance_matrix = np.zeros((starting_cells,starting_cells))
    # Initially, let us consider all cells to interact with each other
    # Since we start with a single cell, it does not matter but this is for
    # completeness
    prev_interaction = np.ones((starting_cells, starting_cells))
    # We will save the interactions into 'save interaction' matrix which will
    # be later used in plotting
    save_interaction = np.empty((starting_cells, starting_cells, total_number_of_steps))
    save_interaction[:,:,0] = prev_interaction
    
    # Velocity-Verlet algorithm itself
    for t in range(0,total_number_of_steps-1):
        # We are determining the acceleration between the cells due to spring 
        # force by calling the 'get acceleration' function
        acceleration_cells, distance_matrix, interaction_matrix = get_acceleration(position_cells, starting_cells, cell_diameter, t, spring_constant, cell_mass, prev_interaction)
        # We obtain the acceleration from the previous function
        # We already had the position and velocities
        # Now we need to check the collisions and modify the velocities 
        # accordingly
        # Then we will be ready to go with the velocity Verlet
        
        collision_indices = np.argwhere(distance_matrix < 1*cell_diameter)
        for coll in collision_indices:
            # Since collision_indices will give both of the colliding cell
            # indices and also the say that the cell is colliding with itself
            # we will just use the upper triangular part of the matrix
            if coll[0] != coll[1] and coll[0]<=coll[1]:
                # Obtaining the components of delta(x) and delta(v) vectors
                position_diff_x =  position_cells[0,coll[1],t] - position_cells[0,coll[0],t]
                position_diff_y =  position_cells[1,coll[1],t] - position_cells[1,coll[0],t]
                velocity_diff_x = velocity_cells[0,coll[1],t] - velocity_cells[0,coll[0],t]
                velocity_diff_y = velocity_cells[1,coll[1],t] - velocity_cells[1,coll[0],t]
                # Checking whether the delta(x) * delta(y) is smaller than 0
                # which means the cells are not going out and needs treatment
                if position_diff_x*velocity_diff_x+position_diff_y*velocity_diff_y<0:
                    rv = np.array([velocity_diff_x, velocity_diff_y]) # delta(x)
                    rp = np.array([position_diff_x, position_diff_y]) # delta(v)
                    # Defining the new cell velocities after the collision
                    change = rv@rp * -rp/(np.linalg.norm(rp)**2)
                    velocity_cells[0,coll[0],t] = velocity_cells[0,coll[0],t] - collision_momentum*change[0]
                    velocity_cells[0,coll[1],t] = velocity_cells[0,coll[1],t] + collision_momentum*change[0]
                    velocity_cells[1,coll[0],t] = velocity_cells[1,coll[0],t] - collision_momentum*change[1]
                    velocity_cells[1,coll[1],t] = velocity_cells[1,coll[1],t] + collision_momentum*change[1]
                    
        # Now we can obtain position, velocity, and acceleration arrays (x,v,a)
        # using the velocity Verlet algorithm, where we also obtain the a(t+1)
        # by calling 'get acceleration' function
        position_cells[:,:,t+1] = position_cells[:,:,t] + velocity_cells[:,:,t] * timestep + acceleration_cells * timestep**2 / 2
        acceleration_cells_nextstep = get_acceleration(position_cells, starting_cells, cell_diameter, t+1, spring_constant, cell_mass, prev_interaction)[0]
        velocity_cells[:,:,t+1] = velocity_cells[:,:,t] + (acceleration_cells+acceleration_cells_nextstep) / 2 * timestep
        
        # Saving determined interactions to be used in the next event (iteration)
        prev_interaction = interaction_matrix
        save_interaction[:,:,t+1] = prev_interaction
        
        # Finding which cells are overlapping at the end of the event
        # We will delete these cells by random selection
        # We also find the non-interacting cells and delete them if the number
        # of alive cells is >16, to allow the system to move around quicker
        # in the beginning
        if t == total_number_of_steps-2:
            huge_collision = np.argwhere(distance_matrix < 0.75*cell_diameter)
            tobedeleted_overlapping = []
            tobedeleted_noninteracting = []
            if starting_cells>16:
                tobedeleted_noninteracting = np.argwhere(np.sum(prev_interaction, axis=0) == 0)
            for duo in huge_collision:
                if duo[0] < duo[1]:
                    tobedeleted_overlapping.append(duo[0])
        # Total interaction count per cell, which will be used while determining
        # the deletion type, the cells on the undesirable regions
        numberofinteractions = np.sum(interaction_matrix,axis=0)
    return first_run, position_cells, velocity_cells, save_interaction, tobedeleted_overlapping, tobedeleted_noninteracting, numberofinteractions

# 'Get acceleration' function, which calculates the cell-cell spring force, and
# outputs the acceleration
# While determining the acceleration, it also outputs the matrix giving the 
# distances between cells, and also the cell-cell interactions
# 'Interaction matrix' is a symmetric matrix by definition and consists of 1 
# and 0, according to the index of two cells
# If [i,j] index of the matrix is 0, it means ith cell does not interact with
# jth cell
def get_acceleration(position,no_cell,dia_cell,t,k,m,previous_interaction):
    # Calculating the distances between cells with each another and constructing
    # a matrix from this
    A_x = np.repeat(position[0,:,t], no_cell, axis = 0).reshape((no_cell, no_cell))
    B_x = np.transpose(A_x)
    A_y = np.repeat(position[1,:,t], no_cell, axis = 0).reshape((no_cell, no_cell))
    B_y = np.transpose(A_y)
    distance_matrix = np.sqrt((A_x-B_x)**2 + (A_y-B_y)**2)
    # This is the frequency in which we update the interactions
    if np.mod(t,mod) == 0 or t == 1: 
        # We give 1 if the 'max interaction' * 'cell diameter' > current distance
        interaction_matrix = np.where(distance_matrix < maxinteraction*dia_cell, 1, 0)
        interaction_matrix = np.where(interaction_matrix > 0*dia_cell, 1, 0)
        # Since [i,j] index of the matrix for i==j will be always 1 since the 
        # distance between the same cell with itself is 0, we manually set the 
        # diagonal part to 0 by ourselves
        interaction_matrix[np.diag_indices_from(interaction_matrix)] = 0
    # If we do not update the interactions at that particular timestep, we
    # use the interactions calculated in the previous timestep
    else:
        interaction_matrix = previous_interaction
    # Here we compute the spring force exerted on a single cell by all the cells
    # interacting with it, using the matrices
    # We will sum all the force in the following form F = k(x-x0) where x0 is 
    # simply the diameter of a cell (d = 2r) 
    distance_vector_x = A_x - B_x
    distance_vector_y = A_y - B_y
    distance_norm_x = np.zeros_like(distance_matrix)
    non_zero_x = distance_vector_x != 0
    distance_norm_x[non_zero_x] = (distance_vector_x[non_zero_x]/distance_matrix[non_zero_x])
    distance_norm_y = np.zeros_like(distance_matrix)
    non_zero_y = distance_vector_x != 0
    distance_norm_y[non_zero_y] = (distance_vector_y[non_zero_y]/distance_matrix[non_zero_y])      
    delta_x = np.sum(interaction_matrix*(distance_vector_x-distance_norm_x), axis=0)
    delta_y = np.sum(interaction_matrix*(distance_vector_y-distance_norm_y), axis=0)
    # Since k was constant, we just multiply with k/m:
    # a = k/m (some sum of (x-xo))
    acceleration = k/m*(np.array([delta_x,delta_y]))
    return acceleration, distance_matrix, interaction_matrix

# This functions takes the cell positions and interactions, and uses them to
# show the system behavior visually
# After we compute everything by the velocity Verlet algorithm at a particular
# timestep, we will plot the calculations and then go forward with the next
# event
def plot(fig,position_cells,save_interaction,numberofinteractions,im):
    fig.set_dpi(100)
    ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
    plt.axis('off')
    x = position_cells[0,:,:]
    y = position_cells[1,:,:]
    # Coloring of the cells
    colors = np.where(numberofinteractions < 2, 'g', np.where(numberofinteractions > allowed_interactions, 'r', 'y')) #allow
    for t in range(1,total_number_of_steps):
        if np.mod(t,plotmod) == 1:
            plt.cla()
            plt.axis('off')
            for cell in range(len(x)):
                circle1 = plt.Circle((x[cell,t], y[cell,t]), cell_diameter/2, color = colors[cell], fill=False)
                ax.add_artist(circle1)
            current_interaction = save_interaction[:,:,t]
            current_interaction = np.triu(current_interaction)
            for i, j in np.argwhere(current_interaction==1):
                line = plt.Line2D(((x[i,t], x[j,t])), (y[i,t], y[j,t]), lw=0.5,ls='--')
                plt.gca().add_line(line)
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            plt.pause(0.001) #MATLAB's drawnow equivalent
            fig = plt.gcf()
            img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
            im.append(img)
    return fig

#%% 5 
# ==============================================================
# This part includes the simulation itself and some printings
# on the console
# It also includes the mitosis commands, where the positions and
# velocities of the newborn cells were appended to the previous
# arrays
# If the mitosis process is not allowed at the current event 
# due to the existence of some cells to be deleted for different
# reasons (colliding, non-interacting, being in undesirable
# regions), this part also deals with the apoptosis process
# ==============================================================

fig = plt.figure(figsize=(5,5))
im = []

# I have limited the total event number to 999, where the system should adapt
# to the environement in 999 events, where the mitosis of the system is counted
# as one event and apoptosis of a single cell is also counted as one event

for iteration in range(1000):
    print('')
    print('#===========================================#')
    print('#')
    print('# Currently in event:', iteration)
    # We run and plot the simulation
    first_run, position_cells, velocity_cells, save_interaction, tobedeleted_overlapping, tobedeleted_noninteracting, numberofinteractions = velocity_verlet(first_run,new_position_cells,new_velocity_cells,starting_cells,cell_diameter,spring_constant,cell_mass,timestep,collision_momentum,total_number_of_steps)
    plot(fig, position_cells, save_interaction, numberofinteractions,im)
    
    # We obtain the positions and velocities of the cells at the last timestep
    positionsatlasttimestep =  position_cells[:,:,-1]
    velocitieslasttimestep = velocity_cells[:,:,-1]
    print('# Alive cells:', positionsatlasttimestep[0,:].shape[0])
    print('# Total apoptosis:', howmanycellswerekilled)
    print('#')
    # Undesired region test
    # We check each cells, whether its coordinates take place in desirable 
    # region or not
    petrilength = np.linspace(-10, 10, num=500)
    xindex = []
    yindex = []
    tobedeleted_undesirable = []
    # We let the system to adapt to the environment if the number of alive cells
    # is higher than 50
    # This allows the system to adopt to hollow shapes, where x=0, y=0 might
    # be undesirable
    if positionsatlasttimestep.shape[1]>50:
        for i in range(positionsatlasttimestep[0,:].shape[0]):
            xindex.append(np.abs(petrilength-positionsatlasttimestep[0,i]).argmin())
            yindex.append(np.abs(petrilength-positionsatlasttimestep[1,i]).argmin())
        for index in range(len(xindex)):
            if condition_map[-yindex[index], xindex[index]] == 0:
                tobedeleted_undesirable.append(index)
    #time.sleep(4) # For better visibility of the console printings       
    
    # If there is no cell to be deleted, we can perform mitosis
    if len(tobedeleted_overlapping) == 0 and len(tobedeleted_noninteracting) == 0 and len(tobedeleted_undesirable) == 0:
        if positionsatlasttimestep.shape[1]>totalnumberofallowedcells:
            break
        print('# The system can perform mitosis')
        print('#')
        print('#===========================================#')
        total_number_of_steps = 50 # After mitosis, we run for 50 timesteps
        # We randomly select number of alive cells times, angles between 0 and 2pi 
        # We later use these random angles for random generation of x and y 
        # coordinates by cos() and sin() operations
        randomangle = np.random.uniform(0,2*np.pi,positionsatlasttimestep.shape[1])
        # We will move the new cells 1 distance away from the mother cells
        newcellpositions = positionsatlasttimestep
        newcellvelocities = division_momentum/2*velocitieslasttimestep
        reversedvelocities = -division_momentum/2*velocitieslasttimestep
        saveindex = []
        # We select the cells who are allowed to divide, due to interaction
        # limit
        for i in range(len(numberofinteractions)-1,-1,-1): 
            if numberofinteractions[i] > allowed_interactions:
                randomangle = np.delete(randomangle, i, axis=0)
                newcellpositions = np.delete(newcellpositions, i, axis=1)
                newcellvelocities = np.delete(newcellvelocities, i, axis=1)
                saveindex.append(i)
        saveindex.sort()
        # This is the randomness of division vector for the new cells
        randommitosis_x = np.cos(randomangle)
        randommitosis_y = np.sin(randomangle)
        
        # I have also introduced a bias in the division, where the division
        # direction is determined %50 random + % bias
        # Bias represents the vector pointing outwards considering the
        # particular cell and the center of mass of the system
        dynamiccenterofmass = np.mean(positionsatlasttimestep,axis=1)
        dynamiccenterofmass_array = np.tile(dynamiccenterofmass, (newcellpositions.shape[1],1)).T
        bounceback_x = randommitosis_x
        bounceback_y = randommitosis_y
        if i > 2: # If there are more than 2 cells, we consider the bias
            bias = (newcellpositions-dynamiccenterofmass_array) / np.linalg.norm((newcellpositions-dynamiccenterofmass_array),axis=0)
            bias_bounceback = (newcellpositions-dynamiccenterofmass_array) / np.linalg.norm((newcellpositions-dynamiccenterofmass_array),axis=0)
        else: # Otherwise it is 0
            bias = np.zeros(newcellpositions.shape[1])
        for pos in saveindex:
            bounceback_x = np.insert(bounceback_x,pos,0)
            bounceback_y = np.insert(bounceback_y,pos,0)
            reversedvelocities[:,pos] = velocitieslasttimestep[:,pos]

        # New cell position = old cell position + %50 random position vector +
        # %50 bias position vector
        # Old cell position = old cell position
        # We previously mentioned the velocity vectors for the mother and 
        # newborn cells
        randommitosis = newcellpositions + 1/2*np.array((randommitosis_x,randommitosis_y)) + 1/2*bias
        bounceback = positionsatlasttimestep #- 1/4*np.array((bounceback_x,bounceback_y)) - 1/4*bias_bounceback
        new_position_cells = np.hstack((bounceback,randommitosis))
        new_velocity_cells = np.hstack((reversedvelocities,newcellvelocities))
        
    # If there is atleast one cell to be deleted, we cannot perform mitosis
    else:
        print('# The system cannot perform mitosis')
        total_number_of_steps = 10 # We run for 10 steps after the apoptosis (25 is also fine)
        howmanycellswerekilled += 1
        # We randomly select a cell from the list of cells to be deleted
        if len(tobedeleted_undesirable) != 0:
            index = random.choice(tobedeleted_undesirable)
            print('#')
            print('# There are some cells in the undesired region.')
            print('# Randomly selecting one cell from the following undesirable cells:')
            print('#', tobedeleted_undesirable)
            print('#')
        elif len(tobedeleted_noninteracting) == 0:
            index = random.choice(tobedeleted_overlapping)
            print('#')
            print('# There are overlapping cells.')
            print('# Randomly selecting one cell from the following overlapping cells:')
            print('#', tobedeleted_overlapping)
            print('#')
        else:
            index = random.choice(tobedeleted_noninteracting)[0]
            print('#===========================================#')
            print('# There are noninteracting cells.')
            print('# Randomly selecting one cell from the following noninteracting cells:')
            print('#', list(tobedeleted_noninteracting.T[0,:]))
            print('#')
        print('#===========================================#')
        # Deleting the position and velocity information of the deleted cell
        positionsatlasttimestep = np.delete(positionsatlasttimestep, index, axis=1)
        velocitieslasttimestep = np.delete(velocitieslasttimestep, index, axis=1)
        new_position_cells = positionsatlasttimestep
        new_velocity_cells = velocitieslasttimestep

im[0].save('evolution.gif', save_all=True, append_images=im[1:], optimize=False, duration=30, loop=0)
print('The event with cell diameter of', cell_diameter, 'took', iteration, 'number of iterations to reach', positionsatlasttimestep.shape[1], 'cells.')
print('During the adaptation', howmanycellswerekilled, 'cells were killed by apoptosis.')