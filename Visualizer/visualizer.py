import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

class Visualizer():
    def __init__(self, init_state):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(projection="3d")
        self.ax.set_xlim(-10000,10000)
        self.ax.set_ylim(-10000,10000)
        self.ax.set_zlim(-10000,10000) 
        
    
        # Mars
        self.central_body_radius = 3839.5 #km
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        self.xm = self.central_body_radius * np.outer(np.cos(u), np.sin(v))
        self.ym = self.central_body_radius * np.outer(np.sin(u), np.sin(v))
        self.zm = self.central_body_radius * np.outer(np.ones(np.size(u)), np.cos(v))
       
        # Orbit Definition
        self.orbit_elements = init_state # State Definition (Dictionary)
        self.orbit_path = dict()
        self.calculate_orbit_path() # Has X,Y,Z data

        # Plot the surface
        self.mars = self.ax.plot_surface(self.xm, self.ym, self.zm, color='r', animated=True)
        self.orbit, = self.ax.plot(self.orbit_path["x"], self.orbit_path["y"], self.orbit_path["z"], '--', animated=True)
        
        plt.show(block=False)
        self.background = self.figure.canvas.copy_from_bbox(self.ax.bbox)
        
    
    def calculate_orbit_path(self):
        theta = np.linspace(0, 2*np.pi, 100)
        e = (self.orbit_elements["apoapsis_radius"] - self.orbit_elements["periapsis_altitude"] - self.central_body_radius)\
            /(self.orbit_elements["apoapsis_radius"] + self.orbit_elements["periapsis_altitude"] + self.central_body_radius)
        
        # Planar Curves
        xp = (self.orbit_elements["apoapsis_radius"]/(1 + e))*(np.cos(theta)-e)
        yp = (self.orbit_elements["apoapsis_radius"]/(1 + e))*np.sqrt(1-e**2)*np.sin(theta)
        zp = np.zeros_like(xp)
        X = np.stack((xp, yp, zp))

        # Rotate by Inclination angle
        i = self.orbit_elements["inclination"]
        RAAN = self.orbit_elements["RAAN"]
        AOP = self.orbit_elements["AOP"]
        
        M_raan = np.reshape(np.array([[np.cos((RAAN*np.pi/180)), np.sin(RAAN*np.pi/180), 0],[-np.sin(RAAN*np.pi/180), np.cos(RAAN*np.pi/180), 0],[0,0,1]]), (3,3))
        M_i = np.reshape(np.array([[1,0,0],[0,np.cos((i*np.pi/180)),np.sin((i*np.pi/180))],[0, -np.sin(i*np.pi/180), np.cos(i*np.pi/180)]]), (3,3))
        M_aop = np.reshape(np.array([[np.cos((AOP*np.pi/180)),np.sin((AOP*np.pi/180)),0],[-np.sin(i*np.pi/180), np.cos(i*np.pi/180),0], [0,0,1]]), (3,3))
        
        Xt = M_aop@M_i@M_raan@X
        self.orbit_path["x"] = Xt[0,:]
        self.orbit_path["y"] = Xt[1,:]
        self.orbit_path["z"] = Xt[2,:]
    
    def update_orbit(self,state):
        self.orbit_elements = state
        self.calculate_orbit_path()
        self.figure.canvas.draw()
        self.plot()
        self.figure.canvas.flush_events()

    def plot(self):
        self.orbit.set_data_3d(self.orbit_path["x"],self.orbit_path["y"],self.orbit_path["z"])
        self.ax.draw_artist(self.orbit)
        self.ax.draw_artist(self.mars)
        self.figure.canvas.blit(self.ax.bbox)