import subprocess
import os
from rewards import get_reward
import numpy as np
import re
import datetime
# Need to modify ABTS.py to return solution as a json dump which is then read here --> Only way to access state while still using subprocess

class Environment():
    def __init__(self, params):
        '''
        Write all the parser arguments to ABTS.py here
        '''

        # This contains all the user-set parameters
        # We should be using this to map action indices to action and to reset env
        self.params = params
        self.target_apoapsis_radius = self.params["environment"]["apoapsis_radius_target"]

        self._call = ["python3", os.getcwd() + "/Simulator/ABTS/ABTS.py"] # Function to be executed
         # Write all runtime independent args here. Also call populate_init_state to add initial OE
        self._args = ["--results", str(0),
                      "--print_res", str(1),
                      "--plot", str(0),
                      "--type_of_mission", "Orbits",
                      "--number_of_orbits", str(1),
                      "--machine", "Laptop",
                      "--thrust_control", "Aerobraking Maneuver",
                      "--montecarlo_analysis", str(0),
                      "--year", str(2001),
                      "--month", str(12),
                      "--final_apoapsis_radius", str(self.params["environment"]["apoapsis_radius_target"]),
                      "--density_model", "Exponential"]
        self.state = None
        _ = self.populate_call_arguments(0)

        

    def step(self, action_idx):
        '''
        Return next_state, observation and terminal
        '''
        deltaV = self.params["setup"]["actions"][int(action_idx)]
        print(deltaV)
        self.run_args = self._call + self._args + self.populate_call_arguments(deltaV) # This should only be the deltaV argument
            
        # Note : phi = 180 if action value is neg else 0
        print("Starting")
        try :
            with subprocess.Popen(self.run_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True) as process:
                log = process.communicate(timeout=200)[0]
            data = re.split(': |\n|\r', ''.join(log.decode('utf-8')))[1:-1]
            #print(data)
            data = dict(zip(data[1::3], map(float, data[2::3])))
            data["year"] = int(data["year"])
            data["month"] = int(data["month"])
            data["day"] = int(data["day"])
            data["hour"] =int(data["hour"])
            data["min"] = int(data["min"])
            data["second"] = int(data["second"])

            self.state = data
        except subprocess.CalledProcessError as e:
            print(e.stderr)
        
        if "target_apoapsis_radius" not in self.state.keys():
            self.state["target_apoapsis_radius"] = self.target_apoapsis_radius
        
        self.is_terminal()
        reward = get_reward(self.state, deltaV, self.target_apoapsis_radius)
        #print(self.state, reward)
        return (reward, self.mask(), self.state["terminal"])

    def reset(self):
        '''
        Call populate_init_state to get new init_state
        '''
        self.state = None
        _ = self.populate_call_arguments(0)
        return self.mask()
    
    def get_observation(self):
        return self.mask()

    def mask(self):
        '''
        Create a list from the state dictionary that only return the partial state
        According to the paper, all states are normalized to 0 --> 1
        '''
        obs = []
        date = datetime.datetime(self.state["year"], self.state["month"], self.state["day"],\
                self.state["hour"], self.state["min"], self.state["second"]).timestamp()
        obs.append((date - datetime.datetime(2001,1,1,0,0,1).timestamp())/datetime.datetime(2003,1,1,0,0,1).timestamp())
        obs.append((self.state["passage_time"]-400)/1200)
        obs.append((self.state["apoapsis_radius"]-self.target_apoapsis_radius)/(10038000 - self.target_apoapsis_radius))
        obs.append((self.state["periapsis_altitude"] - 85000)/(94000 - 85000))
        obs.append((self.state["inclination"] - 79)/(100-79))
        obs.append((self.state["AOP"])/360)
        obs.append((self.state["RAAN"]-90)/(180-90))
        obs.append((self.state["max_heat_rate"]-0.05)/(0.25-0.05))
        obs.append((self.state["max_air_density"])/5e-8)
        
        return obs

    def is_terminal(self):
        '''
        Check if a state is the terminal state or not
        '''
        if np.abs(self.state["apoapsis_radius"] - self.target_apoapsis_radius) <= self.params["termination"]["apoapsis_radius_tolerance"] \
            or self.state["max_heat_rate"] <= self.params["termination"]["low_heat_rate"] \
            or self.state["max_heat_rate"] >= self.params["termination"]["high_heat_rate"] \
            or self.state["periapsis_altitude"] <= self.params["termination"]["low_altitude"] \
            or self.state["periapsis_altitude"] >= self.params["termination"]["high_altitude"] :
                self.state["terminal"] = 1
        else:
            self.state["terminal"] = 0

    def populate_call_arguments(self, deltaV):
        '''
        Use the ranges given in the paper to generate an initial state
        '''
        if self.state is None:
            self.state = dict()
            self.state["apoapsis_radius"] = np.random.uniform(self.params["environment"]["apoapsis_radius_initial"]-self.params["environment"]["apoapsis_radius_dispersion"], \
                                                            self.params["environment"]["apoapsis_radius_initial"]+self.params["environment"]["apoapsis_radius_dispersion"])
            self.state["periapsis_altitude"] = np.random.uniform(self.params["environment"]["periapsis_altitude_initial"]-self.params["environment"]["periapsis_altitude_dispersion"], \
                                                            self.params["environment"]["periapsis_altitude_initial"]+self.params["environment"]["periapsis_altitude_dispersion"])
            self.state["inclination"] = np.random.uniform(self.params["environment"]["inclination_low"], self.params["environment"]["inclination_high"])
            self.state["AOP"] = np.random.uniform(self.params["environment"]["aop_low"], self.params["environment"]["aop_high"])
            self.state["RAAN"] = np.random.uniform(self.params["environment"]["raan_low"], self.params["environment"]["raan_high"])
            self.state["year"] = 2001
            self.state["month"] = 12
            self.state["day"] = np.random.randint(1, 31)
            self.state["hour"] = np.random.randint(0, 23)
            self.state["min"] = np.random.randint(0, 59)
            self.state["second"] = np.random.randint(0, 59)
            self.state["passage_time"] = 0
            self.state["max_air_density"] = 0
            self.state["max_heat_rate"] = 0
            
        run_args = ["--ra_initial_a", str(self.state["apoapsis_radius"]) ,
                        "--hp_initial_a", str(self.state["periapsis_altitude"]),
                        "--inclination", str(self.state["inclination"]),
                        "--omega", str(self.state["AOP"]),
                        "--OMEGA", str(self.state["RAAN"]),
                        "--day", str(self.state["day"]),
                        "--hours", str(self.state["hour"]),
                        "--minutes", str(self.state["min"]),
                        "--secs", str(self.state["second"]),
                        "--delta_v", str(np.abs(deltaV)),
                        "--phi", str(180) if np.sign(deltaV) > 0 else str(0)
                    ]
        return run_args

