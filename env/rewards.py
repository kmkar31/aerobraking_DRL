import numpy as np

def get_reward(state, deltaV, target_apoapsis):

    if state['terminal']:
        distance_target = state['apoapsis_radius'] - target_apoapsis
        max_heat_rate = state['max_heat_rate']
        if distance_target > 100000 or max_heat_rate < 0.05:
            return -4
        elif max_heat_rate >= 0.25 and max_heat_rate < 0.3:
            return -4
        elif max_heat_rate >= 0.3 and max_heat_rate < 0.45:
            return -5
        elif max_heat_rate >= 0.45:
            return -6
        elif distance_target <= 10000:
            return -4
        elif np.abs(distance_target) <= 10000:
            return 10 -0.4*np.rint(distance_target/1000)
        else:
            raise Exception("Unrealized Case")
        
    else:
        normalized_heat_rate = (state['max_heat_rate'] - 0.05)/(0.25 - 0.05)
        distance_normalized = (state['apoapsis_radius'] - target_apoapsis)/(10038000 - target_apoapsis)

        reward = 0.25*(1 - np.float_power(distance_normalized, 0.4))*np.exp(-0.5*((normalized_heat_rate-0.5)/0.1)**2) - 0.1*(1 + deltaV)
        
        return reward



