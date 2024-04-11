import subprocess
# def call_ABTS(state):
import numpy as np

from subprocess import Popen, PIPE
import time
import sys

t = time.time()
print("200 Iterations")
with open(str(t)+'.csv','w+') as f:
  sys.stdout = f
  for i in range(50):
      ra = np.random.uniform(4904000, 11000000)
      hp = np.random.uniform(85000, 135000)
      incl = np.random.uniform(88.6, 98.6)
      raan = np.random.uniform(110, 120)
      aop = np.random.uniform(70, 90)
      V = [-1.0, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1.0]
      deltaV = V[np.random.randint(0,11)]
      print(ra, hp, incl, raan, aop, deltaV)
      subprocess.call(["python3", "ABTS/ABTS.py", '--machine', 'Laptop','--integrator','Python', '--type_of_mission', 'Orbits','--number_of_orbits',str(1),
                  '--control_mode',str(0),'--gravity_model','Inverse Squared',
                  '--hp_initial_a', str(hp),'--density_model', 'Exponential',
                  '--ra_initial_a', str(ra),'--year', str(2001),'--month', str(12), '--day',str(14),'--hours', str(14),'--minutes', str(21), '--secs',str(28),
                  '--hp_step',str(5000000000),'--ra_step',str(100000000),'--MarsGram_version',str(0), '--max_heat_rate', str(0.45),'--max_heat_load', str(30),
                '--aop_dispersion_gnc', str(0), '--vi_dispersion_gnc', str(0),'--final_apoapsis',str(4906000),'--flash2_through_integration',str(0),'--flash1_rate',str(3),
                  '--control_in_loop',str(0),'--second_switch_reevaluation',str(1),'--security_mode',str(0), '--plot', str(0), '--inclination', str(incl), 
                  '--OMEGA', str(raan), '--omega', str(aop), '--delta_v', str(deltaV)], stdout=f)
