#!/usr/bin/env python3

import brian2
from brian2 import NeuronGroup, SpikeMonitor, StateMonitor, defaultclock
from brian2 import mV, pA, ms, second, Hz, pF, Gohm
import matplotlib.pyplot as plt
import numpy as np

# Create a network
net = brian2.Network() # explicitly handles all devices, compilation, and running

# create a neuron with the given parameters:
u_rest = -65 * mV
Rm = 0.05 * Gohm
Cm = 400 * pF
I = 500 * pA
tau = Rm * Cm

# Differential equation:
eqs = '''
du/dt = -u/tau + u_rest/tau + I_inj*Rm/tau : volt (unless refractory)
I_inj : amp
'''

# Limits: 
u_thresh = -50 * mV
u_reset = -80 * mV
t_ref = 2* ms

# Currents:
I_values = np.arange(0, 1000, 10) * pA

# Create neurongroup:
neuron = NeuronGroup(len(I_values), eqs, threshold='u >= u_thresh', 
                     reset='u = u_reset', method='exact', refractory=t_ref)

# Set starting value to u_reset:
neuron.u = u_reset

# Set current:
neuron.I_inj = I_values

# Setup state and spike monitor:
state_mon = StateMonitor(neuron, ['u'], record=True)
spike_mon = SpikeMonitor(neuron)

# Add neurons to the net:
net.add([neuron,state_mon,spike_mon])

# Set simulation time and run the model:
#defaultclock.dt = 0.01*ms
t_sim = 5000 * ms
net.run(t_sim)

# Store the values:
t_ = state_mon.t
u_ = state_mon.u
s_ = spike_mon.spike_trains()

# Calculate frequencies from the simulation:
f_experiment = []
for i in range(len(I_values)):
    f_experiment.append(1/np.mean(np.diff(s_[i])))
f_experiment = np.array(f_experiment)
f_experiment = f_experiment
np.nan_to_num(f_experiment, copy=False)

# Calculate theoretical frequencies (Equation from task 1a):
f_theory = (-tau * np.log((u_thresh - u_rest - I_values*Rm) /
                          (u_reset-u_rest-I_values*Rm)) + t_ref) ** -1
f_theory = f_theory / Hz
np.nan_to_num(f_theory, copy=False)

# Plots:

# f-I-curve:
plt.figure()
plt.plot(I_values/pA, f_theory, label='theory')
plt.plot(I_values/pA, f_experiment, label='simulation')
plt.legend(loc='best')
plt.grid(b=True)
plt.ylabel('$f(I)$ / Hz')
plt.xlabel('$I$ / pA')
plt.title('f-I-curve')
plt.xlim([I_values[0]/pA, I_values[-1]/pA])
plt.savefig('img/fI_curve.pdf', format='pdf')

# f-I-curve zoomed in:
plt.figure()
plt.plot(I_values[75:-1]/pA, f_theory[75:-1], label='theory')
plt.plot(I_values[75:-1]/pA, f_experiment[75:-1], label='simulation')
plt.legend(loc='best')
plt.grid(b=True)
plt.ylabel('$f(I)$ / Hz')
plt.xlabel('$I$ / pA')
plt.title('f-I-curve (zoomed in)')
plt.savefig('img/fI_curve_zoomed.pdf', format='pdf')

# Plot traces:
plt.figure()
plt.plot(t_[0:1000] / ms, u_[25,0:1000] / mV, 
         label=r'$I_\mathrm{{injected}} = {}$ pA'.format(int(I_values[25]/pA)))
plt.plot(t_[0:1000] / ms, u_[50,0:1000] / mV, 
         label=r'$I_\mathrm{{injected}} = {}$ pA'.format(int(I_values[50]/pA)))
plt.plot(t_[0:1000] / ms, u_[90,0:1000] / mV, 
         label=r'$I_\mathrm{{injected}} = {}$ pA'.format(int(I_values[90]/pA)))
plt.legend(bbox_to_anchor=(1, -0.15), ncol=3, handlelength=1)
plt.ylabel('$u(t)$ / mV')
plt.xlabel('$t$ / ms')
plt.xlim([0, 100])
plt.grid(b=True)
plt.title('Traces of the neuron with different injected currents')
plt.savefig('img/traces.pdf', format='pdf', bbox_inches='tight')


# Zoomed traces:
plt.figure()
plt.plot(t_[0:1000] / ms, u_[50,0:1000] / mV, 
         label=r'$I_\mathrm{{injected}} = {}$ pA'.format(int(I_values[50]/pA)))
plt.plot(t_[0:1000] / ms, u_[90,0:1000] / mV, 
         label=r'$I_\mathrm{{injected}} = {}$ pA'.format(int(I_values[90]/pA)))
plt.legend(loc='best')
plt.ylabel('$u(t)$ / mV')
plt.xlabel('$t$ / ms')
plt.xlim([0, 100])
plt.ylim([u_thresh/mV-1, u_thresh/mV+1])
plt.grid(b=True)
plt.title('Peaks of the neuron with different injected currents')
plt.savefig('img/traces_zoomed.pdf', format='pdf', bbox_inches='tight')