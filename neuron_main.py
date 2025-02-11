

import numpy as np
import matplotlib.pyplot as plt
import neuron_model as nm
import neuron_Ifunctors as nI


def count_spikes(V, Vspike):
    """
    Count the number of spikes occurring in the neuron voltage time series
    given in V.

    Args:
        V (numpy array): neuron voltage values for each timestep
        Vspike (float): voltage maximum needed for a spike

    Returns:
        n (int): number of spikes in V
    """
    #### BEGIN SOLUTION #####
    n = 0
    is_spiking = False

    for i in range(1, len(V)):
        if V[i] > Vspike and V[i-1] < Vspike:  #Check if voltage exceeds spike threshold
            is_spiking = True
        elif is_spiking and V[i] < 0:  #Check if voltage crosses back through zero
            n += 1
            is_spiking = False

    return n
    #### END SOLUTION ####


def test_count_spikes():
    """Perform a test of count_spikes."""
    print("Testing count_spikes:")
    V = np.array([0., 11., 9., 11., 9., 11.0, 1.0, 0.0, -1.0, 0., 10.5, 1.0, -1.0, 0.0])
    nSpikes = count_spikes(V, 10.0)
    print(f"   nSpikes = {nSpikes}")
    if nSpikes == 2:
        print("   Result is correct!")
    else:
        print("   Result is incorrect: answer should be 2")


def single_run(Iamp_run):
    """
    This function performs a single simulation of a neuron
    by calling solve_neuron in the neuron_model module.

    Then, the function will:

    * Plot the V, W solution behavior overlayed on the same graph, and it must:
        * include a legend
        * V and W must be plotted in units of mV
        * include a grid
        * the voltage axis should have a range from -30 to 100 mV and be labeled: mV
        * the time axis should have a range from 0 to tF and be labeled: t (ms)
        * the title of the plot should give the current for the run in nA

      See the assignment for an example of what the plot should look like.

    * Determine the number of spikes that occurred in the simulation
      by calling the count_spikes method.

    Args:
        Iamp_run (float): the current to run the simulation withq

    Returns:
        fig (Figure): the Figure object containing the plot
        axs (Axes): the Axes object on which the plot is drawn
        nSpikes (int): the number of spikes in V throughout the simulation
        fSpike (float): the average frequency of the spikes
    """
    # Set up dictionary for neuron model
    p = {}
    p['Vs']    = 0.25
    p['tauV']  = 5e-2
    p['tauW']  = 1e1
    p['alpha'] = 1.25
    p['I'] = nI.Iconst(Iamp=Iamp_run)

    # Set up initial condition, final time, and timestep
    VI = 0
    WI = 0
    uI = np.array([VI, WI])
    tF = 1e2  # final time to simulate to (ms)
    dt = 1e-1  # time increment to give solutions at (ms)

    # Solve neuron model
    t, V, W = nm.solve_neuron(uI, tF, dt, p)

    # Perform plotting
    fig, axs = plt.subplots()
    #### BEGIN SOLUTION #####
    axs.plot(t, V * 1e2, label='V')
    axs.plot(t, W * 1e2, label='W')
    axs.legend()
    axs.set_xlabel('t (ms)')
    axs.set_ylabel('mV')
    axs.set_title(f'Iamp = {Iamp_run:.3f} nA')
    axs.grid(True)
    axs.set_xlim([0, tF])
    axs.set_ylim([-30, 100])
    #### END SOLUTION ####

    # Call count_spikes and determine spike frequency
    nSpikes = count_spikes(V*1e2, 80)  #count spikes in V with threshold 80 mV
    print(f"Number of spikes = {nSpikes}")
    if nSpikes > 0:
        fSpike = nSpikes / tF * 1e3  # Calculate spike frequency in Hz
    else:
        fSpike = 0.0
    print(f"Spike frequency = {fSpike:.1f} Hz")

    # Return figure and axes
    return fig, axs, nSpikes, fSpike


def test_Iconst_noise():
    """
    Perform a test of the Iconst_noise functor in neuron_Ifunctors.py.

    The basis of the test is to call the functor for a large number of times
    and check to see if the mean and variance of the call are within the
    expected range.
    """
    N = int(1e6)
    I = np.zeros(N)
    Iamp_run = 100.0
    Isig_run = 4.0
    Ivar_run = Isig_run**2

    Imean_stderr = Isig_run / np.sqrt(N)
    Imean_lo = Iamp_run - 3 * Imean_stderr
    Imean_hi = Iamp_run + 3 * Imean_stderr

    Ivar_stderr = Ivar_run * np.sqrt(2 / (N-1))
    Ivar_lo = Ivar_run - 3 * Ivar_stderr
    Ivar_hi = Ivar_run + 3 * Ivar_stderr

    Ifunctor = nI.Iconst_noise(Iamp=Iamp_run, Isig=Isig_run)

    for i in range(N):
        I[i] = Ifunctor(i)

    Imean = I.mean()
    Ivar = I.var(ddof=1)

    print("Testing if observed mean of Iconst_noise is within 99.7% confidence")
    print(f"Confidence interval = [{Imean_lo:.5e}, {Imean_hi:.5e}]")
    print(f"    Observed mean I = {Imean:.5e}")
    if Imean_lo < Imean < Imean_hi:
        print("PASSED TEST")
    else:
        print("FAILED TEST: please debug your Iconst_noise implementation")
    print()

    print("Testing if observed variance of Iconst_noise is within 99.7% confidence")
    print(f"Confidence interval = [{Ivar_lo:.5e}, {Ivar_hi:.5e}]")
    print(f"     Observed var I = {Ivar:.5e}")
    if Ivar_lo < Ivar < Ivar_hi:
        print("PASSED TEST")
    else:
        print("FAILED TEST: please debug your Iconst_noise implementation")
    print()


def MC_run(Iamp_run, Isig_run, nruns):
    """
    Set up and run a Monte Carlo simulation of the neuron model.

    Specifically, the function will:

    * Run a Monte Carlo simulation with nruns samples.

    * Determine and print (with nice formatting):
        * the min spike frequency
        * the max spike frequency
        * the mean spike frequency
        * the 95% confidence interval for the mean spike frequency

    * Plot the histogram of the spike frequency.  See the assignment
      for an example of what the plot should look like.

    Args:
        Iamp_run (float): the mean current
        Isig_run (float): the standard deviation of the current
        nruns (int): the number of samples in the Monte Carlo run

    Returns:
        stats (tuple): statistics on the sampled spike frequencies, in the form:
            ((min, max), mean, (CI_low, CI_high))
        fig (Figure): the Figure object containing the histogram
        axs (Axes): the Axes object on which the histogram is drawn
        histbins (tuple): the ENTIRE tuple output of the call to axs.hist()
    """
    #### BEGIN SOLUTION #####
    spike_freqs = [] 
    #dictionary for neuron model
    p = {}
    p['Vs']    = 0.25
    p['tauV']  = 5e-2
    p['tauW']  = 1e1
    p['alpha'] = 1.25
    p['I'] = nI.Iconst_noise(Iamp=Iamp_run, Isig=Isig_run)

    # Set up initial condition, final time, and timestep
    VI = 0
    WI = 0
    uI = np.array([VI, WI])
    tF = 1e3  # final time to simulate to (ms)
    dt = 1e-1  # time increment to give solutions at (ms)
    # Solve neuron model

    for i in range(nruns):
       
        t, V, W = nm.solve_neuron(uI, tF, dt, p)
        nSpikes = count_spikes(V*1e2, 80)  #count spikes in V with threshold 80 mV
        print(f"Number of spikes = {nSpikes}")
        if nSpikes > 0:
            fSpike = nSpikes / tF*1e3  # Calculate spike frequency in Hz
        else:
            fSpike = 0.0
        print(f"Spike frequency = {fSpike:.1f} Hz")
        spike_freqs.append(fSpike)

    # Return figure and axes
    #calculate statistics
    spike_freq_min = np.min(spike_freqs)
    spike_freq_max = np.max(spike_freqs)
    spike_freq_mean = np.mean(spike_freqs)
    spike_freq_std = np.std(spike_freqs)
    confidence_interval = (spike_freq_mean - 2 * spike_freq_std, spike_freq_mean + 2 * spike_freq_std)

    # Plot histogram
    fig, axs = plt.subplots()
    histbins = axs.hist(spike_freqs, bins='auto', color='skyblue', edgecolor='black')

    # Formatting and labels
    axs.set_title(f"Iamp = {Iamp_run:.3f} nA, Isig = {Isig_run:.3f} nA")
    axs.set_xlabel('fSpike (Hz)')
    axs.set_ylabel('count')

    # Return statistics and plot objects
    return ((spike_freq_min, spike_freq_max), spike_freq_mean, confidence_interval), fig, axs, histbins
    #### END SOLUTION ####

if __name__ == "__main__":
    pass
    # Uncomment below what you would like to run

    single_run(Iamp_run=0.556)
    single_run(Iamp_run=0.556)
    single_run(Iamp_run=3.300)
    single_run(Iamp_run=3.301)
    #test_count_spikes()
    #test_Iconst_noise()
    MC_run(Iamp_run=3.300, Isig_run=0.1, nruns=100)


    # Keep this line uncommented to display plots on your screen
    plt.show()
