import numpy as np
import preprocessor as pr
from posturesim import Simulator
from pyabc import ABCSMC, RV, Distribution
from scipy.stats import entropy
from datetime import timedelta
from scipy import  signal
from pyabc.sampler import MulticoreEvalParallelSampler
from functools import partial

def intermittent_model(params,
                       h,
                       mass,
                       bins_com,
                       bins_com_dot,
                       bins_com_ddot,
                       summary_maker):
    
    """
        Simulates an intermittent posture control model and generates summary statistics.
        Parameters:
        -----------
        params : dict
            Dictionary containing model parameters:
            - 'P' (float): Proportional gain.
            - 'D' (float): Derivative gain.
            - 'rho' (float): Control activation threshold.
            - 'r' (float): Control deactivation threshold.
            - 'sigma' (float): Noise intensity.
            - 'delta' (float): Time delay.
        h : float
            Height of the center of mass (COM) in meters.
        mass : float
            Mass of the system in kilograms.
        bins_com : array-like
            Bins for discretizing the center of mass position.
        bins_com_dot : array-like
            Bins for discretizing the center of mass velocity.
        bins_com_ddot : array-like
            Bins for discretizing the center of mass acceleration.
        summary_maker : object
            An object with a method `creat_summary_stat_vec` for generating summary statistics.
        Returns:
        --------
        dict
            A dictionary containing:
            - 'stat_vec' (array-like): Summary statistics vector derived from the simulated data.
        Notes:
        ------
        - The simulation uses an Euler-Maruyama solver to integrate the dynamics.
        - The first 20 seconds of the simulation are ignored, and the remaining 70 seconds are used for inference.
        - The position and acceleration data are detrended before generating summary statistics.
    """
    # Unpack parameters
    P = params['P']
    D = params['D']
    rho = params['rho']
    r = params['r']
    sigma = params['sigma']
    delta = params['delta']
    
    # Generate random initial angle
    th0 = np.random.uniform(-0.01, 0.01)

    # Creating the instance of the simulator
    simulator = Simulator(h, mass, P, D, delta, sigma, r, rho)
    
    # Simulate the system, note that we set the `return_angular_acc_no_noise` to True
    theta, theta_dot, theta_ddot, t, noise_ve = simulator.euler_maryuma_solver(dt=0.001, t_final=80, theta_0=th0, return_angular_acc_no_noise=True)
    
    # Decimate the data
    theta = signal.decimate(theta, 10)

    # Converting theta to CoM
    x  = 1000 * h * theta[20*100: ] # the data of first 20 second is ignored and the remaining 70 sec is used for inference
    
    # Decimating the angular acceleration
    theta_ddot_dec = signal.decimate(theta_ddot, 10)
    acc = 1000 * h * theta_ddot_dec[20*100:]
    
    # Detrending the data
    x = signal.detrend(x, type='linear')
    acc = signal.detrend(acc, type='linear')
    
    # Calculating the summary statistics
    # Note that we set the `normalize_values` to True
    stat_vec, _ = summary_maker.creat_summary_stat_vec(x, 
                                                    x_ddot=acc,
                                                    x_bins=bins_com,
                                                    x_dot_bins=bins_com_dot,
                                                    x_ddot_bins=bins_com_ddot,
                                                    normalize_values=True)
    
    # Returning the summary statistics vector
    return {'stat_vec': stat_vec}

def jensen_shannon_divergence(p, q):
    # Calculate the Jensen-Shannon divergence between two distributions
    # p and q are the summary statistics vectors
    m = 0.5 * (p['stat_vec'] + q['stat_vec'])
    return 0.5 * (entropy(p['stat_vec'], m, base=2) + entropy(q['stat_vec'], m, base=2))

#==================Running the inference========================

# Setting the parameters
subject_ids = list(range(18))
f_sampling = 100.0  # Hz

# You must set theses values for each subject:
g = 9.81
task_type = "OR" # "OR" for Eyes open and Rigid Surface tasks

# Loading data
path_to_file = f'healthy_cop_data_{task_type}.npy'
cops = np.load(path_to_file) # mm

# Loading Demographic data
demo = np.loadtxt('DemographicH.txt',delimiter='\t')
trials = 2

# Converting COP to COM
coms = np.zeros_like(cops[0, :, :])
h_e = 1.0 
prc = pr.CopPreprocessor(fs=f_sampling, cutoff_freq=10.0, order=4, omega_0=np.sqrt(g/h_e))
summarizer = pr.Summarystats(fs=f_sampling)

for subject_id in subject_ids:
    print(f"{'='*30} subject {subject_id} started {'='*30}")
    subject_cop = np.copy(cops[subject_id, :, :]) 
    # Experimental COP preprocessing
    # converting COP to COM
    for i, sig in enumerate(subject_cop):
        coms[i, :] = prc.convert_to_com(subject_cop[i, :])
    # detrending the data
    coms_det = signal.detrend(coms, type='linear', axis=1)
    com_dots_det = np.gradient(coms, axis=1) * f_sampling
    com_ddots_det = np.gradient(com_dots_det, axis=1) * f_sampling
    com_ddots_det = signal.detrend(com_ddots_det, type='linear', axis=1)
    # histogram of the data
    bins_com  = summarizer.observation_bins(coms_det)
    bins_com_dot  = summarizer.observation_bins(com_dots_det)
    bins_com_ddot  = summarizer.observation_bins(com_ddots_det)
    # demographic data of height and mass
    height, m = demo[subject_id, :]
    h = 0.58 * height # in meters
    
    # Creating a partial function for the model
    my_model = partial(intermittent_model,
                    h=h,
                    mass=m,
                    bins_com=bins_com,
                    bins_com_dot=bins_com_dot,
                    bins_com_ddot=bins_com_ddot,
                    summary_maker=summarizer)

    # initializing the summary statistics vector to zeros
    summaries = np.zeros((trials, 75))

    # calculating the summary statistics for each trial
    for i in range(trials):
        # here normalize values is set to off in order to get the values and calculation of the mean
        summaries[i, :], ff = summarizer.creat_summary_stat_vec(coms_det[i, :], 
                                        x_ddot=com_ddots_det[i, :],
                                        x_bins=bins_com,
                                        x_dot_bins=bins_com_dot,
                                        x_ddot_bins=bins_com_ddot,
                                        normalize_values=False)
        
    # creating the summary statistics vector
    n1, n2 = 15, 10
    ss_vec = summaries.mean(axis=0) # mean of the summary statistics for trials

    # normalizing the mean summary statistics vector
    sizes = [n1, n1, n1, n2, n2, n2]
    start = 0
    for size in sizes:
        ss_vec[start:start + size] = ss_vec[start:start + size] / ss_vec[start:start + size].sum() if ss_vec[start:start + size].sum() else ss_vec[start:start + size]
        start += size

    observation_data = ss_vec  # observed summary statistics vector from the experimental data
    print(observation_data.sum())  # this sum must be 1
    # Infrence form the summary statistics of the Experimental data
    # Setting the prior distributions for the parameters
    # Note that the parameters are set to be uniform distribution
    prior = Distribution(P=RV("uniform", 50, 350),
                        D=RV("uniform", 0, 300),
                        delta=RV("uniform", 0.05, 0.45),
                        sigma=RV("uniform", 0.05, 0.55),
                        r=RV("uniform", 0, 0.01),
                        c_on=RV("uniform", 0.3, 0.7))
    # Setting the ABCSMC
    abcsmc = ABCSMC(models=my_model,
                    parameter_priors=prior,
                    distance_function=jensen_shannon_divergence,
                    population_size=500,
                    sampler=MulticoreEvalParallelSampler())
    
    # Setting the database path
    # db_path = '"
    # print(f'path to the data base : {db_path} \n')

    # Setting the observed summary statistics 
    observed_ss = {'stat_vec': observation_data}
    
    abcsmc.new("sqlite:///" + db_path, observed_ss)  # creating the database set the db_path variable in line 198

    # Running the ABCSMC with the minimum epsilon of 0.01 and maximum wall time of 2 hours and 30 minutes
    history = abcsmc.run(minimum_epsilon=0.01, max_walltime=timedelta(hours=2,minutes=30))



