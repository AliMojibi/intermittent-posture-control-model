import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import preprocessor as pr

class Simulator:
    """
    A class for for implementation of the simulation of the intermittent posture control model.
    ...
    
    Attributes
    ----------
    h: (float)
        Distance from ankle to center of mass [m]. this is usually estimated as h = 0.58*height
    
    mass: (float)
        Mass of the subject [kg].
    
    P: (float)
        Proportional gain of controller.

    D: (float)
        Deriviative gain of the controller.

    delta: (float)
        Feedback delay of the model [sec.].

    sigma: (float)
        Noise gain of the model.

    r: (float)
        Raduis of the small off-controll-region around the origin in the phase plane.

    rho: (float)
        A float between 0.0 and 1 which is know as intermittency, and is calculated in the phase plane: rho = (AREA_ON/(AREA_ON + AREA_OFF))
    
    a_s: (float)
        slope of the line in the phase plane whcih a function of rho.

    K: (float)
        passive stiffness at the ankle joint

    I: (float)
        moment of inertia of the inverted pendulum model around the ankle joint.

    Class attributes
    ----------------
    g: (float)
        Gravitational acceleration

    B: (float)
        passive damping at the ankle joint

    Methods
    -------
    on_area:
        calculates the total AREA_ON in phase plane.
    
    _switch_functoin():
        calulates whether the switch is on or off.

    euler_maryuma_solver():
        Uses the Euler-Maryuma algorithm for solving the stochastic delayed differential equation of the system.
    """
    g = 9.81  # gravity
    B = 4.0  # passive damping at the ankle joint

    def __init__(self, h, mass, P, D, delta, sigma, r, rho):
        self.h = h
        self.mass = mass
        self.P = P
        self.D = D
        self.delta = delta
        self.sigma = sigma 
        self.r = r
        self.rho = rho
        self.a_s = -np.tan(np.pi * (self.rho - 0.5))
        self.K = 0.8 * self.mass * self.g * self.h
        self.I = self.mass * (self.h ** 2)

    def on_area(self):
        return 0.5 + np.arctan(-self.a_s) / np.pi  

    def _switch_function(self, th_del, th_dot_del):
        """
        Calculates whether the controller is on or off based on the delayed states of the systems.

        Args
        ----
            th_del (float): delayed theta
            th_dot_del (float): delayed theta dot

        Returns:
            int: 1 if the control is on, 0 otherwise.
        """
        if (th_del * (th_dot_del - self.a_s * th_del) > 0 and (th_del ** 2 + th_dot_del ** 2 > self.r ** 2)):
            return 1
        else:
            return 0
            
    def euler_maryuma_solver(self, t_final=70.0, dt=0.001, theta_0=0.01,
                             seed=None, return_angular_acc_no_noise=False):
        """
        This method uses Euler-MAryuma algorithem for solving the stochastic delayed differential equations (SSDE) of the model.
        ...

        Args
        ----
            t_final (float, optional): 
                Time duration for soving the SSDE. Defaults to 70.0.
            dt (float, optional): 
                Time step for solving the SDDE of the system. Defaults to 0.001.
            theta_0 (float, optional): 
                Initial value of theta (angle of the pendulum) in radian. Defaults to 0.01.
            seed (int, optional): 
                Seed for genrating random numbers. Defaults to None.
            return_angular_acc_no_noise (bool, optional): 
                If you want to recive the noise removed angular acceleration. 
                for definition and application of this parameter see the paper in the link (parmeter alpha tilde):
                https://pubmed.ncbi.nlm.nih.gov/33261318/ 
                Defaults to False.

        Returns
        -------
            (tuple): 
            if return_angular_acc_no_noise is True
                th, th_dot, th_ddot_no_noise, time, sw 
            else
                th, th_dot, time, sw
        """
        
        t_f = t_final + dt  # Final time of the soltion
        time = np.arange(start=0, stop=t_f, step=dt)
        th = np.zeros_like(time)  # Initializing the vector of the solution to the SSDE of the system for the first state variable (theta)
        th_dot = np.zeros_like(th)  # Initializing the vector of the solution to the SSDE of the system for the second state variable (theta dot)
        th[0] = theta_0  # Initial state of the system.

        # Coefficients for solving the SSDE
        a = (self.mass * self.g * self.h - self.K) / self.I
        b = -self.B / self.I
        c = -self.P / self.I
        d = -self.D / self.I
        e = self.sigma / self.I

        sqrtdt = np.sqrt(dt)  # Square root of the solution step   

        k = int(self.delta / dt)  # An index for inserting effect of delay in discrete equations
        
        if seed is not None:
            np.random.seed(seed)  # setting the random seed 
            print(f'random state set! : {seed}')

        XI = np.random.normal(0, 1, time.shape[0])  # Vector of the Noise
        
        sw = np.zeros_like(th)  # Initializing the swith values at each time step 

        # Euler-MAryuma algorithm for solving the SSDE   
        for i, _ in enumerate(time):
            if i != len(time) - 1:
                th[i + 1] = th[i] + th_dot[i] * dt
                if i - k < 0:
                    th_dot[i + 1] = th_dot[i] + (a * th[i] + b * th_dot[i]) * dt + e * XI[i] * sqrtdt
                else:
                    sw[i] = self._switch_function(th[i - k], th_dot[i - k])
                    th_dot[i + 1] = th_dot[i] + (
                            a * th[i] + b * th_dot[i] + c * th[i - k] * sw[i] + d * th_dot[i - k] * sw[i]) * dt + e * XI[i] * sqrtdt
                      
        
        if return_angular_acc_no_noise:

            # Delayed theta values
            th_delayed = np.roll(th, k)  
            th_delayed[0:k] = 0.0  

            # Delayed theta dot values
            th_dot_delayed = np.roll(th_dot, k)
            th_dot_delayed[0:k] = 0.0

            # Calculation of noise removed angular acceleration
            th_ddot_no_noise = a * th + b * th_dot + sw * (c * th_delayed + d * th_dot_delayed)

            return th, th_dot, th_ddot_no_noise, time, sw
        
        else:
            return th, th_dot, time, sw


def generate_observations(param_dict, m=60, h=1, n_trials=2, 
                          t_f=90.0, dt=0.001, dec_factor=10, 
                          th0=None, r_state=None, n_filter=4, 
                          f_cut=10.0):
    """ 
    Generate resampled observation signals for a posture control simulation.
    This function simulates posture control dynamics using the provided parameters
    and generates resampled signals for center of mass (COM) and its derivatives.
    Optionally, it can estimate COM from center of pressure (COP) data.
        param_dict (dict): Dictionary containing simulation parameters:
            - 'P' (float): Proportional gain.
            - 'D' (float): Derivative gain.
            - 'rho' (float): Control activation threshold.
            - 'r' (float): Noise intensity.
            - 'sigma' (float): Noise standard deviation.
            - 'delta' (float): Time delay.
        m (int, optional): Mass of the system. Defaults to 60.
        h (int, optional): Height of the system. Defaults to 1.
        n_trials (int, optional): Number of simulation trials. Defaults to 2.
        t_f (float, optional): Total simulation time in seconds. Defaults to 90.0.
        dt (float, optional): Time step for the simulation in seconds. Defaults to 0.001.
        dec_factor (int, optional): Decimation factor for resampling. Defaults to 10.
        plot_obs (bool, optional): Whether to plot the generated observations. Defaults to False.
        th0 (float, optional): Initial angle for the simulation. If None, a random value is used. Defaults to None.
        r_state (int, optional): Random seed for reproducibility. Defaults to None.
        n_filter (int, optional): Order of the low-pass Butterworth filter. Defaults to 4.
        f_cut (float, optional): Cutoff frequency for the low-pass filter in Hz. Defaults to 10.0.
        tuple: A tuple containing:
            - com_trials (ndarray): Resampled COM signals (in mm).
            - com_dots (ndarray): Resampled COM velocity signals (in mm/s).
            - com_ddots (ndarray): Resampled COM acceleration signals (in mm/s²).
            - sws (ndarray): Switching states from the simulation.
    """
    # Unpacking parameters from the dictionary
    P = param_dict['P']
    D = param_dict['D']
    rho = param_dict['rho']
    r = param_dict['r']
    sigma = param_dict['sigma']
    delta = param_dict['delta']
    
    # Creating the simulator object
    # The simulator object is created with the provided parameters.
    sim1 = Simulator(h=h,
                    mass=m,
                    P=P,
                    D=D,
                    delta=delta,
                    sigma=sigma,
                    r=r,
                    rho=rho)
    
    # Generating the time vector for the simulation
    out_trials = np.zeros((n_trials, 1 + int(t_f/(dec_factor * dt))))
    out_dot_trials = np.zeros_like(out_trials)
    out_ddot_trials = np.zeros_like(out_trials)
    
    # Generating the decimated (resampled) time vector for the simulation
    # The time vector is generated based on the final time, time step, and decimation factor.
    t_dec = np.arange(0, t_f + dt*dec_factor, dt*dec_factor)

    # Looping through the number of trials to generate observations
    for trial in range(n_trials):
        if th0 is None:
            # Generating a random initial angle if th0 is not provided
            th0 = np.random.uniform(-0.01, 0.01)
        # Running the simulation
        theta, _, th_ddot_no_noise, _, sws = sim1.euler_maryuma_solver(t_final=t_f, dt=dt, theta_0=th0,
                                                                         return_angular_acc_no_noise=True,
                                                                         seed=r_state)
        # Decimating the theta signal
        # The theta signal is decimated to reduce the sampling rate.
        out = signal.decimate(theta, dec_factor)
        # Decimating the angular acceleration signal
        out_ddot = signal.decimate(th_ddot_no_noise, dec_factor)

        nyq = 0.5 * ((1/dt)/10)  # Nyquist frequency
        normal_cutoff = f_cut / nyq
        # designing the lowpass filter
        b, a = signal.butter(n_filter, normal_cutoff, btype='low', analog=False)
        # Applying the lowpass filter to the decimated theta signal
        out = signal.filtfilt(b, a, out)

        # Calculating the derivatives of the decimated theta signal
        out_dot = np.gradient(out)/(dt*dec_factor)
        
        out_trials[trial, :] = out  # in radian unit
        out_dot_trials[trial, :] = out_dot
        out_ddot_trials[trial, :] = out_ddot   

        # Calculating the center of mass (COM) and its derivatives
        com_trials = 1000 * h * out_trials # in mm unit
        com_dots = 1000 * h * out_dot_trials
        com_ddots = 1000 * h * out_ddot_trials
        
    return com_trials, com_dots, com_ddots, sws