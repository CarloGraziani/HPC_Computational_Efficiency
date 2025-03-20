import numpy as np

def per_checkpt_cost(jobs, tau_c):
    """
    Cost of a single checkpoint
    Parameters:
    
    jobs (ndarray[:,2]): List of jobs. jobs[:,0] is jobsize, jobs[:,1] is job duration
    
    tau_c (float): Time required for a single checkpoint
    
    Returns: costs of checkpoint for all jobs, in node-hrs, using Eq.(2.2) of CE
    
    """
    
    cost = tau_c * jobs[:,0]**2
    return cost

def per_restart_cost(jobs, u_chk, tau_0, R_0):
    """
    Cost of a single restart
    Parameters:
    
    jobs (ndarray[:,2]): List of jobs. jobs[:,0] is jobsize, jobs[:,1] is job duration
    
    u_chk (ndarray[:]): List of checkpoint costs corresponding to jobs[:]
    
    tau_0 (float): fixed setup time for restarting a job
    
    Returns: Cost of a single restart for all jobs, in node-hrs, using Eq. (2.4) of CE
    
    """
    
    u_r = u_chk + tau_0 * jobs[:,0]
    cost = (np.exp(R_0*u_r)-1) / R_0
    return cost

def per_failure_cost(u_c, u_chk, R_0):
    """
    Cost of a single failure
    Parameters:
    
    u_c (ndarray[:]): List of checkpoint intervals (node-hrs) corresponding to a set of jobs
    
    u_chk (ndarray[:]): List of checkpoint costs corresponding to a set of jobs
    
    R_0 (float): Failure rate (failures/node-hr)
    
    Returns: Cost of a single failure for all jobs (node-hrs) using Eq. (2.3) of CE
    
    """
    
    z_tot = R_0 * (u_c + u_chk)
    cost = (1/R_0) * ( 1 - z_tot * np.exp(-z_tot)/( 1 - np.exp(-z_tot) ) )
    return cost

def optimum_checkpoint_cadence(jobs, R_0, u_chk, tol=1.0E-08, maxiter=100):
    """
    Optimum checkpointing cadence
    Parameters:
    
    jobs (ndarray[:,2]): List of jobs. jobs[:,0] is jobsize, jobs[:,1] is job duration
    
    u_chk (ndarray[:]): List of checkpoint costs corresponding to jobs[:]
    
    R_0 (float): Failure rate (failures/node-hr)
    
    tol (float): Tolerance for root find
    
    maxiter (int): Maximum iterations before concluding something has gone wrong
    
    Returns: Optimum checkpointing cadences for all jobs, in node-hrs, using bisection root finding to
             solve Eq. (3.3) of CE for u_c
    
    """
    
    def g(z_c, z_chk):
        res = np.exp(-z_c-z_chk) - (1-z_c)
        return res
    
    
    n_jobs = jobs.shape[0]

# Bisection root
    z_chk = R_0 * u_chk
    z_lo = np.zeros(n_jobs)
    z_hi = np.ones(n_jobs)
    g_lo = g(z_lo, z_chk)
    g_hi = g(z_hi, z_chk)
    assert(np.all(g_lo * g_hi <0))
    err = 2 * tol
    iter = 0
    while (err > tol):
        assert(iter < maxiter)
        z_mid = 0.5 * (z_hi + z_lo)
        g_mid = g(z_mid, z_chk)
        move_lo = g_lo * g_mid > 0
        move_hi = np.logical_not(move_lo)
        z_lo = np.where(move_lo, z_mid, z_lo)
        z_hi = np.where(move_hi, z_mid, z_hi)
        err = (np.abs( (z_hi - z_lo)/z_mid )).max()
        iter += 1
        
    u_c = z_mid / R_0
    return u_c

def get_N_chk(jobs, u_c, u_chk, continuous=False):
    """
    Return the N_chk parameter, which is the number of checkpoints for a successfully-terminating
    job.
    Parameters:
    
    jobs (ndarray[:,2]): List of jobs. jobs[:,0] is jobsize, jobs[:,1] is job duration
    
    u_c (ndarray[:]): List of checkpoint intervals (node-hrs) corresponding to jobs[:]
    
    u_chk (ndarray[:]): List of checkpoint costs corresponding to jobs[:]
    
    continuous (Bool, default=False): Whether to use the continuous approximation. This makes
      results more smooth, but is likely less accurate.
      
    Returns: N_chk for each job in jobs
    
    """
    
    if not continuous:
        N_chk = jobs[:,0] * jobs[:,1] / (u_c + u_chk)                                                  
        N_chk = N_chk.astype(int) + 1 # Round up
    else:
        N_chk = jobs[:,0] * jobs[:,1] / (u_c + u_chk) + 1
        
    return N_chk

def per_job_chkpt_cost(jobs, u_c, u_chk, R_0, continuous=False):
    """
    Expected per-job cost of prophylactic checkpointing
    Parameters:
    
    jobs (ndarray[:,2]): List of jobs. jobs[:,0] is jobsize, jobs[:,1] is job duration
    
    u_c (ndarray[:]): List of checkpoint intervals (node-hrs) corresponding to jobs[:]
    
    u_chk (ndarray[:]): List of checkpoint costs corresponding to jobs[:]
    
    R_0 (float): Failure rate (failures/node-hr)
    
    Returns: Cost of checkpointing for all jobs (node-hrs) from Eq. (4.3) of CE
    
    """
    
    p_s = np.exp(-R_0*(u_c+u_chk))
    N_chk = get_N_chk(jobs, u_c, u_chk, continuous=continuous)

    L = u_chk * (p_s - p_s**N_chk) / (1 - p_s)

    return L

def per_job_failure_cost(jobs, u_c, u_chk, R_0, continuous=False):
    """
    Expected per job cost due to failures.
    Parameters:
    
    jobs (ndarray[:,2]): List of jobs. jobs[:,0] is jobsize, jobs[:,1] is job duration
    
    u_c (ndarray[:]): List of checkpoint intervals (node-hrs) corresponding to jobs[:]
    
    u_chk (ndarray[:]): List of checkpoint costs corresponding to jobs[:]
    
    R_0 (float): Failure rate (failures/node-hr)
    
    Returns: Cost of failures (node-hrs) for all jobs using Eq. (4.4) of CE
    
    """
    
    u_F = per_failure_cost(u_c, u_chk, R_0)
    
    p_s = np.exp(-R_0*(u_c+u_chk))
    N_chk = get_N_chk(jobs, u_c, u_chk, continuous=continuous)

    L = (1 - p_s**N_chk) * u_F

    return L

def per_job_restart_cost(jobs, u_c, u_chk, R_0, tau_0, continuous=False):
    """
    Per-job restart cost
    
    Parameters:
    
    jobs (ndarray[:,2]): List of jobs. jobs[:,0] is jobsize, jobs[:,1] is job duration
    
    u_c (ndarray[:]): List of checkpoint intervals (node-hrs) corresponding to jobs[:]
    
    u_chk (ndarray[:]): List of checkpoint costs corresponding to jobs[:]
    
    R_0 (float): Failure rate (failures/node-hr)
    
    Returns: Cost of restarts (node-hrs) for all jobs using Eq. (4.5) of CE
    
    """
    
    u_R = per_restart_cost(jobs, u_chk, tau_0, R_0)
    p_s = np.exp(-R_0*(u_c+u_chk))
    N_chk = get_N_chk(jobs, u_c, u_chk, continuous=continuous)

    L = (1 - p_s**N_chk) * u_R

    return L

def total_cost(jobs, u_c, u_chk, R_0, tau_0, return_all=False, continuous=False):
    """
    Per-job total cost
    
    Parameters:
    
    jobs (ndarray[:,2]): List of jobs. jobs[:,0] is jobsize, jobs[:,1] is job duration
    
    u_c (ndarray[:]): List of checkpoint intervals (node-hrs) corresponding to jobs[:]
    
    u_chk (ndarray[:]): List of checkpoint costs corresponding to jobs[:]
    
    R_0 (float): Failure rate (failures/node-hr)
    
    Returns: Total cost (node-hrs) summed over all jobs
    
    """
    
    L_chk = per_job_chkpt_cost(jobs, u_c, u_chk, R_0, continuous)
    L_F = per_job_failure_cost(jobs, u_c, u_chk, R_0, continuous)
    L_R = per_job_restart_cost(jobs, u_c, u_chk, R_0, tau_0, continuous)
    
    cost = L_F + L_chk + L_R
    cost = cost.sum()
    
    if return_all:
        return cost, L_F.sum(), L_chk.sum(), L_R.sum()
    else:
        return cost
    
def per_job_usage(jobs, u_c, u_chk, u_F, R_0, continuous=False):
    """
    Per-job expected usage cost
    
    Parameters:
    
    jobs (ndarray[:,2]): List of jobs. jobs[:,0] is jobsize, jobs[:,1] is job duration
    
    u_c (ndarray[:]): List of checkpoint intervals (node-hrs) corresponding to jobs[:]
    
    u_chk (ndarray[:]): List of checkpoint costs corresponding to jobs[:]
    
    u_F (ndarray[:]): List of failure costs corresponding to jobs[:]
    
    R_0 (float): Failure rate (failures/node-hr)
    
    Returns: Expected usage (node-hrs), using Eq. (4.7) of CE, summed over all  jobs
    
    """
    
    p_s = np.exp(-R_0*(u_c+u_chk))
    N_chk = get_N_chk(jobs, u_c, u_chk, continuous=continuous)
    
    ubar = p_s * (1-p_s**N_chk) * (u_c + u_chk) / (1-p_s) + (1-p_s**N_chk) * u_F
    
    # print(f"Mean Ubar = {ubar.mean():.3f}")
    
    ubar=ubar.sum()
    
    return ubar  