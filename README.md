# HPC_Computational_Efficiency

This repository contains the Python code and Jupyter notebooks that were used to perform the analyses and produce the plots for the manuscript *Application Failures and Machine Computational Efficiency*, by Graziani, Lusch, and Messer, submitted to the SC-2025 conference.

The utility code is contained in the file [efficiency.py](https://github.com/CarloGraziani/HPC_Computational_Efficiency/blob/main/efficiency.py).

The file [jobs2024.csv](https://github.com/CarloGraziani/HPC_Computational_Efficiency/blob/main/jobs2024.csv) contains a database of 331,640 production jobs from the *Frontier* supercomputer at ORNL. These data are one year (CY 2024) of jobs from the *INCITE* and *ALCC* programs, with a further selection for jobs of more than 1 node in size, to exclude interactive and debug jobs.

The notebook [Machine_Efficiency_Analysis.ipynb](https://github.com/CarloGraziani/HPC_Computational_Efficiency/blob/main/Machine_Efficiency_Analysis.ipynb) imports the functions in the efficiency.py file to compute machine efficiency assuming the computational load that characterizes 1 year of production jobs on *Frontier*. The analysis results in the paper, and all but one of the figures, are output by this notebook.

The notebook [Optimal_Checkpointing.ipynb](https://github.com/CarloGraziani/HPC_Computational_Efficiency/blob/main/Optimal_Checkpointing.ipynb) contains code for computing the optimal checkpointing cadence, essentially updating the work of [Daly (2006)](https://doi.org/10.1007/3-540-44864-0_1) to give cadences in usage (node-hrs) rather than in time (hrs).  This notebook also produces Figure 1 of the paper.