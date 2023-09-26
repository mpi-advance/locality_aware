# September 2023 Hackathon Overview

At a high level, the September 2023 hackathon will be looking into separating Topology information from MPI Communicators, with hopes to tackle the following goals: 
 1. Decouple topologies from MPI Communicators so that graph topology creation is more efficient. This in turn will (hopefully) make creating communicators with that topology more efficient.
 2. Extend the functionality of topologies to support features like “flipping the direction” of an existing topology.

The code in this repository is a clone of the locality aware library from MPI Advance. For most of the work, everyone will be running the benchmark [here](benchmarks/neighbor_collective.cpp). Instructions on running the program will be up soon, but the input matrix is found in the `test_data` folder. During the hackathon, we will time various parts of this benchmark, along with the solutions we end up creating. See below for more details. A fuller explanation of the code will also be done at the hackathon.

## Background Material
To prepare for this hackathon, it would be a good idea to become familiar with the following topics:
 1. System and repository access. We will mostly be running on LLNL's Quartz system, so you are encouraged to build this repo on that system (the original README for this repository is [here](README-MPI-ADVANCE.md))  If you do not have access to this system, or wish to do it on another system, it is on you to test building this repository on a that system BEFORE the hackathon starts. We will not be doing any GPU work, and you should be able to work out of this repository (or a fork of it). Please also test access to the Google Sheet linked below.
 2. The MPI graph topology functions and their related neighbor collective functions, and what features they enable.
 3. Additionally, make sure you understand how to time things using MPI (or your favorite tool). 
 4. Dr. Bienz's and Gerald's [paper](https://arxiv.org/abs/2306.01876) on the optimizations they have done in their MPIX versions of the APIs from #2. Note, the use of an `MPIX_Comm` is to store the topology information since the collectives do not pass this information. The `MPIX_Dist_graph_create_adjacent` call is just a normal call to `MPI_Dist_graph_create_adjacent`+ storage of the inputs.
 5. A brief overview of Sparse matrix–vector multiplication (SPMV) will be presented at the Monday, September 18th PSAAP Hacking Session. If you cannot attend, a brief understanding of the concept is recommended. You will not actually be tweaking the application's code use of SPMV, but understanding what the application wants to do on this front will inform the overall goal of this hackathon.

## Results
We will aim to collect the results into a Google Sheets to capture any findings we collect during the hackathon, which can be found [here](https://docs.google.com/spreadsheets/d/1xDqE80EngrAFmneI0dwE1wHwPCu9RiX74rXdMpkEeik/edit?usp=sharing). In that document, each participant will put their results on a different page (denoted by your initials). On that page, there are spots to record the system, the c++ version, the MPI version, the parameters of the run, and the results. For each run, you will initially time six things:
 1. The `form_comm` function
 2. The `communicate` function
 3. The `MPI_Dist_graph_create_adjacent` call
 4. The `MPI_Neighbor_alltoallv` call
 5. The `MPIX_Dist_graph_create_adjacent` call
 6. The `MPIX_Neighbor_alltoallv` call

After creating a solution, more columns may be added to record the results of the solution.
The main page will contain a summary report of everyone's runs.

