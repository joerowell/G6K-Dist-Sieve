This directory contains a series of scripts for running MPI jobs on the Holzfusion / Solardiesel / Atomkohle cluster.

In particular, the useful scripts are:

- ```execute_svp_challenge.sh``` is used for executing the SVP challenge code. This should be run from the root node (i.e
  Holzfusion). This is necessary for a weird MPICH design decision (see caveats below).
- ```run_init.sh``` is used for running the initial compilation of the distributed version of G6K. This should not be used
  more than, say, once.
- ```run_recompile.sh``` is used for recompiling the code across multiple machines.


There are multiple subscripts that are in this directory: each is called by one of these three scripts.

## Configuration
There's two host files in this directory: one that runs a single process per node, and one that runs two processes per
node. These are marked as hosts_singular and hosts respectively. These files are set-up to run based on IP addresses,
rather than on hostnames. This appears to resolve a rather annoying bug in MPICH's Hydra launcher.

The hostnames / ip mapping:

- 10.201.201.7.1 is Holzfusion
- 10.201.201.6.1 is Solardiesel
- 10.201.201.5.1 is Atomkohle

Please note that the Hydra launcher will run jobs in the order specified in the host file In other words, with the listing
given above, the first invoked MPI program will run on Holzfusion, the second on Solardiesel, and so on. To change this behaviour,
change the order of the ip addresses. 

The config files (i.e. files that end in _config) are used for specifying which processes should run, and in which order.
This is explained in more detail below.

## Caveats
Most of the G6K code for running SVP instances relies upon certain commandline arguments. However, MPICH's Hydra launcher does
not easily support passing commands to only one of a subset of processes. This can be a little bit of a headache, since we often
care a lot about the runtime arguments that are supplied.

The scripts that need this (e.g. ```execute_svp_challenge.sh```) circumvent these issues by storing arguments as environment variables,
which can then be extracted by subscripts. The subscripts themselves are called based on the particular configuration file.

As an example, consider ```execute_svp_challenge.sh```.

This script accepts some command line arguments and then starts an MPI instance
with ```hosts``` as the hostfile and ```run_svp_challenge_config``` as the configuration file.

At present, ```run_svp_challenge_config``` looks like this:

```bash
# Holzfusion has 2 physical CPUs, but we also run the top-level python program on this node.
-n 1 ./execute_svp_challenge_py.sh 
-n 1 ./execute_worker.sh --threads 48

# Solardiesel has 2 physical CPUs
-n 2 ./execute_worker.sh --threads 18

# Atomkohle has 2 physical CPUs
-n 2 ./execute_worker.sh --threads 14
```

We then rely upon ```execute_svp_challenge_py.sh``` to extract the arguments given to ```execute_svp_challenge.sh```, with
```execute_worker.sh``` just responsible for setting the number of threads.

You can think of the flow as being like this:

1. We invoke the top-level script with the relevant command line arguments.
2. The top-level script delegates to the configuration file, which launches the relevant subscripts.
3. Each subscript invokes the right functionality with the arguments supplied.

