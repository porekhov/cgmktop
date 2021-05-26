In this example, the parametrization of chlorhexidine is carried out.

The parametrization procedure implies the following steps:

-= Preparation of coarse-grained system =-
1. Use the standard gromacs tools and/or insane.py script to prepare. Solvate the system with the desired solvent, add ions if required, etc. Please, refer to the MARTINI tutorials if necessary, http://cgmartini.nl/index.php/tutorials-general-introduction-gmx5

2. Minimize the system and perform a short equilibration run. The examples of mdp files for both runs are provided (em.mdp, eq.mdp)

3. Edit the iter.sh in order to provide correct names/paths for all-atom structure and trajectory, group ID of the parametrized molecules, itp file, etc. See the comments in iter.sh

4. Run iter.sh

5. Plot the resulting distributions of the bonds/angles using plot.py (check -h for the input parameters).