# remove any pickles from previous runs
rm *.itp.p
# backup the initial itp file generated by aa2cg.py
cp MOL_CG.itp MOL_CG.itp.bak
# enter the maximum number of iterations
for i in {1..20};
do
# run simulation
gmx grompp -f md.mdp -c eq.gro -p topol.top -o md.tpr -maxwarn 10
gmx mdrun -deffnm md -v
# replace GROUP with the group ID corresponding to the parametrized molecule
gmx trjconv -f md.xtc -o md.xtc -pbc mol -s md.tpr << EOF
GROUP
EOF
gmx trjconv -f md.xtc -o md.gro -pbc mol -s md.tpr -b 0 -e 0 << EOF
GROUP
EOF
# replace:
# AA.pdb, AA.xtc with the all-atom structure and trajectory
# MOL_opt_map.dat, MOL_CG.itp, params.dat with the files generated by aa2cg.py
# 
python optimizer.py -s AA.pdb -t AA.xtc -m MOL_opt_map.dat -i MOL_CG.itp -p params.dat -c md
done