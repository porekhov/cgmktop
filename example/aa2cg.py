# -*- coding: utf-8 -*-
# Script for coarse-graining of AA structure
# needs MDAnalysis, numpy, scipy
import sys, os, optparse, re, numpy
import MDAnalysis
import numpy as np
import numpy.linalg
from MDAnalysis.topology.guessers import guess_angles
from MDAnalysis.topology.guessers import guess_dihedrals

parser = optparse.OptionParser()
parser.add_option('-s','--str', dest='str', help = 'AA structure')
parser.add_option('-x','--trj', dest='trj', help = 'AA trajectory')
parser.add_option('-o','--out', dest='out', help = 'CG structure and trajectory')
parser.add_option('-m','--map', dest='map', help='CG to AA map')
parser.add_option('-n','--name', dest='name', help='molecule name')

(options,args) = parser.parse_args()
absaas = os.path.abspath(options.str)
absaat = os.path.abspath(options.trj)
absout = os.path.abspath(options.out)
absmap = os.path.abspath(options.map)
res_type = str(options.name).upper()[:4]

###################
#vector3d.py

import math
import random

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0
SMALL = 1E-6

def is_near_zero(a):
  return a < SMALL

class Vector3d:

  def __init__(self, x=0.0, y=0.0, z=0.0):
    self.x = x
    self.y = y
    self.z = z

  def __add__(self, rhs):
    return Vector3d(rhs.x + self.x, rhs.y + self.y, rhs.z + self.z)

  def __sub__(self, rhs):
    return Vector3d(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)

  def __neg__(self):
    return Vector3d(-self.x, -self.y, -self.z)

  def __pos__(self):
    return Vector3d(self.x, self.y, self.z)

  def __eq__(self, rhs):
    return (is_near_zero(self.x - rhs.x) and \
            is_near_zero(self.y - rhs.y) and \
            is_near_zero(self.z - rhs.z))

  def __ne__(self, rhs):
    return not (x == rhs)

  def __str__(self):
    return "(% .2f, % .2f, % .2f)" % (self.x, self.y, self.z)

  def __repr__(self):
    return "Vector3d(%f, %f, %f)" % (self.x, self.y, self.z)
    
  def __getitem__(self, index):
    if index == 0: return self.x
    if index == 1: return self.y
    if index == 2: return self.z

  def set(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def copy(self):
    return Vector3d(self.x, self.y, self.z)

  def length_sq(self):
    return self.x*self.x + self.y*self.y + self.z*self.z

  def length(self):
    return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

  def scale(self, scale):
    self.x *= scale
    self.y *= scale
    self.z *= scale

  def normalize(self):
    self.scale(1.0 / self.length())

  def scaled_vec(self, scale):
    v = self.copy()
    v.scale(scale)
    return v

  def normal_vec(self):
    return self.scaled_vec(1.0 / self.length())

  def parallel_vec(self, axis):
    axis_len = axis.length()
    if is_near_zero(axis_len):
      result = self
    else:
      result = axis.scaled_vec(dot(self, axis) 
               / axis.length() / axis.length())
    return result

  def perpendicular_vec(self, axis):
    return self - self.parallel_vec(axis)

############################################

def dot(a, b):
  return a.x*b.x + a.y*b.y + a.z*b.z

def CrossProductVec(a, b):
  return Vector3d(a.y*b.z - a.z*b.y,
                  a.z*b.x - a.x*b.z,
                  a.x*b.y - a.y*b.x)

def vec_angle(a, b):
  a_len = a.length()
  b_len = b.length()

  if a_len * b_len < 1E-6:
    return 0.0

  c = dot(a, b) / a_len / b_len

  if c >=  1.0:
    return 0.0
  elif c <= -1.0:
    return math.pi
  else:
    return math.acos(c)

def vec_dihedral(a, axis, c):
  ap = a.perpendicular_vec(axis)
  cp = c.perpendicular_vec(axis)

  angle = vec_angle(ap, cp)

  if dot(CrossProductVec(ap, cp), axis) > 0:
    angle = -angle

  return angle

def pos_dihedral(p1, p2, p3, p4):
  return vec_dihedral(p1-p2, p2-p3, p4-p3)

################################################

class Atom:
    '''Atom object'''
    def __init__(self, id, type, resid, resname, name, cgnr, charge):
        '''Init function for Atom object'''
        self.id = int(id)
        self.resid = int(resid)
        self.resname = str(resname)
        self.type = str(type)
        self.name = str(name)
        self.charge = float(charge)
        self.cgnr = int(cgnr)
    
    def fmt(self):
        '''formatted line for Atom object'''
        return "%3d %5s %3d %5s %5s %3d %1.3f" \
            % (self.id, self.type, self.resid, self.resname, self.name, self.cgnr, self.charge)


class Bond:
    '''Bond object'''
    def __init__(self, atom1, atom2, l, k, id):
        '''Init function for Bond object'''
        self.id = id # unique internal id of the bond
        self.atom1 = int(atom1) # atom1 unique id as in the itp
        self.atom2 = int(atom2) # atom2 unique id as in the itp
        self.type = 1 # type of bond
        self.k = float(k) # current k
        self.l = float(l) # current l

    def fmt(self):
        '''formatted line for Bond object'''
        return "%3d %3d %1d %5.3f %5.3f" \
            % (self.atom1, self.atom2, self.type, self.l, self.k)

class Angle:
    '''Angle object'''
    def __init__(self, atom1, atom2, atom3, phi, k, id):
        '''Init function for Angle object'''
        self.id = id
        self.atom1 = int(atom1)
        self.atom2 = int(atom2)
        self.atom3 = int(atom3)
        self.type = 2
        self.k = float(k)
        self.phi = float(phi)
  
    def fmt(self):
        '''formatted line for Angle object'''
        return "%3d %3d %3d %1d %5.3f %5.3f" \
            % (self.atom1, self.atom2, self.atom3, self.type, self.phi, self.k)

class Dihedral:
    '''Dihedral object'''
    def __init__(self, atom1, atom2, atom3, atom4, type, phi, k, id, mult):
        '''Init function for Angle object'''
        self.id = id
        self.atom1 = int(atom1)
        self.atom2 = int(atom2)
        self.atom3 = int(atom3)
        self.atom4 = int(atom4)
        self.type = int(type)
        self.k = float(k)
        self.phi = float(phi)
        self.mult = int(mult)
  
    def fmt(self):
        '''formatted line for Dihedral object'''
        return "%3d %3d %3d %3d %1d %5.3f %5.3f %1d" \
            % (self.atom1, self.atom2, self.atom3, self.atom4, self.type, self.phi, self.k, self.mult)

class Molecule:
    def __init__(self):
        '''Init function for Molecule object'''
        self.molname = ''
        self._atoms = {}
        self._bonds = []
        self._constraints = []
        self._angles = []
        self._dihedrals = []
        self._exclusions = []
        self.atom_num = 0
        self.res_num = 0
        
    def write_itp(self, fname):
        '''Write an itp file'''
        out = ['; AUTOMATICALLY GENERATED TOPOLOGY FILE FOR ' + res_type]
        out.append(' [ moleculetype ] ')
        out.append('; molname         nrexcl')
        out.append(str(self.molname) + '   3')
        out.append(' [ atoms ] ')
        out.append(';id type resnr residu atom cgnr charge')
        out.extend([i.fmt() for i in self._atoms.values()])
        out.append(' [ bonds ] ')
        out.append(';atom_id1 atom_id2 func_type length force_c')
        out.extend([i.fmt() for i in self._bonds])
        out.append(' [ constraints ] ')
        out.append(';atom_id1 atom_id2 func_type length')
        out.extend(self._constraints)
        out.append(' [ angles ] ')
        out.append(';atom_id1 atom_id2 atom_id3 func_type angle force_c')
        out.extend([i.fmt() for i in self._angles])
        out.append(' [ dihedrals ] ')
        out.append(';atom_id1 atom_id2 atom_id3 atom_id4 func_type dih force_c')
        out.extend([i.fmt() for i in self._dihedrals])
        out.append(' [ exclusions ] ')
        out.extend(self._exclusions)
        fout = open(fname, 'w')
        fout.write("\n".join(out))
        fout.close()

##############################
#### Writing CG structure ####
##############################

print('INFO: Writing CG structure.')

map_cg2aa = {} # dictionary with the CG-to-AA mapping
cg_particle_types = {} #
cg_particle_names = []
cg_particle_names_list = []
res_ids = [] # to do: change to the value of resid field in the mapping

for line in open(absmap, 'r').readlines():
    if line.replace(" ", "")[0] not in [';', '#', '@'] and re.search('[a-zA-Z0-9]', line):
        if line.replace(" ", "").startswith('[mapping]'):
            cur_sec = 'm'
            continue
        if cur_sec == 'm':
            if str(line.split(':')[0]) in map_cg2aa:
                print('Duplicated CG particle definition in the mapping file!')
                exit()
            else:
                map_cg2aa[str(line.split(':')[0])] = line.split(':')[2].split()
                cg_particle_names.append(str(line.split(':')[0]))
                cg_particle_names_list.append(str(line.split(':')[0]))
                res_ids.append(int(1)) # to do: change to the value of id field in the mapping
                cg_particle_types[str(line.split(':')[0])] = line.split(':')[1]

u_aa = MDAnalysis.Universe(str(absaas), str(absaat), guess_bonds=True)

atom_id = 1
chain_id = 'A'

cg_particles = []
cg_opt_map = ['[ mapping ]']

pdb_ids = [i.id for i in u_aa.atoms]
mda_ids = np.arange(1, len(pdb_ids) + 1)
id_dic = dict(zip(pdb_ids, mda_ids))

cg_ids_dic = {}

for particle_id in enumerate(cg_particle_names):
    sel = u_aa.select_atoms('bynum ' + ' '.join([str(id_dic[int(i)]) for i in map_cg2aa[particle_id[1]]]))
    com = sel.center_of_mass()
    type = particle_id[1]
    res_id = res_ids[particle_id[0]]
    cg_particles.append("%6s%5s %4s %3s %1s%4d    %8.3f%8.3f%8.3f" % ("ATOM  ", atom_id, type, res_type, chain_id, res_id, com[0], com[1], com[2]))
    cg_opt_map.append(str(atom_id) + ':' + ' '.join([str(id_dic[int(i)]) for i in map_cg2aa[particle_id[1]]]))
    cg_ids_dic[type] = atom_id
    atom_id += 1

fout = open(res_type + '_opt_map.dat', 'w')
fout.write("\n".join(cg_opt_map))
fout.close()

fout = open(absout + '.pdb', 'w')
fout.write("\n".join(cg_particles))

cg_bonds = np.zeros([len(cg_particles), len(cg_particles)])
aa_bonds = np.zeros([len(u_aa.atoms), len(u_aa.atoms)])

for bond in u_aa.bonds:
    aa_bonds[bond.atoms[0].index, bond.atoms[1].index] = 1
    aa_bonds[bond.atoms[1].index, bond.atoms[0].index] = 1

cg_bonds = []

for cg_particle in cg_particle_names:
    cg_particle_names_list.remove(cg_particle)
    for cg_partner in cg_particle_names_list:
        cg_particle_ids = [id_dic[int(i)] - 1 for i in map_cg2aa[cg_particle]]
        cg_partner_ids = [id_dic[int(i)] - 1 for i in map_cg2aa[cg_partner]]
        if aa_bonds[cg_particle_ids][:,cg_partner_ids].sum() > 0:
            cg_bonds.append("%6s%5s%5s" % ("CONECT", str(cg_ids_dic[cg_particle]), str(cg_ids_dic[cg_partner])))

fout.write('\n' + '\n'.join(cg_bonds) + '\n')
fout.close()

###################################
#### writing the CG trajectory ####
###################################

print('INFO: Writing CG trajectory.')

from MDAnalysis.coordinates.memory import MemoryReader

coordinates = np.expand_dims(MDAnalysis.Universe(str(absout) + '.pdb').atoms.positions, axis = 0)

for ts in u_aa.trajectory:
    coord_ts = []
    for particle_id in enumerate(cg_particle_names):
        sel = u_aa.select_atoms('bynum ' + ' '.join([str(id_dic[int(i)]) for i in map_cg2aa[particle_id[1]]]))
        coord_ts.append(sel.center_of_mass())
    coord_ts = np.expand_dims(np.array(coord_ts), axis = 0)
    coordinates = np.append(coordinates, coord_ts, axis = 0)

u_trj_cg = MDAnalysis.Universe(str(absout) + '.pdb', coordinates, format=MemoryReader, order='fac')
u_trj_cg_all = u_trj_cg.select_atoms('all')
with MDAnalysis.Writer(str(absout) + '.xtc', u_trj_cg_all.n_atoms) as W:
    for ts in u_trj_cg.trajectory:
        W.write(u_trj_cg_all)

#############################################
#### writing parameters for optimization ####
#############################################

print('INFO: Writing parameters to be optimized.')

u_cg = MDAnalysis.Universe(str(absout) + '.pdb')

params_out = []

params_out.append('[bonds]')
bond_id = 1
for bond in u_cg.bonds:
    bond_atom_1_type = cg_particle_types[cg_particle_names[bond.atoms[0].id - 1]]
    bond_atom_2_type = cg_particle_types[cg_particle_names[bond.atoms[1].id - 1]]
    params_out.append((str(bond.atoms[0].id) + ' ' + str(bond.atoms[1].id) + ' ' + str(bond_id) + \
    '; ' + bond_atom_1_type + '-' + bond_atom_2_type))
    bond_id += 1
    
params_out.append('[angles]')
angle_id = 1
for angle in guess_angles(u_cg.bonds):
    angle_atom_1_type = cg_particle_types[cg_particle_names[angle[0]]]
    angle_atom_2_type = cg_particle_types[cg_particle_names[angle[1]]]
    angle_atom_3_type = cg_particle_types[cg_particle_names[angle[2]]]
    params_out.append((str(angle[0] + 1) + ' ' + str(angle[1] + 1) + ' ' + str(angle[2] + 1) + ' ' + str(angle_id)) \
                     + '; ' + angle_atom_1_type + '-' + angle_atom_2_type + '-' + angle_atom_3_type)
    angle_id += 1

fout = open('params.dat', 'w')
fout.write("\n".join(params_out))
fout.close()

# print '# Dihedrals:'
# dih_id = 1
# for dih in guess_dihedrals(guess_angles(u_cg.bonds)):
#     print str(dih[0] + 1), ' ', str(dih[1] + 1), ' ', str(dih[2] + 1), ' ', str(dih[3] + 1), ' ', str(dih_id)
#     dih_id += 1

##############################
### generation of topology ###
##############################

print('INFO: Writing CG topology.')

cur_sec = ''
mol = Molecule()
mol.molname = res_type

u_cg = MDAnalysis.Universe(str(absout) + '.pdb', str(absout) + '.xtc')

bead_type = cg_particle_types

for i in u_cg.atoms:
    if bead_type[str(i.name)][:2] == 'Qa' or bead_type[str(i.name)][:3] == 'SQa':
        charge = -1
    elif bead_type[str(i.name)][:2] == 'Qd' or bead_type[str(i.name)][:3] == 'SQd':
        charge = 1
    else:
        charge = 0
    
    mol._atoms[int(i.id)] = Atom(int(i.id), bead_type[str(i.name)], int(i.resid), mol.molname, str(i.name), int(i.resid), charge)

for line in open('params.dat', 'r').readlines():
    if line.replace(" ", "")[0] != ';' and line.replace(" ", "")[0] != "#" \
                                       and re.search('[a-zA-Z0-9]', line):

        if line.replace(" ", "").startswith('[bonds]'):
            cur_sec = 'b'
            continue

        if line.replace(" ", "").startswith('[angles]'):
            cur_sec = 'a'
            continue

        if line.replace(" ", "").startswith('[dihedrals]'):
            cur_sec = 'd'
            continue

        if cur_sec == 'b':
            fields = line.rstrip('/n').split(';')[0].split()
            d = []
            for ts in u_cg.trajectory:
                d.append(numpy.linalg.norm(u_cg.atoms[int(fields[0]) - 1].position - u_cg.atoms[int(fields[1]) - 1].position)/10)
            d_mean = np.mean(d)
            #d_std = np.std(d)
            #k = 8.314*298/(2*(10*d_std)**2)
            #print k
            mol._bonds.append(Bond(int(fields[0]), int(fields[1]), d_mean, 10000, int(fields[2])))

        if cur_sec == 'a':
            fields = line.rstrip('/n').split(';')[0].split()
            ang = []
            for ts in u_cg.trajectory:
                com1, com2, com3 = u_cg.atoms[int(fields[0]) - 1].position, u_cg.atoms[int(fields[1]) - 1].position, u_cg.atoms[int(fields[2]) - 1].position
                vec1, vec2 = com1 - com2, com3 - com2
                ang.append(np.rad2deg(np.arccos(np.dot(vec1, vec2)/(numpy.linalg.norm(vec1)*numpy.linalg.norm(vec2)))))
            ang_mean = np.mean(ang)
            ang_std = np.std(ang)
            mol._angles.append(Angle(int(fields[0]), int(fields[1]), int(fields[2]), ang_mean, 90, int(fields[3])))
            
        if cur_sec == 'd':
            fields = line.rstrip('/n').split()
            com1, com2, com3, com4 = u_cg.atoms[int(fields[0]) - 1].position, u_cg.atoms[int(fields[1]) - 1].position, u_cg.atoms[int(fields[2]) - 1].position, u_cg.atoms[int(fields[3]) - 1].position
            v1 = Vector3d(com1[0], com1[1], com1[2])
            v2 = Vector3d(com2[0], com2[1], com2[2])
            v3 = Vector3d(com3[0], com3[1], com3[2])
            v4 = Vector3d(com4[0], com4[1], com4[2])
            dih = np.rad2deg(pos_dihedral(v1, v2, v3, v4))
            mol._dihedrals.append(Dihedral(int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]), 1, dih, 10, int(fields[4]), 1))

mol.write_itp(res_type + '_CG.itp')

