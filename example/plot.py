# -*- coding: utf-8 -*-
# Script for the iterative optimization of the CG parameters

import itp
import sys, os, optparse, re
import MDAnalysis
import numpy as np
from numpy.linalg import norm
from scipy import stats
import subprocess
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
import pickle

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

max_iter = 10 # maximal number of iterations
conv_crit = 0.9 # convergence criterium
k_change = 0.5 # scaling factor for the iteration step
l_max = 1 # maximal bond length in nm
b_k_max = 10000 # maximum value for the bond force constant

# calculates Heilliger distance between two distributions
def calc_overlap(x1, x2, pdf1, pdf2):
    min_m = np.min([np.min(x1), np.min(x2)])
    max_m = np.max([np.max(x2), np.max(x2)])
    x = np.linspace(min_m, max_m, 250)
    pdf1_val = gaussian_kde(x1)(x)
    pdf2_val = gaussian_kde(x2)(x)
    func_min = np.minimum(pdf1_val, pdf2_val)
    overlap = np.trapz(func_min, x = x)
    return overlap
# calculates histogram
def calc_hist(vec):
    vec = np.nan_to_num(vec)
    v_min, v_max = min(vec), max(vec)
    x = np.linspace(v_min, v_max, 250)
    g_pdf = gaussian_kde(vec)
    pdf = g_pdf(x)
    return (x, pdf)
# calculates histogram and normal distribution
def calc_hist_pdf(vec, m, std):
    v_min, v_max = min(vec), max(vec)
    x = np.linspace(v_min, v_max, 250)
    g_pdf = gaussian_kde(vec)
    pdf = g_pdf(x)
    pdf_norm = stats.norm.pdf(x, m, std)
    return (x, pdf, pdf_norm)

parser = optparse.OptionParser()
parser.add_option('-s','--str', dest='str', help = 'AA structure')
parser.add_option('-t','--trj', dest='trj', help = 'AA trajectory file')
parser.add_option('-m','--map', dest='map', help='CG to AA map')
parser.add_option('-p','--par', dest='par', help='list of parameters to optimize')
parser.add_option('-i','--itp', dest='itp', help='name of topology file')
parser.add_option('-c','--cg', dest='cg', help='generic name for the CG structure and trajectory')
parser.add_option('-l','--load', dest='load', help='use precalculated distributions, Y/N')

(options,args) = parser.parse_args()
absaas = os.path.abspath(options.str)
absaat = os.path.abspath(options.trj)
absmap = os.path.abspath(options.map)
abspar = os.path.abspath(options.par)

print('Info: Reading the CG2AA mapping...')

# dictionary with the CG-to-AA mapping

map_cg2aa = {}
half_mass_list = []

for line in open(absmap, 'r').readlines():
    if line.replace(" ", "")[0] != ';' and line.replace(" ", "")[0] != "#" \
                                       and re.search('[a-zA-Z0-9]', line):

        if line.replace(" ", "").startswith('[mapping]'):
            cur_sec = 'm'
            continue

        if line.replace(" ", "").startswith('[halfmass]'):
            cur_sec = 'h'
            continue

        if cur_sec == 'm':
            if line.split(':')[0] in map_cg2aa:
                print('Duplicated CG particle definition in the mapping file!')
                exit()
            else:
                map_cg2aa[int(line.split(':')[0])] = line.split(':')[1].split()

        if cur_sec == 'h':
            if line.split()[0] in half_mass_list:
                print('Duplicated CG particle definition in the half-mass section!')
                exit()
            else:
                half_mass_list.append(line.split()[0])

print('Info: Masses for the following atoms will be halved:', " ".join(map(lambda x: str(x), half_mass_list)))

################# ANALYSIS of AA simulation #################
if str(options.load).upper()[0] in ['1', 'T', 'Y']:
    try:
        mol = pickle.load(open(str(options.itp) + ".p", "rb" ))
        print('Info: AA distributions were successfully loaded.')
        print('Info: ' + str(len(mol.b_dic_opt)), ' bond parameters will be shown.')
        print('Info: ' + str(len(mol.an_dic_opt)), ' angle parameters will be shown.')
        print('Info: ' + str(len(mol.dih_dic_opt)), ' dihedral parameters will be shown.')
    except:
        print('Warning: Problems with loading the pre-calculated AA distributions. They will be recalculated now.')
else:
    print('Info: Reading itp topology...')
    mol = itp.Molecule(str(options.itp))
        
    # reading file with the parameters to be optimized
    print('Info: Reading list of parameters to optimize...')
    cur_sec = ''
    for line in open(abspar, 'r').readlines():
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
                if int(fields[2]) not in mol.b_dic_opt:
                    mol.b_dic_opt[int(fields[2])] = [mol.b_atoms2id[(int(fields[0]), int(fields[1]))]]
                else:
                    mol.b_dic_opt[int(fields[2])].append(mol.b_atoms2id[(int(fields[0]), int(fields[1]))])

            if cur_sec == 'a':
                fields = line.rstrip('/n').split(';')[0].split()
                if int(fields[3]) not in mol.an_dic_opt:
                    mol.an_dic_opt[int(fields[3])] = [mol.an_atoms2id[(int(fields[0]), int(fields[1]), int(fields[2]))]]
                else:
                    mol.an_dic_opt[int(fields[3])].append(mol.an_atoms2id[(int(fields[0]), int(fields[1]), int(fields[2]))])

            if cur_sec == 'd':
                fields = line.rstrip('/n').split()
                if int(fields[4]) not in mol.dih_dic_opt:
                    mol.dih_dic_opt[int(fields[4])] = [mol.dih_atoms2id[(int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]))]]
                else:
                    mol.dih_dic_opt[int(fields[4])].append(mol.dih_atoms2id[(int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]))])

    print('Info: ' + str(len(mol.b_dic_opt)), ' bond parameters will be shown.')
    print('Info: ' + str(len(mol.an_dic_opt)), ' angle parameters will be shown.')
    print('Info: ' + str(len(mol.dih_dic_opt)), ' dihedral parameters will be shown.')

    print('Info: reading AA structure and trajectory, calculating AA distributions...')

    u_aa = MDAnalysis.Universe(str(absaas), str(absaat))

    for i in half_mass_list:
        sel = u_aa.select_atoms('bynum ' + i)
        sel.masses = 0.5 * sel.masses

    # calculating distribution of BONDS from AA simulations

    for b_g in mol.b_dic_opt.values():
        data_set = []
        for b_id in b_g:
            # atom 1
            atom1_aa_l = map_cg2aa[mol._bonds[b_id].atom1]
            sel1_s = 'bynum ' + ' or bynum '.join(atom1_aa_l)
            sel1 = u_aa.select_atoms(sel1_s)
            # atom 2
            atom2_aa_l = map_cg2aa[mol._bonds[b_id].atom2]
            sel2_s = 'bynum ' + ' or bynum '.join(atom2_aa_l)
            sel2 = u_aa.select_atoms(sel2_s)
            for ts in u_aa.trajectory:
                com1, com2 = sel1.center_of_mass(), sel2.center_of_mass()
                data_set.append(norm(com1 - com2) * 0.1)
        
        aa_fit = stats.norm.fit_loc_scale(data_set)

        for b_id in b_g:
            mol._bonds[b_id].aa_dist = aa_fit
            mol._bonds[b_id].aa_histogram = calc_hist(data_set)

    # calculating distribution of ANGLES from AA simulations


    for an_g in mol.an_dic_opt.values():
        data_set = []
        for an_id in an_g:
            # atom 1
            atom1_aa_l = map_cg2aa[mol._angles[an_id].atom1]
            sel1_s = 'bynum ' + ' or bynum '.join(atom1_aa_l)
            sel1 = u_aa.select_atoms(sel1_s)
            # atom 2
            atom2_aa_l = map_cg2aa[mol._angles[an_id].atom2]
            sel2_s = 'bynum ' + ' or bynum '.join(atom2_aa_l)
            sel2 = u_aa.select_atoms(sel2_s)
            # atom 3
            atom3_aa_l = map_cg2aa[mol._angles[an_id].atom3]
            sel3_s = 'bynum ' + ' or bynum '.join(atom3_aa_l)
            sel3 = u_aa.select_atoms(sel3_s)

            for ts in u_aa.trajectory:
                com1, com2, com3 = sel1.center_of_mass(), sel2.center_of_mass(), sel3.center_of_mass()
                vec1, vec2 = com1 - com2, com3 - com2
                data_set.append(np.rad2deg(np.arccos(np.dot(vec1, vec2)/(norm(vec1)*norm(vec2)))))

        aa_fit = stats.norm.fit_loc_scale(data_set)
       
        for an_id in an_g:
            mol._angles[an_id].aa_dist = aa_fit
            mol._angles[an_id].aa_histogram = calc_hist(data_set)

    # calculating distribution of DIHEDRALS from AA simulations

    for dih_g in mol.dih_dic_opt.values():
        data_set = []
        for dih_id in dih_g:
            # atom 1
            atom1_aa_l = map_cg2aa[mol._dihedrals[dih_id].atom1]
            sel1_s = 'bynum ' + ' or bynum '.join(atom1_aa_l)
            sel1 = u_aa.select_atoms(sel1_s)
            # atom 2
            atom2_aa_l = map_cg2aa[mol._dihedrals[dih_id].atom2]
            sel2_s = 'bynum ' + ' or bynum '.join(atom2_aa_l)
            sel2 = u_aa.select_atoms(sel2_s)
            # atom 3
            atom3_aa_l = map_cg2aa[mol._dihedrals[dih_id].atom3]
            sel3_s = 'bynum ' + ' or bynum '.join(atom3_aa_l)
            sel3 = u_aa.select_atoms(sel3_s)
            # atom 4
            atom4_aa_l = map_cg2aa[mol._dihedrals[dih_id].atom4]
            sel4_s = 'bynum ' + ' or bynum '.join(atom4_aa_l)
            sel4 = u_aa.select_atoms(sel4_s)

            for ts in u_aa.trajectory:
                com1, com2, com3, com4 = sel1.center_of_mass(), sel2.center_of_mass(), \
                                         sel3.center_of_mass(), sel4.center_of_mass()
                
                v1 = Vector3d(com1[0], com1[1], com1[2])
                v2 = Vector3d(com2[0], com2[1], com2[2])
                v3 = Vector3d(com3[0], com3[1], com3[2])
                v4 = Vector3d(com4[0], com4[1], com4[2])
                
                data_set.append(np.rad2deg(pos_dihedral(v1, v2, v3, v4)))

        aa_fit = stats.norm.fit_loc_scale(data_set)
        
        for dih_id in dih_g:
            mol._dihedrals[dih_id].aa_dist = aa_fit
            mol._dihedrals[dih_id].aa_histogram = calc_hist(data_set)

    pickle.dump(mol, open(str(options.itp) + ".p", "wb" ))

################# ANALYSIS of CG simulation #################

print(str(options.cg))

u_cg = MDAnalysis.Universe(str(options.cg) + '.gro', str(options.cg) + '.xtc')

print('Info: Loaded the CG structure and trajectory.')

# derive distributions from the current CG simulation for bonds, angles and dihedrals
for b_g in mol.b_dic_opt.values():
    data_set = []
    for b_id in b_g:
        sel1_s = 'bynum ' + str(mol._bonds[b_id].atom1)
        sel1 = u_cg.select_atoms(sel1_s)
        sel2_s = 'bynum ' + str(mol._bonds[b_id].atom2)
        sel2 = u_cg.select_atoms(sel2_s)
        for ts in u_cg.trajectory:
            com1, com2 = sel1.center_of_geometry(), sel2.center_of_geometry()
            data_set.append(norm(com1 - com2) * 0.1)
    cg_fit = stats.norm.fit_loc_scale(data_set)
    for b_id in b_g:
        mol._bonds[b_id].cg_dist = cg_fit
        mol._bonds[b_id].hist_dist.append(cg_fit)
        mol._bonds[b_id].cg_histogram = calc_hist(data_set)

for an_g in mol.an_dic_opt.values():
    data_set = []
    for an_id in an_g:
        sel1_s = 'bynum ' + str(mol._angles[an_id].atom1)
        sel1 = u_cg.select_atoms(sel1_s)
        sel2_s = 'bynum ' + str(mol._angles[an_id].atom2)
        sel2 = u_cg.select_atoms(sel2_s)
        sel3_s = 'bynum ' + str(mol._angles[an_id].atom3)
        sel3 = u_cg.select_atoms(sel3_s)
        for ts in u_cg.trajectory:
            com1, com2, com3 = sel1.center_of_geometry(), sel2.center_of_geometry(), sel3.center_of_geometry()
            vec1, vec2 = com1 - com2, com3 - com2
            data_set.append(np.rad2deg(np.arccos(np.dot(vec1, vec2)/(norm(vec1)*norm(vec2)))))
    cg_fit = stats.norm.fit_loc_scale(data_set)
    for an_id in an_g:
        mol._angles[an_id].cg_dist = cg_fit
        mol._angles[an_id].hist_dist.append(cg_fit)
        mol._angles[an_id].cg_histogram = calc_hist(data_set)

for dih_g in mol.dih_dic_opt.values():
    data_set = []
    for dih_id in dih_g:
        sel1_s = 'bynum ' + str(mol._dihedrals[dih_id].atom1)
        sel1 = u_cg.select_atoms(sel1_s)
        sel2_s = 'bynum ' + str(mol._dihedrals[dih_id].atom2)
        sel2 = u_cg.select_atoms(sel2_s)
        sel3_s = 'bynum ' + str(mol._dihedrals[dih_id].atom3)
        sel3 = u_cg.select_atoms(sel3_s)
        sel4_s = 'bynum ' + str(mol._dihedrals[dih_id].atom4)
        sel4 = u_cg.select_atoms(sel4_s)
        for ts in u_cg.trajectory:
            com1, com2, com3, com4 = sel1.center_of_geometry(), sel2.center_of_geometry(), \
                                     sel3.center_of_geometry(), sel4.center_of_geometry()

            v1 = Vector3d(com1[0], com1[1], com1[2])
            v2 = Vector3d(com2[0], com2[1], com2[2])
            v3 = Vector3d(com3[0], com3[1], com3[2])
            v4 = Vector3d(com4[0], com4[1], com4[2])
            data_set.append(np.rad2deg(pos_dihedral(v1, v2, v3, v4)))
    
    cg_fit = stats.norm.fit_loc_scale(data_set)
    for dih_id in dih_g:
        mol._dihedrals[dih_id].cg_dist = cg_fit
        mol._dihedrals[dih_id].hist_dist.append(cg_fit)
        mol._dihedrals[dih_id].cg_histogram = calc_hist(data_set)


#######################################

# plot figure 1 with the bonds    
if len(mol.b_dic_opt) > 0:
    plt_grid_size = round(np.sqrt(len(mol.b_dic_opt)) + 0.5)
    fig1 = plt.figure()
    fig1.canvas.set_window_title('Bonds')
    num_frame = 0
    for b_g in mol.b_dic_opt.keys():
        num_frame += 1
        cg_hist_x = mol._bonds[mol.b_dic_opt[b_g][0]].cg_histogram[0]
        cg_hist_pdf = mol._bonds[mol.b_dic_opt[b_g][0]].cg_histogram[1]
        aa_hist_x = mol._bonds[mol.b_dic_opt[b_g][0]].aa_histogram[0]
        aa_hist_pdf = mol._bonds[mol.b_dic_opt[b_g][0]].aa_histogram[1]
        overlap = calc_overlap(cg_hist_x, aa_hist_x, cg_hist_pdf, aa_hist_pdf)
        bond_atoms = [str(mol._bonds[mol.b_dic_opt[b_g][0]].atom1), str(mol._bonds[mol.b_dic_opt[b_g][0]].atom2)]
        print('Bond ', ' '.join(bond_atoms), ' AA max ', aa_hist_x[np.argmax(aa_hist_pdf)], ' CG max ', cg_hist_x[np.argmax(cg_hist_pdf)], '; Overlap:', overlap)

        ax = fig1.add_subplot(plt_grid_size, plt_grid_size, num_frame)
        #ax.plot(x, pdf1, color='k', lw = 2)
        ax.plot(aa_hist_x, aa_hist_pdf, color='b', lw = 1)
        ax.fill_between(aa_hist_x, aa_hist_pdf, color='b', alpha=0.25)
        ax.plot(cg_hist_x, cg_hist_pdf, color='r', lw = 1)
        ax.fill_between(cg_hist_x, cg_hist_pdf, color='r', alpha=0.25)
        ax.set_xlim([0, 1])
        #ax.set_ylim([0, np.max(np.max(aa_hist_pdf), np.max(pdf2))])
        #ax.title.set_text(str(b_g))
        #ax.legend([str(b_g)])
        plt.yticks([])
        plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [1, 2, 3, 4, 5, 6, 7])
# plot figure 2 with the angles    
if len(mol.an_dic_opt) > 0:
    plt_grid_size = round(np.sqrt(len(mol.an_dic_opt)) + 0.5)
    fig2 = plt.figure()
    fig2.canvas.set_window_title('Angles')
    num_frame = 0
    for an_g in mol.an_dic_opt.keys():
        num_frame += 1
        cg_hist_x = mol._angles[mol.an_dic_opt[an_g][0]].cg_histogram[0]
        cg_hist_pdf = mol._angles[mol.an_dic_opt[an_g][0]].cg_histogram[1]
        aa_hist_x = mol._angles[mol.an_dic_opt[an_g][0]].aa_histogram[0]
        aa_hist_pdf = mol._angles[mol.an_dic_opt[an_g][0]].aa_histogram[1]
        overlap = calc_overlap(cg_hist_x, aa_hist_x, cg_hist_pdf, aa_hist_pdf)
        angle_atoms = [str(mol._angles[mol.an_dic_opt[an_g][0]].atom1), str(mol._angles[mol.an_dic_opt[an_g][0]].atom2), str(mol._angles[mol.an_dic_opt[an_g][0]].atom3)]
        print('Angle ', ' '.join(angle_atoms), ' AA max ', aa_hist_x[np.argmax(aa_hist_pdf)], ' CG max ', cg_hist_x[np.argmax(cg_hist_pdf)], '; Overlap:', overlap)
        ax = fig2.add_subplot(plt_grid_size, plt_grid_size, num_frame)
        #ax.plot(x, pdf1, color='k', lw = 2)
        ax.plot(aa_hist_x, aa_hist_pdf, color='b', lw = 1)
        ax.fill_between(aa_hist_x, aa_hist_pdf, color='b', alpha=0.25)
        ax.plot(cg_hist_x, cg_hist_pdf, color='r', lw = 1)
        ax.fill_between(cg_hist_x, cg_hist_pdf, color='r', alpha=0.25)
        ax.set_xlim([0, 200])
        #ax.set_ylim([0, np.max(np.max(aa_hist_pdf), np.max(pdf2))])
        plt.yticks([])
        plt.xticks([0, 60, 120, 180], [])
        #ax.title.set_text(str(an_g))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.legend([str(an_g)])
    plt.xticks([0, 60, 120, 180], [0, 60, 120, 180])

# plot figure 3 with the dihedrals    
if len(mol.dih_dic_opt) > 0:
    plt_grid_size = round(np.sqrt(len(mol.dih_dic_opt)) + 0.5)
    fig3 = plt.figure()
    fig3.canvas.set_window_title('Dihedrals')
    num_frame = 0
    for dih_g in mol.dih_dic_opt.keys():
        num_frame += 1
        cg_hist_x = mol._dihedrals[mol.dih_dic_opt[dih_g][0]].cg_histogram[0]
        cg_hist_pdf = mol._dihedrals[mol.dih_dic_opt[dih_g][0]].cg_histogram[1]
        aa_hist_x = mol._dihedrals[mol.dih_dic_opt[dih_g][0]].aa_histogram[0]
        aa_hist_pdf = mol._dihedrals[mol.dih_dic_opt[dih_g][0]].aa_histogram[1]
        ax = fig3.add_subplot(plt_grid_size, plt_grid_size, num_frame)
        #ax.plot(x, pdf1, color='k', lw = 2)
        ax.plot(aa_hist_x, aa_hist_pdf, color='b', lw = 1)
        ax.fill_between(aa_hist_x, aa_hist_pdf, color='b', alpha=0.25)
        ax.plot(cg_hist_x, cg_hist_pdf, color='r', lw = 1)
        ax.fill_between(cg_hist_x, cg_hist_pdf, color='r', alpha=0.25)
        ax.set_xlim([-180, 180])
        #ax.set_ylim([0, np.max(np.max(aa_hist_pdf), np.max(pdf2))])
        plt.yticks([])
        plt.xticks([-180, -120, -60, 0, 60, 120, 180], [])
        #ax.title.set_text(str(dih_g))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.legend([str(an_g)])
    plt.xticks([-180, -120, -60, 0, 60, 120, 180], [-180, -120, -60, 0, 60, 120, 180])

plt.show()
