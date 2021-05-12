# -*- coding: utf-8 -*-
# Classes for handling of the MARTINI itp topology files

import re, numpy

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
        self.aa_dist = None # aa distribution to fit the model
        self.cg_dist = None # current cg distribution
        self.type = 1 # type of bond
        self.k = float(k) # current k
        self.l = float(l) # current l
        self.hist_dist = [] # the previous distribution
        self.hist_k = [] # keeps the history of the bond k-s
        self.hist_l = [] # keeps the history of the bond l-s
        self.aa_histogram = None
        self.conv = 0

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
        self.aa_dist = None
        self.cg_dist = None
        self.type = 2
        self.k = float(k)
        self.phi = float(phi)
        self.hist_dist = []
        self.hist_k = []
        self.hist_phi = []  
        self.aa_histogram = None
        self.conv = 0
  
    def fmt(self):
        '''formatted line for Angle object'''
        return "%3d %3d %3d %1d %5.3f %5.3f" \
            % (self.atom1, self.atom2, self.atom3, self.type, self.phi, self.k)

class Dihedral:
    '''Dihedral object'''
    def __init__(self, atom1, atom2, atom3, atom4, type, phi, k, mult, id):
        '''Init function for Angle object'''
        self.id = id
        self.atom1 = int(atom1)
        self.atom2 = int(atom2)
        self.atom3 = int(atom3)
        self.atom4 = int(atom4)
        self.aa_dist = None
        self.cg_dist = None
        self.type = int(type)
        self.k = float(k)
        self.phi = float(phi)
        self.mult = int(mult)
        self.hist_dist = []
        self.hist_k = []
        self.hist_phi = []  
        self.aa_histogram = None
        self.conv = 0
  
    def fmt(self):
        '''formatted line for Dihedral object'''
        if self.type != 2:
            return "%3d %3d %3d %3d %1d %5.3f %5.3f %1d" \
                % (self.atom1, self.atom2, self.atom3, self.atom4, self.type, self.phi, self.k, self.mult)
        if self.type == 2:
            return "%3d %3d %3d %3d %1d %5.3f %5.3f" \
                % (self.atom1, self.atom2, self.atom3, self.atom4, self.type, self.phi, self.k)


class Molecule:
    def __init__(self, fname="phc.itp"):
        '''Init function for Molecule object'''
        self.molname = ''
        self._atoms = {}
        self._bonds = []
        self._constraints = []
        self._angles = []
        self._dihedrals = []
        self._exclusions = []
        # dicts with correspondence of atomic ids to atomic names
        self._map_nm2id = {}
        self._map_id2nm = {}
        # list with element id = bond(angle)_id and element value = atomic ids
        self.b_id2atoms = []
        self.an_id2atoms = []
        self.dih_id2atoms = []
        # dict {atom_id1 atom_id2 ... : bond(angle)_id }
        self.b_atoms2id = {}
        self.an_atoms2id = {}
        self.dih_atoms2id = {}
        # dict with parameters to optimize
        self.b_dic_opt = {}
        self.an_dic_opt = {}
        self.dih_dic_opt = {}
        if fname:
            self.read_itp(fname)
            
    def clear(self):
        '''Clear all fields from Molecule'''
        for atom in self._atoms:
            del atom
        for bond in self._bonds:
            del bond
        for angle in self._angles:
            del angle        
        self._atoms, self._bonds, self._angles, self._dihedrals, self._constraints, \
        self._exclusions = {}, [], [], [], [], []
    
    def read_itp(self, fname):
        '''Load an itp file'''
        self.clear()
        
        cur_sec = '' # currently processed section of an itp file
        
        for line in open(fname, 'r').readlines():
            if line.replace(" ", "")[0] != ';' and line.replace(" ", "")[0] != "#" \
                                               and re.search('[a-zA-Z0-9]', line):
                fields = line.split()
                if line.replace(" ", "").startswith('[moleculetype]'):
                    cur_sec = 'mt'
                    continue
                if line.replace(" ", "").startswith('[atoms]'):
                    cur_sec = 'at'
                    continue
                if line.replace(" ", "").startswith('[bonds]'):
                    cur_sec = 'b'
                    continue
                if line.replace(" ", "").startswith('[angles]'):
                    cur_sec = 'an'
                    continue
                if line.replace(" ", "").startswith('[dihedrals]'):
                    cur_sec = 'd'
                    continue
                if line.replace(" ", "").startswith('[constraints]'):
                    cur_sec = 'c'
                    continue
                if line.replace(" ", "").startswith('[exclusions]'):
                    cur_sec = 'e'
                    continue
                # handling moleculetype
                if cur_sec == 'mt': self.molname = line.split()[0]
                # handling atoms
                if cur_sec == 'at':
                    if fields[0] in self._atoms:
                        print "A duplicated atom_id ", fields[0], " found. Check topology."
                        exit()
                    else:
                        self._atoms[int(fields[0])] = Atom(fields[0], fields[1], fields[2], fields[3], \
                                                fields[4], fields[5], fields[6])
                        self._map_nm2id[str(fields[4])] = int(fields[0])
                        self._map_id2nm[int(fields[0])] = str(fields[4])
                # handling bonds
                if cur_sec == 'b':
                    if (int(fields[0]), int(fields[1])) in self._bonds:
                        print "A duplicated bond ", fields[0], " ", fields[1], " found. Check topology."
                        exit()
                    else:
                        self._bonds.append(Bond(fields[0], fields[1], fields[3], fields[4], len(self._bonds)))
                        self.b_id2atoms.append((int(fields[0]), int(fields[1])))
                        self.b_atoms2id[(int(fields[0]), int(fields[1]))] = len(self._bonds) - 1
                # handling angles
                if cur_sec == 'an':
                    if (int(fields[0]), int(fields[1]), int(fields[3])) in self._angles:
                        print "A duplicated angle ", fields[0], " ", fields[1], " found. Check topology."
                        exit()
                    else:
                        self._angles.append(Angle(fields[0], fields[1], fields[2], \
                                                  fields[4], fields[5], len(self._angles)))
                        self.an_id2atoms.append((int(fields[0]), int(fields[1]), int(fields[2])))
                        self.an_atoms2id[(int(fields[0]), int(fields[1]), int(fields[2]))] = len(self._angles) - 1

                # handling dihedrals
                if cur_sec == 'd':
                    if (int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])) in self._dihedrals:
                        print "A duplicated dihedral ", fields[0], " ", fields[1], " ", fields[2],  " ", fields[3], " found. Check topology."
                        exit()
                    else:
                        if fields[4] != '2':
                            self._dihedrals.append(Dihedral(fields[0], fields[1], fields[2], fields[3],\
                                                  fields[4], fields[5], fields[6], fields[7], len(self._dihedrals)))
                            self.dih_id2atoms.append((int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])))
                            self.dih_atoms2id[(int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]))] = len(self._dihedrals) - 1
                        if fields[4] == '2':
                            self._dihedrals.append(Dihedral(fields[0], fields[1], fields[2], fields[3],\
                                                  fields[4], fields[5], fields[6], 0, len(self._dihedrals)))
                            self.dih_id2atoms.append((int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])))
                            self.dih_atoms2id[(int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]))] = len(self._dihedrals) - 1

                # handling constraints
                if cur_sec == 'c':
                    self._constraints.append(line.rstrip('\n').rstrip('\r'))
                # handling exclusions
                if cur_sec == 'e':
                    self._exclusions.append(line.rstrip('\n').rstrip('\r'))  
                    
    def write_itp(self, fname):
        '''Write an itp file'''
        out = ['; AUTOMATICALLY GENERATED TOPOLOGY FILE AT ITERATION']
        out.append(' [ moleculetype ] ')
        out.append('; molname         nrexcl')
        out.append(str(self.molname) + '   1')
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
        