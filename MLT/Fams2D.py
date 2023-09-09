#%%
import numpy as np
import os
import math
import time
import random
import ctypes

from FamsIf import FamsIf

FIRST_HASHTABLE = 256
SECOND_HASHTABLE = 16

FAMS_MAX_K = 70
FAMS_MAX_L = 500
FAMS_FLOAT_SHIFT = 100000.0
FAMS_ALPHA = 1.0
FAMS_MAXITER = 100

FAMS_PRUNE_WINDOW = 3000
FAMS_PRUNE_MINN = 40
FAMS_PRUNE_MAXP = 10000
FAMS_PRUNE_HDIV = 1
FAMS_PRUNE_MAXM = 100

RAND_MAX = 0x7fff
USE_C_RANDOM = 1
USE_TIME_SEED = 0
DEFAULT_SEED = 100

#%%
class fams_point:
    '''
    points for fams
    mutable: "=" will get its reference
    '''
    def __init__(self):
        self.data_ = np.zeros(0, dtype=np.ushort)
        self.usedFlag_ = np.ushort(0)
        self.window_ = np.int32(0)
        self.weightdp2_ = np.float64(0)

class fams_hash_entry:
    ''' first hash entry '''
    def __init__(self):
        self.whichCut_ = np.short(0)
        self.which2_ = np.int32(0)
        self.pt_ = fams_point()

class fams_hash_entry2:
    ''' second hash entry '''
    def __init__(self):
        self.whichCut_ = np.int32(0)
        self.dp_ = None

class fams_cut:
    ''' fams cuts '''
    def __init__(self):
        self.which_ = np.uint16(0)
        self.where_ = np.uint16(0)

class fams_res_cont:
    ''' fams result '''
    def __init__(self, n):
        self.nel_ = 0
        self.vec_ = [fams_point()] * n

    def push_back(self, in_el: fams_point):
        self.vec_[self.nel_] = in_el
        self.nel_ += 1

    def clear(self):
        self.nel_ = 0
    
    def size(self):
        return self.nel_

class Cfloat:
    ''' float calculation in cpp '''
    def __init__(self):
        ''' init float calculation '''
        cpp_prog = 'c_tools\c_float\c_float.dll'
        self.cpp_lib = ctypes.CDLL(cpp_prog)
        self.cpp_lib.c_pow.restype = ctypes.c_float
        self.cpp_lib.c_div.restype = ctypes.c_double
        self.cpp_lib.c_divUshort.restype = ctypes.c_ushort
        self.cpp_lib.c_mul.restype = ctypes.c_double

    def cpow(self, a, b):
        ''' a ** b '''
        a = ctypes.c_double(a)
        b = ctypes.c_double(b)
        res = self.cpp_lib.c_pow(a, b)
        return res

    def cdiv(self, a, b):
        ''' a / b '''
        a = ctypes.c_double(a)
        b = ctypes.c_double(b)
        res = self.cpp_lib.c_div(a, b)
        return res
    
    def cdivUshort(self, a, b):
        ''' (ushort) a / b '''
        a = ctypes.c_double(a)
        b = ctypes.c_double(b)
        res = self.cpp_lib.c_divUshort(a, b)
        return res

    def cmul(self, a, b):
        ''' a * b '''
        a = ctypes.c_double(a)
        b = ctypes.c_double(b)
        res = self.cpp_lib.c_mul(a, b)
        return res

class tmpMode:
    '''
    temp mode, record the reference of (self.d_points) data in self.modes array, and record the reference of dp_ in HT2 matrix
    '''
    def __init__(self):
        self.pos = 0            # the position in self.modes
        self.emptyflag = 1      # 1: position not assigned
        self.i = []             # rows in HT2, more than one points will be stored
        self.j = []             # columns in HT2

class FAMS2D:
    ''' implementation of fams '''
    def __init__(self, famsif, no_lsh=False):
        ''' init random seed '''
        if USE_C_RANDOM:
            # call cpp random
            cpp_prog = 'c_tools\c_random\c_random.dll'
            self.cpp_lib = ctypes.CDLL(cpp_prog)
            self.cpp_lib.c_rand.restype = ctypes.c_int
            if USE_TIME_SEED:
                pass
            else:
                self.cpp_lib.c_srand(DEFAULT_SEED)
        else:
            t = int(time.time())
            random.seed(t)

        ''' init c cloat '''
        self.cf = Cfloat()

        ''' algorithm '''
        self.hasPoints_ = False
        self.nsel_ = int(0)
        self.psel_ = np.zeros(0, dtype=np.int32)

        ''' hash '''
        self.noLSH_ = no_lsh
        self.M_ = int(0)
        self.M2_ = int(0)
        self.Bs = int(FIRST_HASHTABLE)       # 256
        self.Bs2 = int(SECOND_HASHTABLE)     # 16
        self.hashCoeffs_ = None

        ''' data '''
        self.data_ = np.zeros(0, dtype=np.ushort)
        self.rr_ = np.zeros(0, dtype=np.float64)
        self.n_points = int(0)   # number of points per dimension
        self.d_points = int(0)   # number of dimensions
        self.points_ = None
        self.minVal_ = float(0)
        self.maxVal_ = float(0)

        ''' modes '''
        self.prunedmodes_ = np.zeros(0, dtype=np.ushort)
        self.nprunedmodes_ = np.zeros(0, dtype=np.int32)
        self.modes_ = np.zeros(0, dtype=np.ushort)
        self.hmodes_  = np.zeros(0, dtype=np.uint32)
        self.npm_ = int(0)

        ''' interface '''
        self.famsif = famsif
        self.K = famsif.famsK       # 24
        self.L = famsif.famsL       # 35
        self.k_neigh = famsif.famsk # 100

        ''' file '''
        self.famsDir = famsif.famsDir
        self.famsFile = os.path.join(self.famsDir, 'fams.txt')
        self.pilotFile = os.path.join(self.famsDir, f"pilot_{self.k_neigh}_fams_py.txt")
        self.outFile = os.path.join(self.famsDir, 'out_fams_py.txt')
        self.modesFile = os.path.join(self.famsDir, 'modes_fams_py.txt')
        self.debug_modesFile = os.path.join(self.famsDir, 'py_modes.txt')   # load modes, skip dofams

        ''' temporary '''
        self.t_cut_res_ = np.zeros([FAMS_MAX_L, FAMS_MAX_K], dtype=np.int32)
        self.t_old_cut_res_ = np.zeros([FAMS_MAX_L, FAMS_MAX_K], dtype=np.int32)
        self.t_m_ = np.zeros(FAMS_MAX_L, dtype=np.int32)
        self.t_old_m_ = np.zeros(FAMS_MAX_L, dtype=np.int32)
        self.t_m2_ = np.zeros(FAMS_MAX_L, dtype=np.int32)
        self.t_hjump_ = np.zeros(FAMS_MAX_L, dtype=np.int32)
        self.nnres1_ = np.uint16(0)
        self.nnres2_ = np.uint16(0)

        ''' log '''
        self.log_enable = False
        self.debug = False

    def __rand(self):
        ''' general random method '''
        if USE_C_RANDOM:
            return self.cpp_lib.c_rand()
        else:
            self.bglog('no random method')

    def __cpy(self, dst, src, l):
        ''' copy the data with a certain length '''
        for i in range(l):
            dst[i] = src[i]

    def loadPoints(self):
        ''' read in data '''
        first, i = 0, 0
        filepath = self.famsFile
        with open(filepath, 'r') as f:
            for line in f:
                tmp = line.split()
                if first == 0:
                    self.n_points, self.d_points = int(tmp[0]), int(tmp[1])
                    pttemp = np.zeros(self.n_points * self.d_points, dtype=np.ushort)
                    first = 1
                else:
                    for t in tmp:
                        pttemp[i] = t
                        i += 1

        self.minVal_, self.maxVal_ = min(pttemp), max(pttemp)
        deltaVal = self.maxVal_ - self.minVal_
        if deltaVal == 0:
            deltaVal = 1

        self.data_ = np.zeros(self.n_points * self.d_points, dtype=np.ushort)
        for i in range(self.n_points * self.d_points):
            self.data_[i] = (65535.0 * (pttemp[i] - self.minVal_) / deltaVal).astype(np.ushort)
        self.hasPoints_ = True

        # points_ array: self.n_points * self.d_points
        self.points_ = [fams_point() for i in range(self.n_points)]
        for (i,j) in zip(range(self.n_points), range(0,len(self.data_),self.d_points)):
            self.points_[i].data_ = self.data_[j : j + self.d_points]

    def __CleanSelected(self):
        if self.nsel_ > 0:
            self.nsel_ = int(0)
            self.psel_ = np.zeros(0, dtype=np.int32)
            self.modes_ = np.zeros(0, dtype=np.ushort)
            self.hmodes_  = np.zeros(0, dtype=np.uint32)

    def __SelectMsPoints(self, percent, jump):
        ''' default: percent=0.0 jump=1 '''
        if self.hasPoints_ == False:
            self.bglog("Load points first")
            return

        # when percent=0.0, and jump=1
        tsel = int(math.ceil(self.n_points / (jump + 0.0)))
        if tsel != self.nsel_:
            self.__CleanSelected()
            self.nsel_ = tsel
            self.psel_ = np.zeros(self.nsel_, dtype=np.int32)
            self.modes_ = np.zeros(self.nsel_ * self.d_points, dtype=np.ushort)
            self.hmodes_ = np.zeros(self.nsel_, dtype=np.uint32)
        for i in range(self.nsel_):
            self.psel_[i] = i * jump

    def __GetPrime(self, minp):
        ''' return a prime number greater than minp '''
        i = minp if minp % 2 == 1 else minp + 1
        while True:
            sqt = int(math.sqrt(i))
            if i % 2 == 0:
                i += 2
                continue
            is_prime = True
            for j in range(3, sqt+1, 2):
                if i % j == 0:
                    is_prime = False
                    break
            if is_prime:
                return i
            i += 2

    def __drand48(self):
        ''' get random number '''
        num = self.__rand()
        return (num * 1.0 / RAND_MAX)

    def __MakeCuts(self, cuts):
        ''' make cuts in cuts table '''
        for i in range(self.L):
            ''' cut in L rows: MakeCutL '''
            n1 = int(self.K // (1.0 * self.d_points))    # floor
            ncu = 0
            for m in range(self.d_points):
                for n in range(n1):
                    cuts[i][ncu].which_ = m
                    w = int(min( int(self.__drand48() * self.n_points), self.n_points - 1 ))
                    cuts[i][ncu].where_ = self.points_[w].data_[m]
                    ncu += 1

            which = np.zeros(self.d_points, dtype=np.int32)
            while ncu < self.K:
                wh = int(min( int(self.__drand48() * self.d_points), self.d_points - 1 ))
                if which[wh] != 0:
                    continue
                which[wh] = 1
                w = int(min( int(self.__drand48() * self.n_points), self.n_points - 1 ))
                cuts[i][ncu].which_ = wh
                cuts[i][ncu].where_ = self.points_[w].data_[wh]
                ncu += 1

    def __InitHash(self, nk):
        ''' populate hashCoeffs_ with random numbers '''
        self.hashCoeffs_ = np.zeros(nk, dtype=np.int32)
        for i in range(nk):
            self.hashCoeffs_[i] = self.__rand()

    def __EvalCutRes(self, in_pt, in_part, in_cut_res):
        # produce the boolean vector of a data point with a partition
        if isinstance(in_pt, fams_point):
            for i in range(self.K):
                in_cut_res[i] = bool(in_pt.data_[in_part[i].which_] >= in_part[i].where_)
        # produce the boolean vector of a ms data point with a partition
        else:
            for i in range(self.K):
                in_cut_res[i] = bool(in_pt[in_part[i].which_] >= in_part[i].where_)

    def __HashFunction(self, cutVals, whichPartition, kk, M=0, getdoubleHash=False):
        '''
        compute the hash key and and the double hash key if needed 
        It is possible to give M the the size of the hash table and 
        hjump for the double hash key 
        '''
        res = whichPartition
        for i in range(kk):
            res += cutVals[i] * self.hashCoeffs_[i]
        if M != 0:
            res = tmp = abs(res)
            res %= M
            if getdoubleHash:
                hjump = tmp % (M - 1) + 1
                return (res, hjump)
        return res

    def __AddDataToHash(self, HT, hs, pt, where, Bs, M, which, which2, hjump):
        ''' add a point to the LSH hash table using double hashing '''
        nw = 0
        while True:
            nw += 1
            if (nw > M):
                self.bglog("LSH hash table overflow exiting")
                raise SystemExit
            if hs[where] == Bs:
                where = (where + hjump) % M
                continue
            HT[where][hs[where]].pt_ = pt
            HT[where][hs[where]].whichCut_ = which
            HT[where][hs[where]].which2_ = which2
            hs[where] += 1
            break

    def __LoadBandwidths(self):
        ''' load pilot file, minor difference with cpp on output percision '''
        if not os.path.exists(self.pilotFile):
            return False
        else:
            with open(self.pilotFile, 'rb') as f:
                n = int(f.readline().strip())
                if n != self.n_points:
                    return False

                deltaVal = float(self.maxVal_ - self.minVal_)
                for i in range(self.n_points):
                    bw = f.readline().strip()
                    self.points_[i].window_ = np.int32(bw)
                    # bw = float(f.readline().strip())
                    # self.points_[i].window_ = int(65535.0 * (bw) / deltaVal)

                return True

    def __CompareCutRes(self, in_cr1, in_cr2):
        ''' Compare a pair of L binary vectors
        return true when different '''
        for i in range(self.L):
            for j in range(self.K):
                if in_cr1[i][j] != in_cr2[i][j]:
                    return True
        return False
    
    def __AddDataToRes(self, HT, hs, res, where, Bs, M, which, nnres, which2, hjump):
        ''' perform a query to one partition and retreive all the points in the cell '''
        while True:
            for uu in range(hs[where]):
                if (HT[where][uu].whichCut_ == which) and \
                    (HT[where][uu].which2_ == which2):
                    if HT[where][uu].pt_.usedFlag_ != nnres:
                        res.push_back(HT[where][uu].pt_)
                        HT[where][uu].pt_.usedFlag_ = nnres
            if hs[where] < Bs:
                break
            else:
                where = (where + hjump) % M

    def __GetNearestNeighbours(self, who: fams_point, HT, hs, cuts, res: fams_res_cont, num_l):
        ''' perform an LSH query '''
        for i in range(self.L):
            self.__EvalCutRes(who, cuts[i], self.t_cut_res_[i])
        # if t_cut_res_ and t_old_cut_res_ not same, return
        if self.__CompareCutRes(self.t_cut_res_, self.t_old_cut_res_) == False:
            return

        self.t_old_cut_res_ = np.copy(self.t_cut_res_)
        res.clear()
        self.nnres1_ += 1

        for i in range(self.L):
            m, hjump = self.__HashFunction(self.t_cut_res_[i], i, self.K, self.M_, getdoubleHash=True)
            m2 = self.__HashFunction(self.t_cut_res_[i][1:], i, self.K - 1)
            self.__AddDataToRes(HT, hs, res, m, self.Bs, self.M_, i, self.nnres1_, m2, hjump)
            num_l[i] = res.nel_

    def __DistL1(self, in_pt1: fams_point, in_pt2: fams_point):
        ''' distance in L1 between two data elements '''
        in_res = 0
        for i in range(len(in_pt1.data_)):
            in_res += int(abs(int(in_pt1.data_[i]) - int(in_pt2.data_[i])))
        return in_res

    def __SaveBandwidths(self):
        ''' save bandwidth into pilotFile, minor difference with cpp on output percision '''
        with open(self.pilotFile, 'w') as f:
            f.write(f'{self.n_points}\n')
            deltaVal = np.float64(self.maxVal_ - self.minVal_)
            for i in range(self.n_points):
                bw = self.points_[i].window_
                f.write(f'{bw}\n')
                # # bw = self.points_[i].window_ * deltaVal / 65535.0
                # a = self.cf.cmul(self.points_[i].window_, deltaVal)
                # bw = float(self.cf.cdiv(a, 65535.0))
                # f.write(f'{bw:g}\n')

    def __ComputePilot(self, HT, hs, cuts):
        ''' compute the pilot h_i's for the data points '''
        win_j = 10
        wjd = win_j * self.d_points
        max_win = 7000
        num_l = np.zeros(1000, dtype=np.int32)
        res = fams_res_cont(self.n_points)

        start_time = time.time()
        if self.__LoadBandwidths() == True:
            self.bglog('load bandwidths...')
        else:
            self.bglog('compute bandwidths...')
            for j in range(self.n_points):
                numn = 0
                numns = np.zeros(int(max_win / win_j), dtype=np.int32)

                # using LSH
                self.__GetNearestNeighbours(self.points_[j], HT, hs, cuts, res, num_l)
                for i in range(res.nel_):
                    pt = res.vec_[i]
                    nn = int(self.__DistL1(self.points_[j], pt) / wjd)
                    if nn < int(max_win / win_j):
                        numns[nn] += 1

                for nn in range(int(max_win / win_j)):
                    numn += numns[nn]
                    if numn > self.k_neigh:
                        break
                # adjust the index
                if nn == int(max_win / win_j) - 1:
                    nn += 2
                else:
                    nn += 1
                self.points_[j].window_ = np.int32(nn * wjd)
            self.__SaveBandwidths()

        end_time = time.time()
        self.bglog(f'__ComputePilot: {end_time - start_time:.2f}s')

        for j in range(self.n_points):
            a = self.cf.cdiv(FAMS_FLOAT_SHIFT, self.points_[j].window_)
            b = self.cf.cmul(self.d_points + 2, FAMS_ALPHA)
            tmp = self.cf.cpow(a, b)
            self.points_[j].weightdp2_ = np.float64(tmp)

    def __NotEq(self, in_d1, in_d2, length):
        ''' if not equal, return true '''
        for i in range(length):
            if int(in_d1[i]) != int(in_d2[i]):
                return True
        return False
    
    def __FindInHash(self, HT2, hs2, where, which, M2, hjump):
        ''' perform a query on the second hash table '''
        nw = 0
        while True:
            nw += 1
            if nw > M2:
                self.bglog('Hash Table2 full')
                raise SystemExit
            for uu in range(hs2[where]):
                if HT2[where][uu].whichCut_ == which:
                    return HT2[where][uu].dp_
            if hs2[where] < self.Bs2:
                break
            where = int((where + hjump) % M2)
        return None

    def __InsertIntoHash(self, HT2, hs2, where, which, solution: tmpMode, M, hjump):
        '''
        Insert an mean-shift result into a second hash table so when another mean shift computation is 
        performed about the same C_intersection region, the result can be retreived without further computation
        '''
        nw = 0
        while True:
            nw += 1
            if nw == M:
                self.bglog('Hash Table2 full')
                raise SystemExit
            if hs2[where] == self.Bs2:
                where = int((where + hjump) % M)
                continue

            if solution.emptyflag == 1:
                HT2[where][hs2[where]].dp_ = 1
            else:
                p = solution.pos
                HT2[where][hs2[where]].dp_ = self.modes_[p:p+self.d_points]
            solution.i.append(where)
            solution.j.append(hs2[where])

            HT2[where][hs2[where]].whichCut_ = which
            hs2[where] += 1
            break

    def __GetNearestNeighbours2H(self, who, HT, hs, cuts, res: fams_res_cont, solution: tmpMode, HT2, hs2):
        ''' perform an LSH query using in addition the second hash table, if failed, return None '''
        for i in range(self.L):
            self.__EvalCutRes(who, cuts[i], self.t_cut_res_[i])
            self.t_m_[i], self.t_hjump_[i] = self.__HashFunction(self.t_cut_res_[i], i, self.K, self.M_, getdoubleHash=True)
            self.t_m2_[i] = self.__HashFunction(self.t_cut_res_[i][1:], i, self.K - 1)

        # FAMS_DO_SPEEDUP
        hf, hjump2 = self.__HashFunction(self.t_m_, 0, self.L, self.M2_, getdoubleHash=True)
        hf2 = self.__HashFunction(self.t_m_[int(self.L/2)-1:], 0, int(self.L/2))
        old_sol = self.__FindInHash(HT2, hs2, hf, hf2, self.M2_, hjump2)
        if isinstance(old_sol, np.ndarray):
            return old_sol
        if old_sol == None:
            self.__InsertIntoHash(HT2, hs2, hf, hf2, solution, self.M2_, hjump2)

        # return when same
        if self.__NotEq(self.t_m_, self.t_old_m_, self.L) == False:
            return None
        self.__cpy(self.t_old_m_, self.t_m_, self.L)
        res.clear()
        self.nnres2_ += 1

        for i in range(self.L):
            self.__AddDataToRes(HT, hs, res, self.t_m_[i], self.Bs, self.M_, i, self.nnres2_, self.t_m2_[i], self.t_hjump_[i])

        return None

    def __DistL1Data(self, in_d1, in_pt2: fams_point, in_dist):
        ''' computes the distance if it is less than dist into in_dist, and return the distance of inputs '''
        in_res = np.float64(0)
        for i in range(self.d_points):
            if not (in_res < in_dist):
                break
            in_res += np.float64(abs(np.float64(in_d1[i]) - np.float64(in_pt2.data_[i])))
        return (in_res < in_dist, in_res)

    def __DoMeanShiftAdaptiveIteration(self, res: fams_res_cont, old, ret: np.ndarray):
        ''' perform an FAMS iteration '''
        total_weight = np.float64(0)
        nel = res.nel_
        hmdist = np.inf
        self.rr_ = np.zeros(self.d_points, dtype=np.float64)

        for i in range(nel):
            # using LSH
            ptp: fams_point = res.vec_[i]
            r, dist = self.__DistL1Data(old, ptp, ptp.window_)
            if r == True:
                # w = ptp.weightdp2_ * (1.0 - dist / ptp.window_) ** 2
                tmp = self.cf.cdiv(dist, ptp.window_)
                tmp = self.cf.cmul(np.float64(1.0 - tmp), np.float64(1.0 - tmp))
                w = self.cf.cmul(ptp.weightdp2_, tmp)
                total_weight += np.float64(w)
                for j in range(self.d_points):
                    tmp = self.cf.cmul(ptp.data_[j], w)
                    self.rr_[j] += np.float64(tmp)
                if dist < hmdist:
                    hmdist = dist
                    crtH = ptp.window_

        if total_weight == 0:
            return 0

        for i in range(self.d_points):
            tmp = self.cf.cdivUshort(self.rr_[i], total_weight)
            ret[i] = np.ushort(tmp)

        return crtH
    
    def __updateHT2(self, t: tmpMode, HT2, data):
        ''' update data in HT2 according to the position in t '''
        assert len(t.i) == len(t.j)

        for (i, j) in zip(t.i, t.j):
            HT2[i][j].dp_ = np.zeros(len(data), dtype=np.ushort)
            self.__cpy(HT2[i][j].dp_, data, len(data))

    def __bgISort(self, ra, ira):
        ''' sorting in ascending order and return the sorted index array '''
        n = len(ra)
        if n < 2:
            return
        
        l = (n >> 1) + 1
        ir = n

        while True:
            if l > 1:
                l -= 1
                irra = ira[l - 1]
                rra = ra[l - 1]
            else:
                irra = ira[ir - 1]
                rra = ra[ir - 1]

                ira[ir - 1] = ira[0]
                ra[ir - 1] = ra[0]

                ir -= 1
                if ir == 1:
                    ira[0] = irra
                    ra[0] = rra
                    break

            i = l
            j = l + l

            while j <= ir:
                if j < ir and ra[j - 1] < ra[j]:
                    j += 1
                if rra < ra[j - 1]:
                    ira[i - 1] = ira[j - 1]
                    ra[i - 1] = ra[j - 1]
                    i = j
                    j <<= 1
                else:
                    j = ir + 1

            ira[i - 1] = irra
            ra[i - 1] = rra

    def __PruneModes(self, hprune, npmin):
        ''' join the modes calculated in previous steps '''
        if self.nsel_ < 1:
            raise SystemExit

        tmp = self.cf.cdiv(self.nsel_, FAMS_PRUNE_MAXP)
        jm = np.int32(math.ceil(tmp))   # 1
        self.bglog(f'Join Modes with adaptive h/{int(pow(2, FAMS_PRUNE_HDIV))}, min pt={npmin}, jump={jm}')
        self.bglog('    pass 1')

        hprune = self.d_points
        invalidm = np.zeros(self.nsel_, dtype=np.uint8)
        mcount = np.zeros(self.nsel_, dtype=np.int32)
        cmodes = np.zeros(self.d_points * self.nsel_, dtype=np.float32)

        # set first mode
        for i in range(self.d_points):
            cmodes[i] = self.modes_[i]
        mcount[0] = 1
        maxm = 1

        bar_proc = int(FAMS_PRUNE_MAXP / 10)
        for cm in range(1, self.nsel_, jm):     # jm = 1
            if cm % bar_proc == 0:
                self.bglog('.', end='')

            pmodes_i = 0 + cm * self.d_points     # index in modes_[]
            cminDist = np.float64(self.d_points * 1e7)
            iminDist = -1

            # compute closest mode
            for cref in range(maxm):
                if invalidm[cref]:
                    continue

                cdist = np.float64(0)
                ctmodes_i = 0 + cref * self.d_points      # index in cmodes[]
                for cd in range(self.d_points):
                    tmp = self.cf.cdiv(cmodes[ctmodes_i + cd], mcount[cref])      # ctmodes_i -> index in cmodes
                    cdist += np.float64(math.fabs(tmp - self.modes_[pmodes_i + cd]))      # pmodes_i -> index in modes_

                if cdist < cminDist:
                    cminDist = cdist
                    iminDist = cref

            # join
            hprune = self.hmodes_[cm] >> FAMS_PRUNE_HDIV
            if cminDist < hprune:
                # old mode, just add
                for cd in range(self.d_points):
                    cmodes[iminDist * self.d_points + cd] += self.modes_[pmodes_i + cd]
                mcount[iminDist] += 1
            else:
                # new mode, create
                for cd in range(self.d_points):
                    cmodes[maxm * self.d_points + cd] = self.modes_[pmodes_i + cd]
                mcount[maxm] = 1
                maxm += 1

            # check for valid modes
            if maxm > 2000:
                for i in range(maxm):
                    if mcount[i] < 3:
                        invalidm[i] = 1

        self.bglog('done')
        self.bglog('    pass 2')

        stemp = np.zeros(maxm, dtype=np.int32)
        istemp = np.arange(maxm, dtype=np.int32)
        for i in range(maxm):
            stemp[i] = mcount[i]
        self.__bgISort(stemp, istemp)

        # find number of relevant modes
        nrel = 1
        i = maxm - 2
        while i >= 0:
            if stemp[i] >= npmin:
                nrel += 1
            else:
                break
            i -= 1

        if nrel > FAMS_PRUNE_MAXM:
            nrel = FAMS_PRUNE_MAXM
        
        # rearange only relevant modes
        mcount2 = np.zeros(nrel, dtype=np.int32)
        cmodes2 = np.zeros(self.d_points * nrel, dtype=np.float32)

        for i in range(nrel):
            cm = istemp[maxm - i - 1]
            mcount2[i] = np.float32(mcount[cm])
            self.__cpy(cmodes2[i * self.d_points:], cmodes[cm * self.d_points:], self.d_points)

        mcount = np.zeros(self.nsel_, dtype=np.int32)
        mcount[0] = 1
        for i in range(1, self.nsel_, jm):
            mcount[i] = 1
        maxm = nrel

        bar_proc = int(self.nsel_ / 10)
        for cm in range(1, self.nsel_):
            if cm % self.nsel_ == 0:
                self.bglog('.')
            if mcount[cm]:
                continue

            pmodes_i = 0 + cm * self.d_points     # index in modes_[]
            cminDist = np.float64(self.d_points * 1e7)
            iminDist = -1

            # compute closest mode
            for cref in range(maxm):
                cdist = np.float64(0)
                ctmodes_i = 0 + cref * self.d_points      # index in cmodes2[]
                for cd in range(self.d_points):
                    tmp = self.cf.cdiv(cmodes[ctmodes_i + cd], mcount2[cref])      #     ctmodes_i -> index in cmodes
                    cdist += np.float64(math.fabs(tmp - self.modes_[pmodes_i + cd]))      # pmodes_i -> index in modes_
                if cdist < cminDist:
                    cminDist = cdist
                    iminDist = cref

            # join
            hprune = self.hmodes_[cm] >> FAMS_PRUNE_HDIV
            if cminDist < hprune:
                # old mode, just add
                for cd in range(self.d_points):
                    cmodes2[iminDist * self.d_points + cd] += self.modes_[pmodes_i + cd]
                mcount2[iminDist] += 1
            else:
                # new mode, but discard in second pass
                pass

        # put the modes in the order of importance (count)
        stemp = np.zeros(maxm, dtype=np.int32)
        istemp = np.arange(maxm, dtype=np.int32)
        for i in range(maxm):
            stemp[i] = mcount2[i]
        self.__bgISort(stemp, istemp)

        # find number of relevant modes
        nrel = 1
        i = maxm - 2
        while i >= 0:
            if stemp[i] >= npmin:
                nrel += 1
            else:
                break
            i -= 1

        self.prunedmodes_ = np.zeros(self.d_points * nrel, dtype=np.ushort)
        self.nprunedmodes_ = np.zeros(nrel, np.int32)
        self.npm_ = nrel
        cpm_i = 0   # index in prunedmodes_[]

        for i in range(self.npm_):
            self.nprunedmodes_[i] = stemp[maxm - i - 1]
            cm = istemp[maxm - i - 1]
            for cd in range(self.d_points):
                tmp = self.cf.cdivUshort(cmodes2[cm * self.d_points + cd], mcount2[cm])
                self.prunedmodes_[cpm_i] = np.ushort(tmp)
                cpm_i += 1

        self.bglog('done')

    def __loadModes(self):
        ''' load modes_ and hmodes_ from file '''
        with open(self.debug_modesFile, 'r') as f:
            f.readline()

            # hmodes_
            i = 0
            flag = 0
            for l in f.readlines():
                if 'modes_ start' in l:
                    flag = 1
                    i = 0
                    continue

                if flag == 0:
                    # hmodes_
                    data = l.split()
                    for d in data:
                        self.hmodes_[i] = np.uint32(d)
                        i += 1
                else:
                    # modes_
                    data = l.split()
                    for d in data:
                        self.modes_[i] = np.ushort(d)
                        i += 1

    def __DoFAMS(self, HT, hs, cuts, HT2, hs2):
        ''' perform FAMS starting from a subset of the data points '''
        if self.debug:
            self.__loadModes()
            return

        currentpt = fams_point()
        res = fams_res_cont(self.n_points)
        oldMean = np.zeros(self.d_points, dtype=np.ushort)
        crtMean = np.zeros(self.d_points, dtype=np.ushort)
        bar_proc = int(self.nsel_ / 10)
        tMode = [tmpMode() for i in range(self.n_points)]

        self.bglog('Start MS iterations', end='')
        start_time = time.time()
        for j in range(self.nsel_):
            # processing
            if j % bar_proc == 0:
                self.bglog('.', end='')

            who = self.psel_[j]
            currentpt: fams_point = self.points_[who]
            self.__cpy(crtMean, currentpt.data_, len(currentpt.data_))
            crtH = j    # crtH: reference of self.hmodes_[j]
            self.hmodes_[crtH] = currentpt.window_
            tMode[j].emptyflag = 1  # no data is assigned to tMode[j]

            for i in range(FAMS_MAXITER):
                if self.__NotEq(oldMean, crtMean, len(oldMean)) == False:
                    break

                # using LSH
                sol = self.__GetNearestNeighbours2H(crtMean, HT, hs, cuts, res, tMode[j], HT2, hs2)
                # sol =>  1. None: pass
                #         2. 1                 -\
                #         3. array, [d_points] --> update modes_ and HT2
                if isinstance(sol, int) and sol == 1:
                    # sol => 1
                    tMode[j].emptyflag = 0
                    tMode[j].pos = p = j * self.d_points
                    self.__cpy(self.modes_[p:p+self.d_points], crtMean, len(crtMean))
                    self.__updateHT2(tMode[j], HT2, crtMean)
                elif isinstance(sol, np.ndarray):
                    # sol => array, [d_points]
                    tMode[j].emptyflag = 0
                    tMode[j].pos = p = j * self.d_points
                    self.__cpy(self.modes_[p:p+self.d_points], sol, len(sol))
                    self.__updateHT2(tMode[j], HT2, sol)
                    break
                else:
                    # sol => None
                    pass

                self.__cpy(oldMean, crtMean, len(crtMean))

                newH = self.__DoMeanShiftAdaptiveIteration(res, oldMean, crtMean)
                if newH == 0:
                    self.__cpy(crtMean, oldMean, len(oldMean))
                    break
                self.hmodes_[crtH] = newH

            if tMode[j].emptyflag == 1:     # update modes_ and HT2
                tMode[j].emptyflag = 0
                tMode[j].pos = p = j * self.d_points
                self.__cpy(self.modes_[p:p+self.d_points], crtMean, len(crtMean))
                self.__updateHT2(tMode[j], HT2, crtMean)

        self.bglog('done')
        end_time = time.time()
        self.bglog(f'__DoFAMS: {end_time - start_time:.2f}s')

    def runFams(self, percent=0.0, jump=1, width=float(-1)):
        ''' do the fams algorithm '''
        if self.hasPoints_ == False:
            self.bglog("Load points first")
            return

        # select points for algorithm
        self.__SelectMsPoints(percent, jump)

        # make partitions
        cuts = [[fams_cut() for x in range(FAMS_MAX_K)] for y in range(self.L)]
        # pop first 20 random numbers
        for i in range(20):
            self.__rand()
        self.__MakeCuts(cuts)

        # hash init
        self.__InitHash(self.K + self.L)

        # allocate array for the hash table
        ex1 = int(3 * self.n_points * self.L / self.Bs)
        ex2 = int(self.nsel_ * 20 * 3 / self.Bs2)
        self.M_ = self.__GetPrime(ex1)      # 4111
        self.M2_ = self.__GetPrime(ex2)     # 37501
        HT = [[fams_hash_entry() for x in range(self.Bs)] for y in range(self.M_)]
        hs = [0 for x in range(self.M_)]
        HT2 = [[fams_hash_entry2() for x in range(self.Bs2)] for y in range(self.M2_)]
        hs2 = [0 for x in range(self.M2_)]
        
        # insert data to partitions
        cut_res = [False] * FAMS_MAX_K
        for i in range(self.n_points):
            for j in range(self.L):
                self.__EvalCutRes(self.points_[i], cuts[j], cut_res)
                m, hjump = self.__HashFunction(cut_res, j, self.K, self.M_, getdoubleHash=True)
                m2 = self.__HashFunction(cut_res[1:], j, self.K - 1)
                self.__AddDataToHash(HT, hs, self.points_[i], m, self.Bs, self.M_, j, m2, hjump)

        # run pilot
        self.__ComputePilot(HT, hs, cuts)

        # do mean shift
        self.__DoFAMS(HT, hs, cuts, HT2, hs2)

        # join modes
        self.__PruneModes(FAMS_PRUNE_WINDOW, FAMS_PRUNE_MINN)

    def SaveModes(self):
        if self.nsel_ < 1:
            self.bglog('not data')
            raise SystemExit
        self.bglog('Save convergence points', end='')

        with open(self.outFile, 'w') as f:
            idx = 0
            for i in range(self.nsel_):
                for j in range(self.d_points):
                    tmp = int(self.modes_[idx]) * int(self.maxVal_ - self.minVal_)
                    tmp = self.cf.cdiv(tmp, 65535.0)
                    tmp += self.minVal_
                    idx += 1
                    f.write(f'{tmp:g} ')
                f.write('\n')
        self.bglog('done')

    def SavePrunedModes(self):
        if self.nsel_ < 1:
            self.bglog('not data')
            raise SystemExit
        self.bglog('Save joined convergence points', end='')

        with open(self.modesFile, 'w') as f:
            idx = 0
            for i in range(self.npm_):
                f.write(f'{self.nprunedmodes_[i]} ')
                for j in range(self.d_points):
                    tmp = int(self.prunedmodes_[idx]) * int(self.maxVal_ - self.minVal_)
                    tmp = self.cf.cdiv(tmp, 65535.0)
                    tmp += self.minVal_
                    idx += 1
                    f.write(f'{tmp:g} ')
                f.write('\n')
        self.bglog('done')

    def bglog(self, fmt, end='\n'):
        if self.log_enable:
            print(fmt, end=end)

    def test(self):
        ''' test code '''
        raise SystemExit

if __name__ == "__main__":
    f = FamsIf()

    ''' init '''
    fams = FAMS2D(f)

    ''' load points '''
    fams.loadPoints()

    ''' run fams '''
    fams.runFams()
    
    ''' save modes and pruned modes '''
    fams.SaveModes()
    fams.SavePrunedModes()




#%%