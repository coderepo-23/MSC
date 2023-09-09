# %%
from enum import Enum
import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import os
import time
import shutil

data_folders = ['nitromethane', 'methanol', 'acetone', 'hand_cream', 'h2o2', 'h2o', 'garnier_fructis', 'whiskey', 'brandy_chantre', 'olive_oil', 'nivea_lotion', 'sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9', 'sample10', 'sample11', 'sample12', 'sample13', 'sample14', 'sample15', 'sample16', 'sample17', 'sample18', 'sample19', 'sample20']

class Datatype(Enum):
    NONE = 0
    CHAR = 1
    UCHAR = 2
    SHORT = 3
    USHORT = 4
    INT = 5
    UINT = 6
    LONG = 7
    ULONG = 8
    FLOAT = 9
    DOUBLE = 10

class Dataorder(Enum):
    UNDEF_ORDER = 0
    ROW_MAJOR = 1
    COLUMN_MAJOR =2

class FamsFile_ASCII:
    ''' File interface for FAMS '''
    def __init__(self):
        self._data = None
        self._ndims = 0
        self._dim = None
        self._filename = ''
        self._datatype = Datatype.USHORT
        self._dataorder = Dataorder.ROW_MAJOR

    def SetDimensions(self, numpy_input_dimensions):
        self._ndims = len(numpy_input_dimensions)
        self._dim = np.copy(numpy_input_dimensions)

    def setDataorder(self, dataorder):
        self._dataorder = dataorder

    def setDataByUShort(self, numpy_input_data):
        self._datatype = Datatype.USHORT
        length = len(numpy_input_data)
        size = 1
        for n in self._dim:
            size *= n
        size = min(length, size)
        self._data = np.copy(numpy_input_data[0:size])

    def setFilename(self, filename):
        self._filename = filename

    def writeFile(self):
        if (len(self._dim) <= 1 or len(self._data) <= 1):
            return

        data_filename = self._filename
        
        try:
            with open(data_filename, "w") as data_file:
                d_entries = 1
                for i in range(self._ndims - 1):
                    d_entries *= self._dim[i]
                d_dims = self._dim[self._ndims - 1]
                # title
                data_file.write(f"{d_entries} {d_dims}\n")
                # content
                for i in range(d_entries):
                    for j in range(d_dims):
                        if self._dataorder == Dataorder.ROW_MAJOR:
                            data_file.write(f"{self._data[i*d_dims + j]} ")
                        elif self._dataorder == Dataorder.COLUMN_MAJOR:
                            data_file.write(f"{self._data[j*d_entries + i]} ")
                    data_file.write("\n")
        except IOError:
            print("Cannot open multivariate data file for writing.")

class FamsIf:
    def __init__(self):
        ''' raw data '''
        self.npMat = np.zeros(0)
        self.numDims = 0
        self.oshape = np.zeros(0)

        ''' spectraul data '''
        self.ushrtArray = np.zeros(0)
        self.dLambda_data = np.zeros(0)
        self.multispectral_data = np.zeros(0)

        ''' binning '''
        self.adptiveTest = False
        # self.binMethod = 'all channels'
        # self.binMethod = 'reduced spectrum'   # select 11 channels

        # self.binMethod = 'mean uniform'
        # self.binMethod = 'mean adaptive'
        # self.binMethod = 'median uniform'
        self.binMethod = 'median adaptive'

        # self.binMethod = 'myreduced template'
        # self.binMethod = 'reduced template'
        self.spectralStride = 8     # 8
        self.selected_channels = []

        ''' file directroies '''
        self.famsDir = 'tmp'
        self.famsRes = 'FAMS_res'
        if not os.path.exists(self.famsDir):
            os.makedirs(self.famsDir)
        if not os.path.exists(self.famsRes):
            os.makedirs(self.famsRes)

        self.fieldName = 'data'     # the field name for .h5

        self.c_program_file = os.path.join(self.famsDir, 'fams2D')
        self.c_modesFile = os.path.join(self.famsDir, 'modes_fams.txt')
        self.c_h5outputFile = os.path.join(self.famsDir, 'seg.h5')  # the result file

        self.py_modesFile = os.path.join(self.famsDir, 'modes_fams_py.txt')
        self.py_h5outputFile = os.path.join(self.famsDir, 'seg.h5')  # the result file

        ''' algorithm parameters '''
        self.famsK = 24
        self.famsL = 35
        if self.binMethod == 'reduced spectrum':
            self.famsk = 100
        else:
            self.famsk = 220

        ''' debug '''
        self.debug = False
        self.redirect = True
        self.c_run = True
        self.py_run= False

        self.c_debugFile = os.path.join(self.famsDir, 'debug_c.txt')
        self.c_cmmd = f'"{self.c_program_file}" {self.famsK} {self.famsL} {self.famsk} fams {self.famsDir + os.path.sep}'
        self.c_r_cmmd = self.c_cmmd + f' > {self.c_debugFile}'

        self.py_debugFile = os.path.join(self.famsDir, 'debug_py.txt')
        self.py_cmmd = f'python.exe Fams2D.py'
        self.py_r_cmmd = f'python.exe Fams2D.py' + f' > {self.py_debugFile}'

    def loadMat(self, filename):
        ''' load data from .h5 files '''
        f = h5py.File(filename, 'r')
        self.npMat = np.array(f['data']['value'], order='F').transpose()
        f.close()
        # read in data
        self.npMat = np.squeeze(self.npMat)
        self.numDims = len(self.npMat.shape)    # 3
        self.oshape  = self.npMat.shape         # 100, 100, 128

        # remove invalid entry
        flatIterator = self.npMat.flat
        for entry in flatIterator:
            if(math.isnan(entry) or math.isinf(entry)):
                entry=0

    def normalizedGradient(self):
        ''' normalized spectral gradient '''

        # calculate the normalized spectral gradient
        aMax, aMin = np.nanmax(self.npMat), np.nanmin(self.npMat)
        dLambda_data = np.gradient((self.npMat - aMin)/(aMax - aMin))
        dLambda_data = (dLambda_data[self.numDims-1]*1023)+1023
        dLambda_data = dLambda_data.astype(np.ushort)

        self.npMat = (self.npMat - aMin)/(aMax - aMin)
        self.npMat = self.npMat*65535
        self.npMat = self.npMat.astype(np.ushort)
        self.ushrtArray = np.reshape(self.npMat, self.oshape, order='C')
        self.dLambda_data = np.reshape(dLambda_data, self.oshape, order='C')

    def __uniformBinning(self):
        ''' select engergy channels uniformly '''
        # select some of the channels
        if self.spectralStride > 0:
            self.selected_channels = []
            offset = math.floor(math.floor(89 / self.spectralStride) / 2)
            for binIdx in range(0, 89, self.spectralStride):
                tmp = 16+offset+binIdx
                if(tmp < 105):
                    self.selected_channels.append(tmp)
        # full spectrum
        else:
            self.selected_channels = [19,27,35,43,51,59,67,75,83,91,99]

    def __reducedBinning(self):
        ''' use a pre binned data '''
        enl = self.oshape[self.numDims - 1]
        for i in range(enl):
            self.selected_channels.append(i)

    def __adaptiveBinningTest(self):
        pass

    def binning(self):
        ''' get bins from raw data '''
        if self.adptiveTest:
            self.__adaptiveBinningTest()
        else:
            self.__reducedBinning()

    def preProcess(self):
        ''' allocate matrix and set up command '''
        nfeatures = 2*len(self.selected_channels)
        # multispectral_data: data for result
        self.multispectral_data = np.zeros( (self.oshape[0],self.oshape[1], nfeatures), dtype=(self.ushrtArray.dtype) )
        # multispectral_train: data for training
        multispectral_train = np.zeros( (self.oshape[0],self.oshape[1], nfeatures), dtype=self.ushrtArray.dtype )

        for X_index in itertools.islice(itertools.count(),0,self.oshape[0]):
            for Y_index in itertools.islice(itertools.count(),0,self.oshape[1]):
                self.multispectral_data[X_index,Y_index,:] = np.concatenate((self.ushrtArray[X_index,Y_index,self.selected_channels],self.dLambda_data[X_index,Y_index,self.selected_channels]))
                multispectral_train[X_index,Y_index,:] = np.concatenate((self.ushrtArray[X_index,Y_index,self.selected_channels],self.dLambda_data[X_index,Y_index,self.selected_channels]))

        self.multispectral_data = np.asarray(self.multispectral_data, dtype=np.float32)

        ''' save training data '''
        fname = os.path.join(self.famsDir, 'fams.txt')
        dimArray = np.array([multispectral_train.shape[0], multispectral_train.shape[1], multispectral_train.shape[2]], dtype=np.uint32)
        arrayLen = multispectral_train.shape[0]*multispectral_train.shape[1]*multispectral_train.shape[2]
        multispectral_train = np.reshape(multispectral_train, arrayLen)
        sliceFile = FamsFile_ASCII()
        sliceFile.SetDimensions(dimArray)
        sliceFile.setDataorder(Dataorder.ROW_MAJOR)
        sliceFile.setDataByUShort(multispectral_train.astype(np.ushort))
        sliceFile.setFilename(fname)
        sliceFile.writeFile()

    def __file_islocked(self, filepath):
        ''' check whether the file is ready to read '''
        locked = None
        try:
            with open(filepath, "a") as f:
                locked = False
        except:
            locked = True
        return locked
    
    def __hold(self, path):
        while (not os.path.exists(path)) or (self.__file_islocked(path)):
            print("not ready for reading ...")
            time.sleep(5)
        print ("ready for processing ...")

    def runFams(self):
        ''' calling programs to run fams '''
        start_famsproc = time.time()

        if self.debug:
            if self.c_run:
                print('calling cpp: ', end='')
                if self.redirect:
                    cpp = os.system(self.c_r_cmmd)
                else:
                    cpp = os.system(self.c_cmmd)
                print(f'{cpp}')

            if self.py_run:
                print('calling py: ', end='')
                if self.redirect:
                    py = os.system(self.py_r_cmmd)
                else:
                    py = os.system(self.py_cmmd)
                print(f'{py}')
            print('debug done')
            raise SystemExit
            # debug: program ends at here
        else:
            if self.c_run:
                cmmd = self.c_cmmd
                print(f'calling {cmmd}')
                status = os.system(cmmd)
                if(status==0):
                    self.__hold(self.c_modesFile)
                else:
                    print(f'error! status: {status}')
                    raise SystemExit

            if self.py_run:
                cmmd = self.py_cmmd
                print(f'calling {cmmd}')
                status = os.system(cmmd)
                if(status==0):
                    self.__hold(self.py_modesFile)
                else:
                    print(f'error! status: {status}')
                    raise SystemExit

        end_famsproc = time.time()
        print(f"Ran FAMS segmentation in {end_famsproc-start_famsproc:.2f} seconds")

    def __loadModes(self, filepath):
        ''' laod modes generated by algorithm '''
        _modes = np.zeros(1, dtype=np.float32, order='C')
        _numEntries = np.zeros(1, dtype=np.uintc, order='C')
        if(os.path.exists(filepath) == False):
            return None
        modes = []
        numEntries = []
        with open(filepath, 'r') as text_file:
            for line in text_file:
                dataStringArray = line.split(" ")
                numEntries.append(int(dataStringArray[0]))
                modeData = []
                for entryString in dataStringArray[1:len(dataStringArray)]:
                    entryString=entryString.strip()
                    #print entryString
                    if(len(entryString)==0):
                        continue
                    modeData.append(float(entryString))
                modes.append(modeData)
            _numEntries = np.array(numEntries, dtype=np.uintc, order='C')
            _modes = np.array(modes, dtype=np.float32, order='C')
        return _modes

    def __applyseg(self, outFile, h5File):
        ''' apply the segmentation '''
        start_segmentData = time.time()

        ''' load modes '''
        modes = self.__loadModes(outFile)
        dist = np.zeros(modes.shape[0], dtype=np.float32)
        segments = np.zeros((self.oshape[0],self.oshape[1]), dtype=np.ushort)

        ''' apply seg '''
        final_segments = []
        segments = np.zeros((self.oshape[0],self.oshape[1]), dtype=np.ushort)
        for X_index in itertools.islice(itertools.count(),0,self.oshape[0]):
            for Y_index in itertools.islice(itertools.count(),0,self.oshape[1]):
                for M_index in itertools.islice(itertools.count(), modes.shape[0]):
                    dv = modes[M_index,:] - self.multispectral_data[X_index,Y_index,:]
                    dist[M_index] = np.linalg.norm(dv)
                seg = np.argmin(dist)
                segments[X_index,Y_index]=seg
                if seg not in final_segments:
                    final_segments.append(seg)

        ''' file operation '''
        if not self.debug:
            os.remove(os.path.join(self.famsDir, "fams.txt"))
            os.remove(os.path.join(self.famsDir, "modes_fams.txt"))
            os.remove(os.path.join(self.famsDir, "out_fams.txt"))
            os.remove(os.path.join(self.famsDir, f"pilot_{self.famsk}_fams.txt"))

        ''' h5 file for result '''
        f = h5py.File(h5File,'w')
        data_group = f.create_group(self.fieldName)
        segment_data_hdf5 = segments.transpose()
        data_group.create_dataset('value', data=segment_data_hdf5);
        f.close()

        end_segmentData = time.time()
        print(f"Applied Segmentation to the rest of the data in {end_segmentData-start_segmentData:.2f} seconds (final # segments: {len(final_segments)})")

    def applySegmentation(self):
        if self.c_run:
            self.__applyseg(self.c_modesFile, self.c_h5outputFile)

        if self.py_run:
            self.__applyseg(self.py_modesFile, self.py_h5outputFile)

    def __show(self, file, name='', saveas=None):
        ''' show segmentation result '''
        f = h5py.File(file, 'r')
        data = np.array(f[self.fieldName]['value'], order='F').transpose()
        f.close()

        plt.figure()
        plt.imshow(data, cmap='gray')
        plt.axis('off')
        plt.text(0, 0, name, fontsize=15, color='white', weight='bold', va='top', ha='left', bbox={'facecolor': 'gray', 'alpha': 0.5})
        if saveas:
            plt.savefig(saveas, dpi=300, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close()

    def showSegmentation(self, name='', saveas=None):
        if self.c_run:
            self.__show(self.c_h5outputFile, name, saveas)

        if self.py_run:
            self.__show(self.py_h5outputFile, name, saveas)

    def copyImgs(self):
        ''' copy the full-spectrun, reduced spectrum and manual segmentation data to target location '''
        src_base = '../MUSIC2D_HDF5'
        tar_base = 'dataset'
        for fp in data_folders:
            full_path = os.path.join(src_base, fp, 'fullSpectrum/reconstruction/reconstruction.h5')
            reduced_path = os.path.join(src_base, fp, 'reducedSpectrum/reconstruction/reconstruction.h5')
            manual_path = os.path.join(src_base, fp, 'manualSegmentation/manualSegmentation.h5')

            tar_dir = os.path.join(tar_base, fp)
            manual_template_dir = os.path.join(self.famsRes, 'manual template')

            full_tar_path = os.path.join(tar_dir, 'full.h5')
            reduced_tar_path = os.path.join(tar_dir, 'reduced.h5')
            manual_tar_path = os.path.join(tar_dir, 'manualSegmentation.h5')
            manual_template_path = os.path.join(manual_template_dir, f'{fp}.h5')

            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)
            if not os.path.exists(manual_template_dir):
                os.makedirs(manual_template_dir)
            shutil.copy(full_path, full_tar_path)
            shutil.copy(reduced_path, reduced_tar_path)
            shutil.copy(manual_path, manual_tar_path)
            shutil.copy(manual_path, manual_template_path)

        raise SystemExit

    def do_adaptiveBinning(self, name):
        ''' try different way of binning '''
        filename = f'dataset/{name}/full.h5'
        f = h5py.File(filename, 'r')
        fdata = np.array(f['data']['value'], order='F').transpose()
        f.close()
        fdata = np.squeeze(fdata).transpose()   # 128 100 100

        # reduced channels
        filename = f'dataset/{name}/reduced.h5'
        f = h5py.File(filename, 'r')
        rdata = np.array(f['data']['value'], order='F').transpose()
        f.close()
        rdata = np.squeeze(rdata).transpose()   #10 100 100
        sr = [np.std(rdata[i]) for i in range(10)]

        binloc = [None for _ in range(10)]

        # try manually reduced bin
        if self.binMethod == 'myreduced template':
            binloc[0] = (15,20)
            binloc[1] = (21,24)
            binloc[2] = (25,29)
            binloc[3] = (30,39)
            binloc[4] = (40,49)
            binloc[5] = (50,59)
            binloc[6] = (60,69)
            binloc[7] = (70,79)
            binloc[8] = (80,89)
            binloc[9] = (90,100)
            reduced_bin_sigma = [np.std(np.mean(fdata[binloc[i][0]:(binloc[i][1]+1)], axis=0)) for i in range(len(binloc))]

            avg_ch = [np.mean(fdata[binloc[i][0]:(binloc[i][1]+1)], axis=0) for i in range(len(binloc))]

        # reduced bin template
        if self.binMethod == 'reduced template':
            avg_ch = rdata

        # try median first
        if self.binMethod == 'median adaptive':
            '''
            1. do binning, ensuring the sigma of each bin is same
            2. compute the median channel of each bin
            '''
            binloc[0] = (0,32)
            binloc[1] = (33,33)
            binloc[2] = (34,34)
            binloc[3] = (35,39)
            binloc[4] = (40,41)
            binloc[5] = (42,43)
            binloc[6] = (44,45)
            binloc[7] = (46,49)
            binloc[8] = (50,51)
            binloc[9] = (110,len(fdata))
            median_bin_sigma = [np.std(np.median(fdata[binloc[i][0]:(binloc[i][1]+1)], axis=0)) for i in range(len(binloc))]

            avg_ch = [np.median(fdata[binloc[i][0]:(binloc[i][1]+1)], axis=0) for i in range(len(binloc))]

            raise SystemExit

        # do median on uniform binning
        if self.binMethod == 'median uniform':
            avg_ch = [np.median(fdata[(i*16):((i+1)*16)], axis=0) for i in range(8)]

        if self.binMethod == 'mean adaptive':
            binloc[0] = (0,32)
            binloc[1] = (33,33)
            binloc[2] = (34,34)
            binloc[3] = (35,39)
            binloc[4] = (40,41)
            binloc[5] = (42,43)
            binloc[6] = (44,45)
            binloc[7] = (46,49)
            binloc[8] = (50,51)
            binloc[9] = (52,len(fdata))

            avg_ch = [np.mean(fdata[binloc[i][0]:(binloc[i][1]+1)], axis=0) for i in range(len(binloc))]

        # uniform binning, 1-128
        if self.binMethod == 'mean uniform':
            avg_ch = [np.mean(fdata[(i*16):((i+1)*16)], axis=0) for i in range(8)]

        # try all channels
        if self.binMethod == 'all channels':
            avg_ch = fdata

        # select 11 channls
        if self.binMethod == 'reduced spectrum':
            avg_ch = fdata

        # load data
        self.npMat = np.stack(avg_ch).transpose()
        self.numDims = len(self.npMat.shape)
        self.oshape = self.npMat.shape

        flatIterator = self.npMat.flat
        for entry in flatIterator:
            if(math.isnan(entry) or math.isinf(entry)):
                entry=0

        # select fixed 11 channels
        if self.binMethod == 'reduced spectrum':
            self.selected_channels = [19,27,35,43,51,59,67,75,83,91,99]
        # load all binned channels
        else:
            enl = self.oshape[self.numDims - 1]
            for i in range(enl):
                self.selected_channels.append(i)

    def compare_with_manualSegmentation(self):
        ''' compare the accuracy with manual segmantation '''
        accuracy_path = os.path.join(self.famsRes, self.binMethod, 'accuracy.txt')
        with open(accuracy_path, 'w') as accuracy_file:
            for name in data_folders:
                ''' 0. load manual segmatation and fams result '''
                fn = os.path.join(self.famsRes, 'manual template', f'{name}.h5')
                f = h5py.File(fn, 'r')
                manualSeg = np.array(f['data']['value'], order='F').transpose()
                f.close()
                
                fn = os.path.join(self.famsRes, self.binMethod, f'{name}.h5')
                f = h5py.File(fn, 'r')
                famsSeg = np.array(f['data']['value'], order='F').transpose()
                f.close()

                val = np.unique(manualSeg)
                index = {}
                for i in range(len(val)):
                    tmp = np.where(manualSeg == val[i])
                    index[i] = list(zip(tmp[0], tmp[1]))

                correct = 0
                for _, seg_idx in index.items():
                    region = [famsSeg[x, y] for x, y in seg_idx]
                    val, counts = np.unique(region, return_counts=True)
                    correct += max(counts)

                accuracy = (correct / (100 * 100)) * 100
                if accuracy >= 100:
                    accuracy = 0
                accuracy_file.write(f'{accuracy:.2f}%\n')

    def plot_accuracy(self):
        ''' plot the accuracy of different binning method '''
        fn1 = os.path.join(self.famsRes, self.binMethod, 'accuracy.txt')
        fn2 = os.path.join(self.famsRes, 'reduced template', 'accuracy.txt')
        ac1, ac2 = [], []
        with open(fn1, 'r') as f1, open(fn2, 'r') as f2:
            for l1, l2 in zip(f1.readlines(), f2.readlines()):
                ac1.append(float(l1.rstrip('%\n')) / 100)
                ac2.append(float(l2.rstrip('%\n')) / 100)

        plt.figure()
        plt.plot(ac2, 'x', label='reduced template')
        plt.plot(ac1, 'o', label=self.binMethod)
        plt.xlabel('Samples')
        plt.ylabel('Accuracy percentage')
        plt.legend()
        plt.grid()

        save_path = os.path.join(self.famsRes, f'{self.binMethod} vs. reduced template.png')
        plt.savefig(save_path, dpi=300)

    def test(self):
        raise SystemExit

if __name__ == "__main__":
    ''' run single image '''
    if 1:
        fi = FamsIf()
        name = 'acetone'

        # preprocess
        fi.do_adaptiveBinning(name)
        fi.normalizedGradient()
        fi.preProcess()

        # run fams
        fi.runFams()

        # apply modes
        fi.applySegmentation()

        # show result
        # fi.showSegmentation()
        fi.showSegmentation(name=name, saveas=f'tmp/{name}.png')

        raise SystemExit

    ''' run fams on dataset '''
    if 1:
        for n in data_folders:
            fi = FamsIf()
            path = os.path.join(fi.famsRes, fi.binMethod)
            if not os.path.exists(path):
                os.makedirs(path)
            res = os.path.join(path, f'{n}.png')
            fi.c_h5outputFile = os.path.join(fi.famsRes, fi.binMethod, f'{n}.h5')

            fi.do_adaptiveBinning(n)
            fi.normalizedGradient()
            fi.preProcess()
            fi.runFams()
            fi.applySegmentation()
            fi.showSegmentation(name=n, saveas=res)

        fi = FamsIf()
        fi.compare_with_manualSegmentation()
        fi.plot_accuracy()

        raise SystemExit



# %%
