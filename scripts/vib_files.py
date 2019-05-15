#Find all vibration files in this path
import os
from os.path import sep, join

def pjoin(*args, **kwargs):
  return join(*args, **kwargs).replace(sep, '/')

class FilePath():
    def __init__(self, pathstr, pathlist, filename):
        self.path = pathstr
        self.folders = pathlist
        self.filename = filename
        self.fullpath = pjoin(pathstr, filename)

class SequencePath():
    def __init__(self, sp, sl, sf):
        self.SequenceFile = FilePath(sp, sl, sf)
        self.SequenceCount = 0
        self.TDMSCount = 0
        self.TestSequence = []
        self.TDMSFiles = []

        self.getSequence()
        self.getTDMSFiles()
        self.copySequences()

    def getSequence(self):
        with open(self.SequenceFile.fullpath) as f:
            for line in f:
                if line[0] != '#' and line != '\n':
                    self.SequenceCount = self.SequenceCount + 1
                    nonewline = line.replace('\n', '')
                    self.TestSequence.append(nonewline)

    def getTDMSFiles(self):
        self.TDMSFiles = []
        for dp, dn, filenames in os.walk(self.SequenceFile.path):
            for f in filenames:
                if (os.path.splitext(f)[1] == '.tdms' and ('Vibration' in os.path.splitext(f)[0])):
                    unixpath = dp.replace(sep, '/')
                    self.TDMSFiles.append(FilePath(unixpath, dn, f))
                    self.TDMSCount = self.TDMSCount + 1

    def copySequences(self):
        if self.SequenceCount > 0 and self.TDMSCount == self.SequenceCount:
            pass
        else:
            print('Error in {0}'.format(self.SequenceFile.fullpath))

def getAllSequenceFilesInPath(path):
    Sequences = []
    for dp, dn, filenames in os.walk(path):
        for f in filenames:
            if (os.path.splitext(f)[1] == '.txt' and ('sequence' in os.path.splitext(f)[0])):
                unixpath = dp.replace(sep, '/')
                Sequences.append(SequencePath(unixpath, dn, f))
    return(Sequences)

def getSequenceFilesInPath(search_path = './TDMS/'):
    result = [ pjoin(dp, f) 
        for dp, dn, filenames in os.walk(search_path) 
        for f in filenames 
        if os.path.splitext(f)[1] == '.txt' and 
            ('sequence' in os.path.splitext(f)[0]) ]

    return(result)

def getTdmsFilesInPath(search_path = './TMDS/'):
    result = [ pjoin(dp, f) 
        for dp, dn, filenames in os.walk(search_path) 
        for f in filenames 
        if os.path.splitext(f)[1] == '.tdms']

    return(result)

if __name__ == '__main__':
    '''res = getTdmsFilesInPath('./TDMS/')
    print (len(res))
    for f in res:
        print(f)
    ''' 
    res = getAllSequenceFilesInPath('./TDMS/')
    for f in res:
        print(f.SequenceCount)
        print(f.TestSequence)
        for i in range (0, f.SequenceCount):    
            pathname = f.TDMSFiles[i].path
            outfilename = 'meta_'
            for part in pathname.split('/')[2:]:
                outfilename = outfilename + part + '_'

            outfilename = outfilename + f.TestSequence[i] + ".ipynb"
            print(outfilename)

            with open('./VibTemplate.ipynb') as e:
                fileData = e.read()
                fileData = fileData.replace('<META_TEST>',  f.TestSequence[i])
                fileData = fileData.replace('<META_FILENAME>', repr(f.TDMSFiles[i].fullpath))
                d = open(outfilename, 'w')
                d.write(fileData)
                d.close()


    #dst = pjoin(res[0].SequenceFile.path, res[0].TestSequence[0] + '.ipynb')

    #fileData = open('.\\VibTemplate.ipynb').read()
    '''
    with open('.\\VibTemplate.ipynb') as f:
        fileData = f.read()
        filePath = repr(res[0].TDMSFiles[0].fullpath)
        #print(filePath)
        fileData = fileData.replace('<META_TEST>', res[0].TestSequence[0])
        fileData = fileData.replace('<META_FILENAME>', filePath)
        d = open('.\\out.ipynb', 'w')
        d.write(fileData)
        d.close()
    '''
