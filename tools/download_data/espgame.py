# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os
import tarfile
import cPickle
import PIL.Image
import glob

from downloader import DataDownloader

class EspgameDownloader(DataDownloader):
    """
    See details about the MIRFLICKR-25000 dataset here:
    http://press.liacs.nl/mirflickr/mirdownload.html
    """

    def urlList(self):
        return [
            'http://hunch.net/~learning/ESP-ImageSet.tar.gz'
        ]

    def uncompressData(self):        
        filename = 'ESP-ImageSet.tar.gz'
        filepath = os.path.join(self.outdir, filename)
        assert os.path.exists(filepath), 'Expected "%s" to exist' % filename

        if not os.path.exists(os.path.join(self.outdir, 'ESP-ImageSet')):
            print "Uncompressing file=%s ..." % filename
            with tarfile.open(filepath) as tf:
                tf.extractall(self.outdir)


    def processData(self):
        print "Processing labels"
        self.__buildLabels()
        print "Extracting data"
        self.__extractData() 

    def __buildLabels(self):
        label_path = os.path.join(self.outdir, "labels.txt")
        if not os.path.exists(label_path):
            files = glob.glob(os.path.join(self.outdir, "ESP-ImageSet/LABELS/*.desc"))
            self.labels = []
            for f in files:
                self.labels.extend([line.rstrip('\n') for line in open(f)])


            result = []
            i = 0
            for entry in set(self.labels):
                i += 1
                result.append((entry, self.labels.count(entry)))
                print i
            result.sort(key = lambda x: -x[1])
            result = map( lambda x:x[0], result[0:200])
            self.labels = list(set(result))
            print len(self.labels)

            with open(label_path, "w") as fd:
                for l in self.labels:
                    fd.write(l+'\n')
                fd.close()
        else:
            with open(label_path, "r") as fd:
                self.labels = [line.rstrip('\n') for line in fd]


    def __extractData(self):
        """
        """
        data_path = os.path.join(self.outdir, "files.txt")
        if not os.path.exists(data_path):
            data_file = open(data_path, "w")
            files = glob.glob(os.path.join(self.outdir, "ESP-ImageSet/LABELS/*.desc"))

            print len(files)

            for f in files:
                res = os.path.join(self.outdir, "ESP-ImageSet/images/" + os.path.splitext(f)[0])
                with open(f, 'r') as fd:
                    lines = [line.rstrip('\n') for line in fd]
                    for l in self.labels:
                        if l in lines:
                            res = res + ' 1'
                        else:
                            res = res + ' -1'

                    res += "\n"
                    fd.close()
                data_file.write(res)
            data_file.close()
            print "Done."



