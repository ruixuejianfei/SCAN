from __future__ import absolute_import
import os
import os.path as osp
from reid.data.datasequence import Datasequence
from utils.osutils import mkdir_if_missing
from utils.serialization import write_json
import tarfile
import zipfile
from glob import glob
import shutil
import numpy as np

datasetname = 'prid_2011'
# flowname = 'prid2011flow_opencv'
flowname = 'prid_xiong'

class infostruct(object):
    pass

class PRID2011SEQUENCE(Datasequence):

    def __init__(self, root,  split_id=0, seq_len=12, seq_srd=6, num_val=1, download=False):
        super(PRID2011SEQUENCE, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            self.imgextract()

        self.load(seq_len, seq_srd, num_val)

        self.query, query_pid, query_camid, query_num = self._pluckseq_cam(self.identities, self.split['query'], seq_len, seq_srd, 0)
        self.queryinfo = infostruct()
        self.queryinfo.pid = query_pid
        self.queryinfo.camid = query_camid
        self.queryinfo.tranum = query_num

        self.gallery, gallery_pid, gallery_camid, gallery_num = self._pluckseq_cam(self.identities, self.split['gallery'], seq_len, seq_srd, 1)
        self.galleryinfo = infostruct()
        self.galleryinfo.pid = gallery_pid
        self.galleryinfo.camid = gallery_camid
        self.galleryinfo.tranum = gallery_num

    @property
    def other_dir(self):
        return osp.join(self.root, 'others')

    def download(self):

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        fpath1 = osp.join(raw_dir, datasetname + '.zip')
        fpath2 = osp.join(raw_dir, flowname + '.tar')

        if osp.isfile(fpath1) and osp.isfile(fpath2):
            print("Using the download file:" + fpath1 + " " + fpath2)
        else:
            print("Please firstly download the files")
            raise RuntimeError("Downloaded file missing!")

    def imgextract(self):

        raw_dir = osp.join(self.root, 'raw')
        exdir1 = osp.join(raw_dir, datasetname)
        exdir2 = osp.join(raw_dir, flowname)
        fpath1 = osp.join(raw_dir, datasetname + '.zip')
        fpath2 = osp.join(raw_dir, flowname + '.tar')

        if not osp.isdir(exdir1):
            print("Extracting tar file")
            cwd = os.getcwd()
            zip_ref = zipfile.ZipFile(fpath1, 'r')
            mkdir_if_missing(exdir1)
            zip_ref.extractall(exdir1)
            zip_ref.close()
            os.chdir(cwd)

        if not osp.isdir(exdir2):
            print("Extracting tar file")
            cwd = os.getcwd()
            tar_ref = tarfile.open(fpath2)
            mkdir_if_missing(exdir2)
            os.chdir(exdir2)
            tar_ref.extractall()
            tar_ref.close()
            os.chdir(cwd)

        ## recognizing the dataset
        # Format

        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        others_dir = osp.join(self.root, 'others')
        mkdir_if_missing(others_dir)

        fpaths1 = sorted(glob(osp.join(exdir1, 'multi_shot', '*/*/*.png')))
        fpaths2 = sorted(glob(osp.join(exdir2, '*/*/*.png')))

        identities_images = [[[] for _ in range(2)] for _ in range(200)]
        identities_others = [[[] for _ in range(2)] for _ in range(200)]

        for fpath in fpaths1:
            fname = fpath
            fname_list = fname.split('/')
            cam_name = fname_list[-3]
            pid_name = fname_list[-2]
            frame_name = fname_list[-1]
            cam_id = 1 if cam_name =='cam_a' else 2
            pid_id = int(pid_name.split('_')[-1])
            if pid_id > 200:
                continue
            frame_id = int(frame_name.split('.')[-2])
            imagefname = ('{:08d}_{:02d}_{:04d}.png'
                          .format(pid_id-1, cam_id-1, frame_id-1))
            identities_images[pid_id - 1][cam_id - 1].append(imagefname)
            shutil.copy(fpath, osp.join(images_dir, imagefname))

        for fpath in fpaths2:
            fname = fpath
            fname_list = fname.split('/')
            cam_name = fname_list[-3]
            pid_name = fname_list[-2]
            frame_name = fname_list[-1]
            cam_id = 1 if cam_name =='cam_a' else 2
            pid_id = int(pid_name.split('_')[-1])
            if pid_id > 200:
                continue
            frame_id = int(frame_name.split('.')[-2])
            flowfname = ('{:08d}_{:02d}_{:04d}.png'
                          .format(pid_id-1, cam_id-1, frame_id-1))
            identities_others[pid_id - 1][cam_id - 1].append(flowfname)
            shutil.copy(fname, osp.join(others_dir, flowfname))




        meta = {'name': 'prid2011-sequence', 'shot': 'sequence', 'num_cameras': 2,
                'identities': identities_images}

        write_json(meta, osp.join(self.root, 'meta.json'))
        # Consider fixed training and testing split
        num = 200
        splits = []
        for i in range(10):
            pids = np.random.permutation(num)
            pids = (pids -1).tolist()
            trainval_pids = pids[:num // 2]
            test_pids = pids[num // 2:]
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}

            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
    def _pluckseq_cam(self, identities, indices, seq_len, seq_str, camid):
        ret = []
        per_id = []
        cam_id = []
        tra_num = []

        for index, pid in enumerate(indices):
            pid_images = identities[pid]
            cam_images = pid_images[camid]
            seqall = len(cam_images)
            seq_inds = [(start_ind, start_ind + seq_len) \
                        for start_ind in range(0, seqall - seq_len, seq_str)]
            if not seq_inds:
                seq_inds = [(0, seqall)]
            for seq_ind in seq_inds:
                ret.append((seq_ind[0], seq_ind[1], pid, index, camid))
            per_id.append(pid)
            cam_id.append(camid)
            tra_num.append(len(seq_inds))
        return ret, per_id, cam_id, tra_num
