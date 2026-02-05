""" create data samples """
import lmdb
import math
import numpy as np
#import pyarrow
import pyarrow
import pickle
import os
import glob

class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps,
                 audio_base_dir,audio_fps):
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.audio_base_dir = audio_base_dir
        self.audio_fps = audio_fps

        self._build_audio_path_map()
        
        print(clip_lmdb_dir)
        print("-----")
        import os

        file_path = clip_lmdb_dir
        if os.path.exists(file_path):
            print(f"file {file_path} exists")
        else:
            print(f"file {file_path} not exists")
        
        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # create db for samples
        #map_size = 1024 * 50  # in MB
        #map_size <<= 20  # in B
        map_size_gb = 2048 # 2048 GB = 2 TB
        map_size = 1024 * map_size_gb  # map_size = 1024 * 2048 (in MB)
        map_size <<= 20  # map_size (in B)
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

    def _build_audio_path_map(self):
        self.audio_path_map = {}
        self.splits = ['trn', 'val', 'tst']
        self.roles = ['interloctr', 'main-agent']
        for split in self.splits:
            for role in self.roles:
                spec_dir = os.path.join(self.audio_base_dir, split, role, 'spec')
                spec_files = glob.glob(os.path.join(spec_dir, '*.npy'))

                for spec_file in spec_files:
                    filename = os.path.basename(spec_file)
                    file_id = filename.replace('.npy', '')
                    #加入字典
                    self.audio_path_map[file_id] = spec_file 

    def run(self):
        src_txn = self.src_lmdb_env.begin(write=False)

        # sampling and normalization
        cursor = src_txn.cursor()
        for key, value in cursor:
            video = pickle.loads(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                self._sample_from_clip(vid, clip)

        # print stats
        with self.dst_lmdb_env.begin() as txn:
            print('no. of samples: ', txn.stat()['entries'])

        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

    def _sample_from_clip(self, vid, clip):
        clip_skeleton = clip['poses']
        #clip_audio_raw = clip['audio_raw']
        #clip_word_list = clip['words']

        # load audio features
        audio_feature_path = self.audio_path_map.get(vid)
        if audio_feature_path is None:
            print(f"Audio feature for video id {vid} not found.")
            return
        clip_audio_features = np.load(audio_feature_path)#shaple(n,256,256)
        print(f'shape{clip_audio_features.shape}')

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_words_list = []
        sample_audio_list = []
        
        num_subdivision = math.floor(
            (len(clip_skeleton) - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            sample_audio = clip_audio_features[start_idx:fin_idx]
            '''
            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps
            #sample_words = self.get_words_in_time_range(word_list=clip_word_list,
            #                                            start_time=subdivision_start_time,
            #                                            end_time=subdivision_end_time)
            
            # filtering
            #if len(sample_words) < 3:
            #    continue

            # raw audio
            #audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            #audio_end = audio_start + self.audio_sample_length
            audio_start = math.floor(subdivision_start_time * self.audio_fps)
            audio_end = math.ceil(subdivision_end_time * self.audio_fps)
            sample_audio = clip_audio_features[audio_start:audio_end]
            '''
            if sample_audio.shape[0] < self.n_poses:
                pad_width = self.n_poses - sample_audio.shape[0]
                sample_audio = np.pad(sample_audio, ((0, pad_width), (0, 0)), mode='edge')

            elif sample_audio.shape[0] > self.n_poses:
                sample_audio = sample_audio[:self.n_poses]

            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps
            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            sample_skeletons_list.append(sample_skeletons)
            #sample_words_list.append(sample_words)
            sample_audio_list.append(sample_audio)
            aux_info.append(motion_info)

        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                for poses, audio, aux in zip(sample_skeletons_list,
                                                    sample_audio_list, aux_info):
                    poses = np.asarray(poses)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    v = [poses, audio, aux]
                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1

    @staticmethod
    def get_words_in_time_range(word_list, start_time, end_time):
        words = []

        for word in word_list:
            _, word_s, word_e = word[0], word[1], word[2]

            if word_s >= end_time:
                break

            if word_e <= start_time:
                continue

            words.append(word)

        return words