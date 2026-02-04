""" create data samples """
import lmdb
import math
import numpy as np
import pyarrow
import os

class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, data_mean=None, data_std=None):
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps

        # open original LMDB
        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # --- BERT ADAPTATION: Auto-locate Embeddings ---
        # Determines the embedding folder based on the LMDB file location.
        lmdb_grandparent_dir = os.path.dirname(os.path.dirname(clip_lmdb_dir))
        self.embedding_dir = os.path.join(lmdb_grandparent_dir, "embedding")

        if not os.path.exists(self.embedding_dir):
            raise FileNotFoundError(f"Embedding folder not found: {self.embedding_dir}")
        print(f"Using embeddings from: {self.embedding_dir}")

        # --- BERT ADAPTATION: Increase Map Size ---
        # Storing dense BERT embeddings increases dataset size significantly.
        # We increase the map_size to 1TB to avoid MapFullError.
        map_size = 1024 * 1024 * 1024 * 1024  # 1 TB
        
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

        self.data_mean = np.array(data_mean).squeeze() if data_mean is not None else None
        self.data_std = np.array(data_std).squeeze() if data_std is not None else None

    def run(self):
        src_txn = self.src_lmdb_env.begin(write=False)
        
        # sampling and normalization
        cursor = src_txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            
            # --- BERT ADAPTATION: Load Embeddings ---
            # Load the corresponding .npz file for the video once
            npz_path = os.path.join(self.embedding_dir, f"{vid}_embeddings.npz")
            if not os.path.isfile(npz_path):
                continue
                
            try:
                npz_data = np.load(npz_path, allow_pickle=True)
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                continue
            
            if 'intervals' not in npz_data:
                print(f"WARNING: {vid} has no time intervals. Please run the updated embedding script.")
                continue

            for clip_idx, clip in enumerate(clips):
                self._sample_from_clip(vid, clip, npz_data)

        # print stats
        with self.dst_lmdb_env.begin() as txn:
            print('no. of samples generated: ', txn.stat()['entries'])

        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

    def _sample_from_clip(self, vid, clip, npz_data):
        clip_skeleton = clip['poses']
        clip_audio_raw = clip['audio_raw']
        # clip_word_list is replaced by direct npz lookup

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_embeddings_list = []
        sample_audio_list = []
        
        num_subdivision = math.floor(
            (len(clip_skeleton) - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps
            
            # --- BERT ADAPTATION: Get Embeddings by Time ---
            sample_embeddings = self.get_embeddings_in_time_range(
                npz_data,
                start_time=subdivision_start_time,
                end_time=subdivision_end_time
            )

            # raw audio processing with safe padding
            audio_len = len(clip_audio_raw)
            pose_len = len(clip_skeleton)
            
            if pose_len > 0:
                audio_start = math.floor(start_idx / pose_len * audio_len)
                audio_end = audio_start + self.audio_sample_length
                
                # Ensure we don't go out of bounds
                if audio_end > audio_len: 
                    audio_end = audio_len
                    
                sample_audio = clip_audio_raw[audio_start:audio_end]
                
                # Pad with zeros if audio is shorter than expected
                if len(sample_audio) < self.audio_sample_length:
                     pad_len = self.audio_sample_length - len(sample_audio)
                     sample_audio = np.pad(sample_audio, (0, pad_len), 'constant')
            else:
                sample_audio = np.zeros(self.audio_sample_length)

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            sample_skeletons_list.append(sample_skeletons)
            sample_embeddings_list.append(sample_embeddings)
            sample_audio_list.append(sample_audio)
            aux_info.append(motion_info)

        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                for emb, poses, audio, aux in zip(sample_embeddings_list, sample_skeletons_list,
                                                  sample_audio_list, aux_info):
                    poses = np.asarray(poses)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    # Save embeddings instead of word list
                    v = [emb, poses, audio, aux]
                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1

    def get_embeddings_in_time_range(self, npz_data, start_time, end_time):
        """
        Retrieves concatenated BERT embeddings for phrases that overlap 
        with the current time window.
        """
        intervals = npz_data['intervals']
        
        # Find indices of phrases that overlap the current window
        # Logic: interval_start <= window_end AND interval_end >= window_start
        indices = np.where((intervals[:, 0] <= end_time) & (intervals[:, 1] >= start_time))[0]

        if len(indices) == 0:
            # Return a zero vector if no text overlaps (silence/no speech)
            return np.zeros((1, 3072), dtype=np.float32)

        selected_embeddings = []
        for idx in indices:
            key = f"phrase_{idx}"
            if key in npz_data:
                emb = npz_data[key]
                selected_embeddings.append(emb)
        
        if not selected_embeddings:
            return np.zeros((1, 3072), dtype=np.float32)

        # Concatenate embeddings to form the sequence input for the encoder
        return np.concatenate(selected_embeddings, axis=0)