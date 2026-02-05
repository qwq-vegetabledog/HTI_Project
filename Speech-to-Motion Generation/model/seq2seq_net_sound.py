import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math

'''
Based on the following Se2Seq implementations:
- https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq
- https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
'''


class AudiofetureProjector(nn.Module):
    def __init__(self, target_embed_size=1):
        super(AudiofetureProjector, self).__init__()
        #input:(B,1,256,256)-> output:(B,1,target_embed_size)
        '''
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, stride = 2, padding = 1), # (B,32,128,128)
            nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1), # (B,64,64,64)
            nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1), # (B,128,32,32)
            nn.BatchNorm2d(128),nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1), # (B,256,16,16)
            nn.BatchNorm2d(256),nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)), # (B,256,1,1)
        )
        self.fc = nn.Linear(256, target_embed_size)
        
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),     
            nn.Linear(256, target_embed_size)
        )
        '''
        self.net = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, target_embed_size) 
        )
    def forward(self, input_audio):
        print("the shape of input_audio:",input_audio.shape)
        B, T, D = input_audio.shape
        input_for_cnn = input_audio.view(B*T,D)
        #features = self.cnn(input_for_cnn).squeeze() # (B*T,256,1,1)
        #features = self.fc(features)
        '''
        features = self.cnn(input_for_cnn)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = features.view(B,T,-1).transpose(0,1).contiguous() 
        #features = features.view(B,T,1).transpose(0,1).contiguous() # (T,B,256)
        #changer to (T,B,1)
        '''
        features = self.net(input_for_cnn)
        features = features.view(B,T,-1).transpose(0,1).contiguous()
        return features



class EncoderRNN(nn.Module):
    #def __init__(self, input_size=1, embed_size, hidden_size, n_layers=1, dropout=0.5, pre_trained_embedding=None):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.5, pre_trained_embedding=None):    
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.embed_size = embed_size
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.n_layers = n_layers
        self.dropout = dropout
        #(B,N,256,256)
        #if pre_trained_embedding is not None:  # use pre-trained embedding (e.g., word2vec, glove)
        #    assert pre_trained_embedding.shape[0] == input_size
        #    assert pre_trained_embedding.shape[1] == embed_size
        #    self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding), freeze=False)
        #else:
        #    self.embedding = nn.Embedding(input_size, embed_size)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, input_seqs, input_lengths, hidden=None):
        '''
        :param input_seqs:
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input_lengths:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()
        
        embedded = self.input_projection(input_seqs)
        embedded = F.relu(embedded)

        #embedded = self.embedding(input_seqs)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu())
        outputs, hidden = self.gru(packed, hidden)

        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        print("the shape of encoder outputs:",outputs.shape)
        return outputs, hidden



class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(H, encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.1,
                 discrete_representation=False, speaker_model=None):
        super(BahdanauAttnDecoderRNN, self).__init__()

        # define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.discrete_representation = discrete_representation
        self.speaker_model = speaker_model

        # define embedding layer
        if self.discrete_representation:
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.dropout = nn.Dropout(dropout_p)

        if self.speaker_model:
            self.speaker_embedding = nn.Embedding(speaker_model.n_words, 8)

        # calc input size
        if self.discrete_representation:
            input_size = hidden_size  # embedding size
        linear_input_size = input_size + hidden_size
        if self.speaker_model:
            linear_input_size += 8

        # define layers
        self.pre_linear = nn.Sequential(
            nn.Linear(linear_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.attn = Attn(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)

        # self.out = nn.Linear(hidden_size * 2, output_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def freeze_attn(self):
        for param in self.attn.parameters():
            param.requires_grad = False

    def forward(self, motion_input, last_hidden, encoder_outputs, vid_indices=None):
        '''
        :param motion_input:
            motion input for current time step, in shape [batch x dim]
        :param last_hidden:
            last hidden state of the decoder, in shape [layers x batch x hidden_size]
        :param encoder_outputs:
            encoder outputs in shape [steps x batch x hidden_size]
        :param vid_indices:
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        '''

        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        if self.discrete_representation:
            word_embedded = self.embedding(motion_input).view(1, motion_input.size(0), -1)  # [1 x B x embedding_dim]
            motion_input = self.dropout(word_embedded)
        else:
            motion_input = motion_input.view(1, motion_input.size(0), -1)  # [1 x batch x dim]

        # attention
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)  # [batch x 1 x T]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # [batch x 1 x attn_size]
        context = context.transpose(0, 1)  # [1 x batch x attn_size]

        # make input vec
        rnn_input = torch.cat((motion_input, context), 2)  # [1 x batch x (dim + attn_size)]
        
        ###########################
        # print("the shape of rnn_input1:",rnn_input.shape)###########################
        ###########################
        
        if self.speaker_model:
            assert vid_indices is not None
            speaker_context = self.speaker_embedding(vid_indices).unsqueeze(0)
            rnn_input = torch.cat((rnn_input, speaker_context), 2)  # [1 x batch x (dim + attn_size + embed_size)]
            #######################
            # print("error at if self.speaker_model")
            ######################
        ################3
        # here is the question 
        #################

        ################################################################################################################################################################
        rnn_input = self.pre_linear(rnn_input.squeeze(0))
        
        ###########################
        # print("the shape of rnn_input2:",rnn_input.shape)###########################
        ###########################
        
        rnn_input = rnn_input.unsqueeze(0)

        ###########################
        # print("the shape of rnn_input3:",rnn_input.shape)###########################
        ###########################

        # rnn
        output, hidden = self.gru(rnn_input, last_hidden)

        # post-fc
        output = output.squeeze(0)  # [1 x batch x hidden_size] -> [batch x hidden_size]
        output = self.out(output)

        return output, hidden, attn_weights


class Generator(nn.Module):
    def __init__(self, args, motion_dim, discrete_representation=False, speaker_model=None):
        super(Generator, self).__init__()
        self.output_size = motion_dim
        self.n_layers = args.n_layers
        self.discrete_representation = discrete_representation
        self.decoder = BahdanauAttnDecoderRNN(input_size=motion_dim,
                                              hidden_size=args.hidden_size,
                                              output_size=self.output_size,
                                              n_layers=self.n_layers,
                                              dropout_p=args.dropout_prob,
                                              discrete_representation=discrete_representation,
                                              speaker_model=speaker_model)

    def freeze_attn(self):
        self.decoder.freeze_attn()

    def forward(self, z, motion_input, last_hidden, encoder_output, vid_indices=None):
        if z is None:
            input_with_noise_vec = motion_input
        else:
            assert not self.discrete_representation  # not valid for discrete representation
            input_with_noise_vec = torch.cat([motion_input, z], dim=1)  # [bs x (10+z_size)]

        return self.decoder(input_with_noise_vec, last_hidden, encoder_output, vid_indices)


class Seq2SeqNet(nn.Module):
    def __init__(self, args, pose_dim, n_frames, 
                acoustic_feature_dim=256, latent_dim=32, speaker_model=None):
        super().__init__()
        #self.encoder = EncoderRNN(
        #    n_words, word_embed_size, args.hidden_size, args.n_layers,
        #    dropout=args.dropout_prob, pre_trained_embedding=word_embeddings)
        self.encoder = EncoderRNN(
            input_size = args.hidden_size,
            #embed_size = args.hidden_size,
            hidden_size = args.hidden_size,
            n_layers = args.n_layers,
            dropout = args.dropout_prob,
            pre_trained_embedding = None
        )
        #self.encoder = EfficientSpectrogramCNN(latent_dim=latent_dim)

        self.audio_projector = AudiofetureProjector(target_embed_size=args.hidden_size)

        self.decoder = Generator(args, pose_dim, speaker_model=speaker_model)
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.n_pre_poses = args.n_pre_poses
        self.pose_dim = pose_dim
        if latent_dim!=args.hidden_size:
            self.latent_to_hidden = nn.Linear(latent_dim, args.hidden_size)
        else:
            self.latent_to_hidden = None

    def forward(self, in_audio, in_text=None, target_vec=None, in_lengths=None, poses=None, vid_indices=None):
        # reshape to (seq x batch x dim)
        #in_text = in_text.transpose(0, 1)
        batch_size = in_audio.size(0)
        embedded_audio = self.audio_projector(in_audio) 
        current_audio_frames = embedded_audio.size(0)
        if poses is None:
        # 如果是推断模式且没有提供 poses，可能需要报错或使用全零填充
            raise ValueError("poses (target gesture sequence) cannot be None during training/evaluation")
        poses = poses.transpose(0, 1)
        
        audio_seq_len = in_audio.shape[1]
        audio_lengths = torch.LongTensor([audio_seq_len]*batch_size).to(in_audio.device)
        if self.encoder.do_flatten_parameters:
            self.encoder.gru.flatten_parameters()
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded_audio, 
                                                         audio_lengths.cpu(),
                                                         enforce_sorted=False)
        #encoder_outputs, encoder_hidden = self.encoder.gru(packed, None)
        encoder_outputs, encoder_hidden = self.encoder(embedded_audio, audio_lengths)
        #encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs)  # unpack (back to padded)
        #encoder_outputs = encoder_outputs[:, :, :self.encoder.hidden_size] + encoder_outputs[:, :, self.encoder.hidden_size:]  # Sum bidirectional outputs

        #outputs = torch.zeros(self.n_frames, poses.size(1), self.decoder.output_size).to(poses.device)
        outputs = torch.zeros(current_audio_frames, poses.size(1), self.decoder.output_size).to(poses.device)
        # run words through encoder
        #encoder_outputs, encoder_hidden = self.encoder(in_text, in_lengths, None)
        #decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # use last hidden state from encoder

        decoder_hidden = encoder_hidden[::2] + encoder_hidden[1::2] # 汇总双向 GRU 的状态
        decoder_hidden = decoder_hidden[:self.decoder.n_layers]

        # run through decoder one time step at a time
        decoder_input = poses[0]  # initial pose from the dataset
        outputs[0] = decoder_input

        for t in range(1, current_audio_frames):

            #######################
            # 这里有问题
            ######################
            # print("Question at the step:", t)

            decoder_output, decoder_hidden, _ = self.decoder(None, decoder_input, decoder_hidden, encoder_outputs,
                                                            vid_indices)
            outputs[t] = decoder_output

            if t < self.n_pre_poses:
                decoder_input = poses[t]  # next input is current target
            else:
                decoder_input = decoder_output  # next input is current prediction
            print(f"decoder shape at step {t}:", decoder_output.shape)
            
            #####################################3
            # print("the size of decoder_hidden",decoder_hidden.shape)
            # print("the size of decoder_output",decoder_output.shape)
            

        return outputs.transpose(0, 1)
'''
    def forward(self, in_spectrograms,unused_length_arg, poses, vid_indices):
        #the shape of in_spectrograms: (batch, 1, 256, 256)
        # reshape to (seq x batch x dim)
        latent_vectors = self.encoder.encode(in_spectrograms)  # (batch, latent_dim)
        if self.latent_to_hidden is not None:
            decoder_hidden = self.latent_to_hidden(latent_vectors)  # (batch, hidden_size)
            decoder_hidden = decoder_hidden.unsqueeze(0).repeat(self.decoder.n_layers, 1, 1)  # (layers, batch, hidden_size)
        else:
            decoder_hidden = latent_vectors.unsqueeze(0).repeat(self.decoder.n_layers, 1, 1)  # (layers, batch, hidden_size)   

        encoder_outputs = None 
        poses = poses.transpose(0, 1)
        outputs = torch.zeros(self.n_frames, poses.size(1), self.decoder.output_size).to(poses.device)
        decoder_input = poses[0]  # initial pose from the dataset
        outputs[0] = decoder_input

        for t in range(1, self.n_frames):
            decoder_output, decoder_hidden, _ = self.decoder(None, 
                                                             decoder_input, 
                                                             decoder_hidden, 
                                                             encoder_outputs,
                                                             vid_indices)
            outputs[t] = decoder_output
            if t < self.n_pre_poses:
                decoder_input = poses[t]  # next input is current target
            else:
                decoder_input = decoder_output  # next input is current prediction
            print("the size of decoder_hidden",decoder_hidden.shape)
            print("the size of decoder_output",decoder_output.shape)
        return outputs.transpose(0, 1)
    '''

