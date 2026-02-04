import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
Based on the following Se2Seq implementations:
- https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq
- https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
'''

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # --- BERT MODIFICATION: Projection Layer ---
        # Reduces the BERT dimension (3072) to hidden_size before entering the RNN.
        # This saves memory and reduces parameters compared to feeding raw BERT vectors.
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GRU accepts 'hidden_size' as input (post-projection).
        # We use batch_first=True because BERT data comes as (Batch, Sequence, Dim).
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, 
                          dropout=self.dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, lengths):
        '''
        :param x: Input tensor of shape (Batch, Sequence, 3072) - From BERT
        :param lengths: List or tensor containing the actual length of each sequence
        :return: 
            outputs: (Batch, Sequence, Hidden)
            hidden: (Layers, Batch, Hidden)
        '''
        # 1. Projection (Batch, Seq, 3072) -> (Batch, Seq, Hidden)
        x = self.projection(x)

        # 2. Pack Padded Sequence
        # Allows the RNN to process variable lengths efficiently.
        # lengths must be on CPU for pack_padded_sequence.
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # 3. Pass through GRU
        outputs, hidden = self.gru(packed)
        
        # 4. Unpack
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # outputs shape: (Batch, Seq, Hidden * 2) if bidirectional

        # 5. Sum bidirectional outputs
        # We sum the forward and backward directions to maintain the 'hidden_size' dimension.
        # This is critical so the Decoder (Attn) receives the expected dimension size.
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            
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
        :param hidden: previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs: encoder outputs from Encoder, in shape (T,B,H)
        :return attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        
        # Repeat the hidden state for each time step of the encoder
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        
        attn_energies = self.score(H, encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) 
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

        self.out = nn.Linear(hidden_size, output_size)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def freeze_attn(self):
        for param in self.attn.parameters():
            param.requires_grad = False

    def forward(self, motion_input, last_hidden, encoder_outputs, vid_indices=None):
        '''
        :param motion_input: motion input for current time step, in shape [batch x dim]
        :param last_hidden: last hidden state of the decoder, in shape [layers x batch x hidden_size]
        :param encoder_outputs: encoder outputs in shape [steps x batch x hidden_size]
        :return decoder output, hidden state, attention weights
        '''
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        if self.discrete_representation:
            word_embedded = self.embedding(motion_input).view(1, motion_input.size(0), -1)
            motion_input = self.dropout(word_embedded)
        else:
            motion_input = motion_input.view(1, motion_input.size(0), -1)  # [1 x batch x dim]

        # attention
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)  # [batch x 1 x T]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # [batch x 1 x attn_size]
        context = context.transpose(0, 1)  # [1 x batch x attn_size]

        # make input vec
        rnn_input = torch.cat((motion_input, context), 2)  # [1 x batch x (dim + attn_size)]

        if self.speaker_model:
            assert vid_indices is not None
            speaker_context = self.speaker_embedding(vid_indices).unsqueeze(0)
            rnn_input = torch.cat((rnn_input, speaker_context), 2)

        rnn_input = self.pre_linear(rnn_input.squeeze(0))
        rnn_input = rnn_input.unsqueeze(0)

        # rnn
        output, hidden = self.gru(rnn_input, last_hidden)

        # post-fc
        output = output.squeeze(0)  # [batch x hidden_size]
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
            assert not self.discrete_representation
            input_with_noise_vec = torch.cat([motion_input, z], dim=1)

        return self.decoder(input_with_noise_vec, last_hidden, encoder_output, vid_indices)


class Seq2SeqNet(nn.Module):
    def __init__(self, args, pose_dim, n_frames, n_words, word_embed_size, word_embeddings, speaker_model=None):
        super().__init__()
        
        # --- BERT CONFIGURATION ---
        # We ignore n_words and word_embed_size arguments (originally for GloVe/Word2Vec)
        # We fix input_size=3072 (BERT Concatenated Dimension)
        self.encoder = EncoderRNN(
            input_size=3072, 
            hidden_size=args.hidden_size,
            num_layers=args.n_layers,
            dropout=args.dropout_prob,
            bidirectional=True
        )
        
        self.decoder = Generator(args, pose_dim, speaker_model=speaker_model)

        self.n_frames = n_frames
        self.n_pre_poses = args.n_pre_poses
        self.pose_dim = pose_dim

    def forward(self, in_text, in_lengths, poses, vid_indices):
        '''
        in_text: (Batch, Seq, 3072) -> From BERT DataLoader
        in_lengths: (Batch)
        poses: (Batch, Seq, Pose_Dim)
        '''
        
        # 1. Encoder (Process BERT)
        # The encoder now returns (Batch, Seq, Hidden) due to batch_first=True
        encoder_outputs, encoder_hidden = self.encoder(in_text, in_lengths)
        
        # 2. Prepare Hidden State for Decoder
        # Encoder is bidirectional: (num_layers * 2, Batch, Hidden)
        # We sum directions to match Decoder's expected (num_layers, Batch, Hidden)
        num_layers = self.encoder.num_layers
        hidden_size = self.encoder.hidden_size
        
        encoder_hidden = encoder_hidden.view(num_layers, 2, encoder_hidden.size(1), hidden_size)
        encoder_hidden = encoder_hidden.sum(dim=1) 
        decoder_hidden = encoder_hidden

        # 3. Prepare Outputs for Attention
        # The Original Decoder/Attention iterates over time (dim 0), so it expects (Seq, Batch, Hidden).
        # The Encoder (batch_first) returned (Batch, Seq, Hidden). We transpose:
        encoder_outputs = encoder_outputs.transpose(0, 1)

        # 4. Prepare Poses (Target)
        # The loop below iterates t from 0 to n_frames. We need poses to be (Seq, Batch, Dim).
        poses = poses.transpose(0, 1)

        outputs = torch.zeros(self.n_frames, poses.size(1), self.decoder.output_size).to(poses.device)

        # 5. Decoding Loop
        decoder_input = poses[0]  # initial pose
        outputs[0] = decoder_input

        for t in range(1, self.n_frames):
            # We pass encoder_outputs (Seq, Batch, H) to attention
            decoder_output, decoder_hidden, _ = self.decoder(
                None, 
                decoder_input, 
                decoder_hidden, 
                encoder_outputs,
                vid_indices
            )
            outputs[t] = decoder_output

            if t < self.n_pre_poses:
                decoder_input = poses[t]  # Teacher forcing (early frames)
            else:
                decoder_input = decoder_output  # Autoregressive

        # Return to (Batch, Seq, Dim)
        return outputs.transpose(0, 1)