import torch.nn as nn
import torch
from model_parts import *

class Model_ConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(Model_ConvLSTM, self).__init__()

        self.encoder_1 = ConvLSTMCell(input_dim=in_chan,hidden_dim=nf,kernel_size=(3,3),bias=True)
        self.encoder_2 = ConvLSTMCell(input_dim=nf,hidden_dim=nf,kernel_size=(3,3),bias=True)
        self.encoder_3 = ConvLSTMCell(input_dim=nf,hidden_dim=nf,kernel_size=(3,3),bias=True)
        self.encoder_4 = ConvLSTMCell(input_dim=nf,hidden_dim=nf,kernel_size=(3,3),bias=True)
        self.encoder_5 = ConvLSTMCell(input_dim=nf,hidden_dim=nf,kernel_size=(3,3),bias=True)
        self.encoder_6 = ConvLSTMCell(input_dim=nf,hidden_dim=nf,kernel_size=(3,3),bias=True)

        self.decoder_1 = ConvLSTMCell(input_dim=nf,hidden_dim=nf,kernel_size=(3,3),bias=True)
        self.decoder_2 = ConvLSTMCell(input_dim=nf,hidden_dim=nf,kernel_size=(3,3),bias=True)
        self.decoder_3 = ConvLSTMCell(input_dim=nf,hidden_dim=nf,kernel_size=(3,3),bias=True)
        self.decoder_4 = ConvLSTMCell(input_dim=nf,hidden_dim=nf,kernel_size=(3,3),bias=True)
        self.decoder_5 = ConvLSTMCell(input_dim=nf,hidden_dim=nf,kernel_size=(3,3),bias=True)
        self.decoder_6 = ConvLSTMCell(input_dim=nf,hidden_dim=nf,kernel_size=(3,3),bias=True)

    def forward(self, x, future_seq=1, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # input dimensions
        b, seq_len, _, h, w = x.size()

        # initial hidden states
        h_t, c_t = self.encoder_1.init_hidden(batch_size=b, image_size=(h,w))
        h_t2, c_t2 = self.encoder_2.init_hidden(batch_size=b, image_size=(h,w))
        h_t3, c_t3 = self.encoder_3.init_hidden(batch_size=b, image_size=(h,w))
        h_t4, c_t4 = self.encoder_4.init_hidden(batch_size=b, image_size=(h,w))
        h_t5, c_t5 = self.encoder_5.init_hidden(batch_size=b, image_size=(h,w))
        h_t6, c_t6 = self.encoder_6.init_hidden(batch_size=b, image_size=(h,w))
        h_t7, c_t7 = self.decoder_1.init_hidden(batch_size=b, image_size=(h,w))
        h_t8, c_t8 = self.decoder_2.init_hidden(batch_size=b, image_size=(h,w))
        h_t9, c_t9 = self.decoder_3.init_hidden(batch_size=b, image_size=(h,w))
        h_t10, c_t10 = self.decoder_4.init_hidden(batch_size=b, image_size=(h,w))
        h_t11, c_t11 = self.decoder_5.init_hidden(batch_size=b, image_size=(h,w))
        h_t12, c_t12 = self.decoder_6.init_hidden(batch_size=b, image_size=(h,w))

        # output production through Encoder-Decoder structure
        outputs = []
        
        # Encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1(input_tensor=x[:,t,:,:,:],cur_state=[h_t,c_t])
            h_t2, c_t2 = self.encoder_2(input_tensor=h_t,cur_state=[h_t2,c_t2])
            h_t3, c_t3 = self.encoder_3(input_tensor=h_t2,cur_state=[h_t3,h_t3])
            h_t4, c_t4 = self.encoder_4(input_tensor=h_t3,cur_state=[h_t4,h_t4])
            h_t5, c_t5 = self.encoder_5(input_tensor=h_t4,cur_state=[h_t5,h_t5])
            h_t6, c_t6 = self.encoder_6(input_tensor=h_t5,cur_state=[h_t6,h_t6])

        # Decoder
        for t in range(future_step):
            h_t7, c_t7 = self.decoder_1(input_tensor=h_t6,cur_state=[h_t7,c_t7])
            h_t8, c_t8 = self.decoder_2(input_tensor=h_t7,cur_state=[h_t8,c_t8])
            h_t9, c_t9 = self.decoder_3(input_tensor=h_t8,cur_state=[h_t9,c_t9])
            h_t10, c_t10 = self.decoder_4(input_tensor=h_t9,cur_state=[h_t10,c_t10])
            h_t11, c_t11 = self.decoder_5(input_tensor=h_t10,cur_state=[h_t11,c_t11])
            h_t12, c_t12 = self.decoder_6(input_tensor=h_t11,cur_state=[h_t12,c_t12]) 
            outputs += [h_t12]            

        # Calculating output values using activation function
        outputs = torch.stack(outputs,1)
        outputs = outputs.permute(0,2,1,3,4)
        outputs = torch.nn.ReLU()(outputs)

        return outputs
