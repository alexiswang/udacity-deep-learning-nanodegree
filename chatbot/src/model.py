import torch.nn as nn
import torch
import random
import torchtext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, embedding_size, dropout=0.5):
        
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        # self.embedding provides a vector representation of the inputs to our model
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        # self.lstm, accepts the vectorized input and passes a hidden state
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):       
        '''
        Inputs: x, the src vector
        Outputs: output, the encoder outputs
                hidden, the hidden state
                cell_state, the cell state
        '''
        # x = [src_len, batch_size]
        embedded = self.dropout(self.embedding(x)) 
        # embeded = [src_len, batch_size, emb_dim]

        output, (hidden, cell_state) = self.lstm(embedded)
        # output = [src_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        
        return output, hidden, cell_state
    

class Decoder(nn.Module):
      
    def __init__(self, output_size, hidden_size, embedding_size, dropout):
        
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        # self.embedding provides a vector representation of the target to our model
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # self.lstm, accepts the embeddings and outputs a hidden state
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=2, dropout=dropout)
        # self.ouput, predicts on the hidden state via a linear output layer  
        self.fc = nn.Linear(self.hidden_size, self.output_size)  
        self.softmax = nn.LogSoftmax(dim = 1) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden, cell_state):
        '''
        Inputs: x, input tensor is the index of the word
        Outputs: output, output tensor (1, vocab_len), the largest one indicate that word is predicated
                hidden, the hidden state, (1, 1, hidden_size)
                cell_state, cell state (1, 1, hidden_size)
        '''
        # x = [batch_size]
        x = x.unsqueeze(0)
        # x = [1, batch_size]
        embeded = self.dropout(self.embedding(x))
        # embedded = [1, batch_size, emb_dim]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        output, (hidden, cell_state) = self.lstm(embeded, (hidden, cell_state))
        # output = [seq_len(1), batch_size, hid_dim * n_direction]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        fc_out = self.fc(output[0])
        # fc_out = [batch_size, output_dim]
        output = self.softmax(fc_out)
        # output = [batch_size, output_dim]
        
        return output, hidden, cell_state
        
        

class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder):  
        super(Seq2Seq, self).__init__()     

        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hidden_size == decoder.hidden_size
   
    def forward(self, src, trg, teacher_forcing_ratio = 1):
        """
        Inputs: src, source data (question) tensor
                trg, target data (answer) tensor
                teacher_forcing, applied to the decoder in case of seq2seq models. It is a strategy for training Recurrent
                Neural Netowrks that uses ground truth as input instead of model output from prior time step as input. 
        Outputs: output, predited sequence tensor
        """

        output = {'decoder_output': []}      
        
        trg_len = trg.size(0) if trg is not None else 10

        encoder_output, encoder_hidden, cell_state = self.encoder(src)

        # first decoder input is 0 = SOS_token 
        decoder_input = torch.zeros(trg.size(1)).long().to(device)
        decoder_hidden = encoder_hidden
        for i in range(trg_len):

            decoder_output, decoder_hidden, cell_state = self.decoder(decoder_input, decoder_hidden, cell_state)
            output['decoder_output'].append(decoder_output)
            is_teacher = random.random() < teacher_forcing_ratio

            if self.training: 
                decoder_input = trg[i] if is_teacher else decoder_output.argmax(1)
            else:
                decoder_input = decoder_output.argmax(1)

        return output
    



