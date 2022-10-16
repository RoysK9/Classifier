import torch.nn as nn


class TimeDistributed(nn.Module):
    
    def __init__(self, module):
        super().__init__()
        self.td = module                                    # get module

    def forward(self, x):

        length = len(x.size())

        if(length == 5):
            batch,seq_len,ch,h,w = x.size()                  # get input shape (batch first)
            x    = x.contiguous().view(batch*seq_len,ch,h,w) # (seq_len*batch,ch,h,w)
            x    = self.td(x)                                # Do the module processing.
            args = list(x.size()[1:])                        # get list [ch,h,w] or [input_size]
            x    = x.view(batch, seq_len, *args)             # (seq_len,batch,ch,h,w)

        elif(length == 3):
            batch,seq_len,ch = x.size()                      # get input shape (batch first)
            x    = x.contiguous().view(batch*seq_len,ch)     # (seq_len*batch,ch,h,w)
            x    = self.td(x)                                # Do the module processing.
            args = list(x.size()[1:])                        # get list [ch,h,w] or [input_size]
            x    = x.view(batch, seq_len, *args)             # (seq_len,batch,ch,h,w)

        return x