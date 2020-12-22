import torch
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import copy


def _inflate(tensor, times, dim):
        repeat_dims = [1] * tensor.dim()
        repeat_dims[dim] = times
        return tensor.repeat(*repeat_dims)

class TopKDecoder(torch.nn.Module):
    def __init__(self, decoder_rnn, k):
        super(TopKDecoder, self).__init__()
        self.rnn = decoder_rnn
        self.k = k
        self.hidden_size = self.rnn.hidden_size
        self.V = self.rnn.output_size
        self.SOS = self.rnn.sos_id
        self.EOS = self.rnn.eos_id

    def forward(self, inputs=None, input_lengths=None, encoder_hidden=None, encoder_outputs=None,
            function=F.log_softmax, teacher_forcing_ratio=0, retain_output_probs=True):
        inputs, batch_size, max_length = self.rnn._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                                 function, teacher_forcing_ratio)

        self.pos_index = Variable(torch.LongTensor(range(batch_size)) * self.k).view(-1, 1) # batch_size x 1
        
        # Inflate the initial hidden states to be of size: b*k x h
        encoder_hidden = self.rnn._init_state(encoder_hidden)
        if encoder_hidden is None:
            hidden = None
        else:
            if isinstance(encoder_hidden, tuple):
                hidden = tuple([torch.repeat_interleave(h, self.k, dim=1) for h in encoder_hidden])
                #hidden = tuple([_inflate(h, self.k, 1) for h in encoder_hidden])
            else:
                hidden = torch.repeat_interleave(encoder_hidden, self.k, dim=1)
                #hidden = _inflate(encoder_hidden, self.k, 1)

        # ... same idea for encoder_outputs and decoder_outputs
        if self.rnn.use_attention:
            inflated_encoder_outputs = torch.repeat_interleave(encoder_outputs, self.k, dim=0)
            #inflated_encoder_outputs = _inflate(encoder_outputs, self.k, 0)
        else:
            inflated_encoder_outputs = None

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = torch.Tensor(batch_size * self.k, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(0, batch_size)]), 0.0)
        sequence_scores = Variable(sequence_scores)

        # Initialize the input vector
        input_var = Variable(torch.transpose(torch.LongTensor([[self.SOS] * batch_size * self.k]), 0, 1))
        #input_var = torch.LongTensor([self.SOS] * batch_size * self.k).view(-1, 1)
        
        if torch.cuda.is_available():
            input_var = input_var.cuda()

        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()        
        
        decoder_action = torch.tensor(()).to(device)
        
        input_len = copy.deepcopy(input_lengths)
        input_len = torch.repeat_interleave(torch.tensor(input_len), self.k, dim=0)
        #input_len = _inflate(torch.tensor(input_len), self.k, 0)
        # encoder_hidden[0] shape is (num_layers, batch_size, hidden_dim), input_len shape is (batch_size)
        decoder_input = input_var[:, 0].unsqueeze(1) ## (batchsize x k) x 1  
        
        for di in range(0, max_length):
            
            decoder_pos = None
            if self.rnn.position_embedding == "sin":
                pos, input_len = self.rnn.sin_encoding(input_var.cpu().tolist(),
                    batch_size * self.k, 1, input_len, self.rnn.embedding_size)
                decoder_pos = pos[:, 0].unsqueeze(1)
            elif self.rnn.position_embedding == "length":
                pos, input_len = self.rnn.length_encoding(input_var.cpu().tolist(),
                    batch_size * self.k, 1, input_len)  # pair of (batch_size x max_len x embed_size), input_len
                decoder_pos = pos[:, 0].unsqueeze(1)                                                        
            #decoder_pos = _inflate(decoder_pos, self.k, 0)

            # Run the RNN one step forward
            log_softmax_output, hidden, step_attn = self.rnn.forward_step(input_var, decoder_pos, hidden,
                                                                  inflated_encoder_outputs, function=function)
            # If doing local backprop (e.g. supervised training), retain the output layer
            if retain_output_probs:
                stored_outputs.append(log_softmax_output)
            # To get the full sequence scores for the new candidates, add the local scores for t_i to the predecessor scores for t_(i-1)
            #sequence_scores2 = _inflate(sequence_scores, self.V, 1) # bk x vocab_size
            
            sequence_scores = torch.repeat_interleave(sequence_scores, self.V, dim=1)
            
            if torch.cuda.is_available():
                sequence_scores = sequence_scores.cuda()
                self.pos_index = self.pos_index.cuda()
                                        
                    
            sequence_scores += log_softmax_output.squeeze(1)
            
            scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k, dim=1)   # from (k x b x v) to (b x k) (applies to both)
            
            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            input_var = (candidates % self.V).view(batch_size * self.k, 1)                                                
            sequence_scores = scores.view(batch_size * self.k, 1)            
    
            # Update fields for next timestep
            predecessors = (candidates / self.V + self.pos_index.expand_as(candidates)).view(batch_size * self.k, 1)
            
            if isinstance(hidden, tuple):
                hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in hidden])
            else:
                hidden = hidden.index_select(1, predecessors.squeeze())

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = input_var.data.eq(self.EOS)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)
            stored_hidden.append(hidden)

        # Do backtracking to return the optimal values
        output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
                                                    stored_predecessors, stored_emitted_symbols,
                                                    stored_scores, batch_size, self.hidden_size)

        # Build return objects
        decoder_outputs = [step[:, 0, :] for step in output]
        if isinstance(h_n, tuple):
            decoder_hidden = tuple([h[:, :, 0, :] for h in h_n])
        else:
            decoder_hidden = h_n[:, :, 0, :]
        metadata = {}
        metadata['inputs'] = inputs
        metadata['output'] = output
        metadata['h_t'] = h_t
        metadata['score'] = s
        metadata['topk_length'] = l
        metadata['topk_sequence'] = p
        metadata['length'] = [seq_len[0] for seq_len in l]
        metadata['sequence'] = [seq[0] for seq in p]
        return decoder_outputs, decoder_hidden, metadata

    def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, b, hidden_size):
        lstm = isinstance(nw_hidden[0], tuple)

        # initialize return variables given different types
        output = list()
        h_t = list()
        p = list()
        if lstm:
            state_size = nw_hidden[0][0].size()
            h_n = tuple([torch.zeros(state_size), torch.zeros(state_size)])
        else:
            h_n = torch.zeros(nw_hidden[0].size())
            
        l = [[self.rnn.max_length] * self.k for _ in range(b)]

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(b, self.k).topk(self.k)
        # initialize the sequence scores with the sorted last step beam scores
        
        s = torch.Tensor(b, self.k)
        s.fill_(-float('Inf'))
        
        if torch.cuda.is_available():
            s = s.cuda()

        batch_eos_found = [0] * b   # the number of EOS found
                                    # in the backward loop below for each batch

        t = self.rnn.max_length - 1
        
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)
        
        while t >= 0:
            # Re-order the variables with the back pointer
            current_output = nw_output[t].index_select(0, t_predecessors)
            if lstm:
                current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
            else:
                current_hidden = nw_hidden[t].index_select(1, t_predecessors)
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            eos_indices = symbols[t].data.squeeze(1).eq(self.EOS).nonzero()                        
            
            
            if eos_indices.dim() > 0:
                
                for i in range(eos_indices.size(0)-1, -1, -1):
                    
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / self.k)
                    
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                    
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.k + res_k_idx
                    # Replace the old information in return variables
                    # with the new ended sequence information
                    
                    min_score, min_index = torch.min(s, 1)                                        
                    
                    if min_score[b_idx] < scores[t][idx.item()].item():
                        res_k_idx = min_index[b_idx]
                        res_idx = b_idx * self.k + res_k_idx
                        
                        t_predecessors[res_idx] = predecessors[t][idx.item()]

                        current_output[res_idx, :] = nw_output[t][idx.item(), :]
                        if lstm:
                            current_hidden[0][:, res_idx, :] = nw_hidden[t][0][:, idx.item(), :]
                            current_hidden[1][:, res_idx, :] = nw_hidden[t][1][:, idx.item(), :]
                            h_n[0][:, res_idx, :] = nw_hidden[t][0][:, idx.item(), :].data
                            h_n[1][:, res_idx, :] = nw_hidden[t][1][:, idx.item(), :].data
                        else:
                            current_hidden[:, res_idx, :] = nw_hidden[t][:, idx.item(), :]
                            h_n[:, res_idx, :] = nw_hidden[t][:, idx.item(), :].data

                        current_symbol[res_idx, :] = symbols[t][idx.item()]

                        s[b_idx, res_k_idx] = scores[t][idx.item()].item()
                        l[b_idx][res_k_idx] = t + 1
                    
            # record the back tracked results
            output.append(current_output)
            h_t.append(current_hidden)
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        if torch.cuda.is_available():
            h_n = [h.cuda() for h in h_n]
        
        s, re_sorted_idx = s.topk(self.k)
                                
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        output = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(output)]
        p = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(p)]
        if lstm:
            h_t = [tuple([h.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for h in step]) for step in reversed(h_t)]
            h_n = tuple([h.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size) for h in h_n])
        else:
            h_t = [step.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for step in reversed(h_t)]
            h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size)
        s = s.data

        return output, h_t, h_n, s, l, p

    def _mask_symbol_scores(self, score, idx, masking_score=-float('inf')):
            score[idx] = masking_score

    def _mask(self, tensor, idx, dim=0, masking_score=-float('inf')):
        if len(idx.size()) > 0:
            indices = idx[:, 0]
            tensor.index_fill_(dim, indices, masking_score)
