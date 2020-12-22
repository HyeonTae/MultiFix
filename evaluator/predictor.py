import torch
from torch.autograd import Variable

class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab, pad):
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pad = pad

    def get_decoder_features(self, src_seq):
        src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        with torch.no_grad():
            _, _, other = self.model(src_id_seq, [len(src_seq)])

        return other

    def get_decoder_features_batch(self, src_seq):
        lengths = []
        
        max_len = max([len(src_seq[i]) for i in range(len(src_seq))])
        
        for i, seq in zip(range(len(src_seq)),src_seq):
            lengths.append(len(seq))
            src_id = [self.src_vocab.stoi[tok] for tok in seq]
            if len(seq) < max_len:
                for j in range(max_len - len(seq)):
                    src_id.append(self.pad)
            #tmp = torch.LongTensor(src_id).view(1, -1)
            
            if i == 0:
                src_id_seq = torch.LongTensor(src_id).view(1, -1)
            else:
                tmp = torch.LongTensor(src_id).view(1, -1)
                src_id_seq = torch.cat((src_id_seq, tmp), dim=0)

        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        with torch.no_grad():
            _, _, other = self.model(src_id_seq, lengths)

        return other

    def predict(self, src_seq):
        other = self.get_decoder_features(src_seq)

        length = other['length'][0]

        tgt_att_list = []
        encoder_outputs = []
        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        if 'attention_score' in list(other.keys()):
            for i in range(len(other['attention_score'][0][0])):
                tgt_att_list.append([other['attention_score'][di][0].data[i].cpu().numpy() for di in range(length)])
            encoder_outputs = other['encoder_outputs'].cpu().numpy()

        if other['encoder_action'] is not None:
            action = torch.cat((other['encoder_action'],
                other['decoder_action'][:, :length-1, :]), dim=0).cpu().numpy()
        else:
            action = None

        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq, tgt_att_list, encoder_outputs, action

    def predict_batch(self, src_seq):
        batch_size = len(src_seq)
        other = self.get_decoder_features_batch(src_seq)

        tgt_id_seq = list()
        lengths = other['length']
        for i in range(batch_size):
            tgt_id_seq.append([other['sequence'][di][i].data[0] for di in range(lengths[i])])

        tgt_seq = list()
        for seq in tgt_id_seq:
            tmp = [self.tgt_vocab.itos[tok] for tok in seq]
            tgt_seq.append(" ".join(tmp[:-1]))

        return tgt_seq

    def predict_n_batch(self, src_seq, n = 1):
        batch_size = len(src_seq)
        other = self.get_decoder_features_batch(src_seq)

        result_list = list()

        for i in range(batch_size):
            result = []

            for x in range(0, int(n)):
                length = other['topk_length'][i][x]
                tgt_id_seq = [other['topk_sequence'][di][i, x, 0].data for di in range(length)]
                tgt_seq = ' '.join([self.tgt_vocab.itos[tok] for tok in tgt_id_seq][:-1])

                result.append(tgt_seq)

            result_list.append(result)

        return result_list
