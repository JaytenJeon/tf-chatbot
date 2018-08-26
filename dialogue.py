class Dialogue:
    def __init__(self, path):
        self.dialogues = self.load_data(path)
        self.sentences = sum(self.dialogues, [])
        self.voc_arr = ['_P_', '_S_', '_E_', '_U_'] + self.make_voc()
        self.voc_dict = {voc: i for i, voc in enumerate(self.voc_arr)}
        self.voc_size = len(self.voc_arr)

        self.index_in_epoch = 0

    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            dialogues = []
            sentences = []
            for line in file:
                line = line.strip()
                if line == '':
                    dialogues.append(sentences)
                    sentences = []
                else:
                    sentences.append(line)
        return dialogues

    def tokenizer(self, sentence):
        sentence = re.sub("[.,!?\"':;)(]", ' ', sentence)
        tokens = sentence.split(' ')
        return tokens

    def tokens_to_ids(self, tokens):
        ids = [self.voc_dict[token] if token in self.voc_arr else self.voc_dict['_U_'] for token in tokens]
        return ids

    def ids_to_tokens(self, ids):
        tokens = [self.voc_arr[id] for id in ids]
        return tokens

    def pad(self, tokens, max_len):
        padded = tokens + ['_P_'] * (max_len - len(tokens))
        return padded

    def make_voc(self):
        voc_set = set()
        for sentence in self.sentences:
            voc_set.update(self.tokenizer(sentence))
        return list(voc_set)

    def make_seq_data(self, sentences):
        seq_data = [self.tokenizer(sentence) for sentence in sentences]
        return seq_data

    def max_len(self, batch_set):

        enc_max_len = max([len(batch_set[i]) for i in range(0, len(batch_set) - 1)]) + 1
        dec_max_len = max([len(batch_set[i + 1]) for i in range(0, len(batch_set) - 1)]) + 1

        max_len = max(enc_max_len, dec_max_len)

        return enc_max_len, dec_max_len, max_len

    def next_batch(self, batch_size):
        enc_batch = []
        dec_batch = []
        target_batch = []
        enc_seq_len = []
        dec_seq_len = []

        index = self.index_in_epoch
        batch_set = self.seq_data[index:index + batch_size]
        enc_max_len, dec_max_len, max_len = self.max_len(batch_set)

        self.index_in_epoch = self.index_in_epoch + batch_size
        if self.index_in_epoch > len(self.seq_data):
            self.index_in_epoch = 0

        for i in range(0, len(batch_set) - 1):
            enc_input = batch_set[i]
            dec_input = batch_set[i + 1]
            enc_seq_len.append(len(enc_input))
            dec_seq_len.append(len(dec_input) + 1)
            enc = self.tokens_to_ids(self.pad(enc_input, max_len))
            dec = self.tokens_to_ids(self.pad(['_S_'] + dec_input, max_len))
            target = self.tokens_to_ids(self.pad(dec_input + ['_E_'], dec_max_len))

            enc_batch.append(enc)
            dec_batch.append(dec)
            target_batch.append(target)

        return enc_batch, dec_batch, target_batch, enc_seq_len, dec_seq_len, dec_max_len

    def next_dialogue(self):
        enc_batch = []
        dec_batch = []
        target_batch = []
        enc_seq_len = []
        dec_seq_len = []

        index = self.index_in_epoch
        batch_set = self.make_seq_data(self.dialogues[index])

        enc_max_len, dec_max_len, max_len = self.max_len(batch_set)

        self.index_in_epoch += 1
        if self.index_in_epoch >= len(self.dialogues):
            self.index_in_epoch = 0

        for i in range(0, len(batch_set) - 1):
            enc_input = batch_set[i]
            dec_input = batch_set[i + 1]
            enc_seq_len.append(len(enc_input))
            dec_seq_len.append(len(dec_input) + 1)
            enc = self.tokens_to_ids(self.pad(enc_input, max_len))
            dec = self.tokens_to_ids(self.pad(['_S_'] + dec_input, max_len))
            target = self.tokens_to_ids(self.pad(dec_input + ['_E_'], dec_max_len))

            enc_batch.append(enc)
            dec_batch.append(dec)
            target_batch.append(target)

        return enc_batch, dec_batch, target_batch, enc_seq_len, dec_seq_len, dec_max_len

