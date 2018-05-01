import numpy as np
import io


class EmbeddingsModel(object):
    def __init__(self):
        super(EmbeddingsModel, self).__init__()

    def get_vsz(self):
        pass

    def get_dsz(self):
        pass

    def lookup(self, word, nullifabsent=True):
        pass


class Word2VecModel(EmbeddingsModel):

    def _read_file(self, filename, known_vocab, keep_unused):
        idx = 0
        with io.open(filename, "rb") as f:
            header = f.readline()
            vsz, self.dsz = map(int, header.split())

            self.nullv = np.zeros(self.dsz, dtype=np.float32)
            self.vocab["<PAD>"] = idx
            idx += 1

            word_vectors = [self.nullv]
            width = 4 * self.dsz

            for i in range(vsz):
                word = Word2VecModel._readtospc(f)
                raw = f.read(width)
                if keep_unused is False and word not in known_vocab:
                    continue

                # Otherwise add it to the list and remove from knownvocab
                if known_vocab and word in known_vocab:
                    known_vocab[word] = 0

                vec = np.fromstring(raw, dtype=np.float32)
                word_vectors.append(vec)

                self.vocab[word] = idx
                idx += 1
        return word_vectors, idx

    @staticmethod
    def _read_line_mmap(m, width, start):
        current = start+1
        while m[current:current+1] != b' ':
            current += 1

        vocab = m[start:current].decode('utf-8')
        raw = m[current+1:current+width+1]
        value = np.fromstring(raw, dtype=np.float32)
        return vocab, value, current+width + 1

    def _read_file_mm(self, filename, known_vocab, keep_unused):
        import mmap
        import contextlib
        idx = 0
        with open(filename, 'rb') as f:
            with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
                header_end = m[:50].find(b'\n')
                vsz, self.dsz = map(int, (m[:header_end]).split(b' '))
                self.nullv = np.zeros(self.dsz, dtype=np.float32)
                self.vocab["<PAD>"] = idx
                idx += 1
                current = header_end + 1
                word_vectors = [self.nullv]
                width = 4 * self.dsz
                for i in range(vsz):
                    word, vec, current = Word2VecModel._read_line_mmap(m, width, current)
                    if keep_unused is False and word not in known_vocab:
                        continue
                    if known_vocab and word in known_vocab:
                        known_vocab[word] = 0

                    word_vectors.append(vec)
                    self.vocab[word] = idx
                    idx += 1
                return word_vectors, idx

    def __init__(self, filename, known_vocab=None, unif_weight=None, keep_unused=False, normalize=False, use_mmap=False):
        super(Word2VecModel, self).__init__()

        #import time
        #print('MMAP', use_mmap)
        #start_time = time.time()
        uw = 0.0 if unif_weight is None else unif_weight
        self.vocab = {}

        reader = self._read_file_mm if use_mmap else self._read_file
        word_vectors, idx = reader(filename, known_vocab, keep_unused)
        #duration = time.time() - start_time
        #print('Load time (s) {:.4f}, words/s {:.4f}'.format(duration, len(word_vectors)/duration))
        if known_vocab is not None:
            unknown = {v: cnt for v, cnt in known_vocab.items() if cnt > 0}
            for v in unknown:
                word_vectors.append(np.random.uniform(-uw, uw, self.dsz))
                self.vocab[v] = idx
                idx += 1

        if normalize is True:
            for i in range(len(word_vectors)):
                norm = np.linalg.norm(word_vectors[i])
                word_vectors[i] = word_vectors[i] if norm == 0.0 else word_vectors[i]/norm

        self.weights = np.array(word_vectors)
        self.vsz = self.weights.shape[0] - 1

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    @staticmethod
    def _readtospc(f):

        s = bytearray()
        ch = f.read(1)

        while ch != b'\x20':
            s.extend(ch)
            ch = f.read(1)
        s = s.decode('utf-8')
        return s.strip()

    def lookup(self, word, nullifabsent=True):
        if word in self.vocab:
            return self.weights[self.vocab[word]]
        if nullifabsent:
            return None
        return self.nullv


class GloVeModel(EmbeddingsModel):

    def __init__(self, filename, known_vocab=None, unif_weight=None, keep_unused=False, normalize=False):
        super(GloVeModel, self).__init__()
        uw = 0.0 if unif_weight is None else unif_weight
        self.vocab = {}
        idx = 1

        word_vectors = []
        with io.open(filename, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if keep_unused is False and word not in known_vocab:
                    continue

                # Otherwise add it to the list and remove from knownvocab
                if known_vocab and word in known_vocab:
                    known_vocab[word] = 0
                vec = np.asarray(values[1:], dtype=np.float32)
                word_vectors.append(vec)
                self.vocab[word] = idx
                idx += 1
            self.dsz = vec.shape[0]
            self.nullv = np.zeros(self.dsz, dtype=np.float32)
            word_vectors = [self.nullv] + word_vectors
            self.vocab["<PAD>"] = 0

        if known_vocab is not None:
            unknown = {v: cnt for v, cnt in known_vocab.items() if cnt > 0}
            for v in unknown:
                word_vectors.append(np.random.uniform(-uw, uw, self.dsz))
                self.vocab[v] = idx
                idx += 1

        if normalize is True:
            for i in range(len(word_vectors)):
                norm = np.linalg.norm(word_vectors[i])
                word_vectors[i] = word_vectors[i] if norm == 0.0 else word_vectors[i]/norm

        self.weights = np.array(word_vectors)
        self.vsz = self.weights.shape[0] - 1

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def lookup(self, word, nullifabsent=True):
        if word in self.vocab:
            return self.weights[self.vocab[word]]
        if nullifabsent:
            return None
        return self.nullv


class RandomInitVecModel(EmbeddingsModel):

    def __init__(self, dsz, known_vocab, counts=True, unif_weight=None):
        super(RandomInitVecModel, self).__init__()
        uw = 0.0 if unif_weight is None else unif_weight
        self.vocab = {}
        self.vocab["<PAD>"] = 0
        self.dsz = dsz
        self.vsz = 0

        if counts is True:
            attested = [v for v, cnt in known_vocab.items() if cnt > 0]
            for k, v in enumerate(attested):
                self.vocab[v] = k + 1
                self.vsz += 1
        else:
            print('Restoring existing vocab')
            self.vocab = known_vocab
            self.vsz = len(self.vocab) - 1

        self.weights = np.random.uniform(-uw, uw, (self.vsz+1, self.dsz))

        self.nullv = np.zeros(self.dsz, dtype=np.float32)
        self.weights[0] = self.nullv

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def lookup(self, word, nullifabsent=True):
        if word in self.vocab:
            return self.weights[self.vocab[word]]
        if nullifabsent:
            return None
        return self.nullv
