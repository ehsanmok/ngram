"""
Raw C port to Mojo
Run:
    mojo ngram_v00.mojo
OR
    mojo build ngram_v00.mojo && ./ngram_v00

How many issues can you spot before moving to the next version ngram_v01.mojo?
"""

import sys
from sys.ffi import external_call
from memory import memset_zero

alias c_char = UInt8
alias c_int = Int32
alias c_long = UInt64
alias c_void = UInt8
alias c_size_t = Int
alias c_uint32_t = UInt32
alias c_uint64_t = UInt64
alias c_float = Float32

alias NUM_TOKENS = 27
alias EOF = -1
alias EOT_TOKEN = 0


fn powi(base: c_int, exp: c_int) -> c_size_t:
    var result = 1
    for i in range(exp):
        result *= int(base)

    return result


fn logf(arg: c_float) -> c_float:
    return external_call["logf", c_float, c_float](arg)


fn expf(arg: c_float) -> c_float:
    return external_call["expf", c_float, c_float](arg)


fn random_u32(state: UnsafePointer[c_uint64_t]) -> c_uint32_t:
    state[0] ^= state[0] >> 12
    state[0] ^= state[0] << 25
    state[0] ^= state[0] >> 27
    return ((state[0] * 0x2545F4914F6CDD1D) >> 32).cast[DType.uint32]()


fn random_f32(state: UnsafePointer[c_uint64_t]) -> c_float:
    return (random_u32(state) >> 8).cast[DType.float32]() / c_float(16777216.0)


fn sample_discrete(
    probs: UnsafePointer[c_float], n: c_int, coinf: c_float
) raises -> c_int:
    debug_assert(
        coinf >= 0.0 and coinf < 1.0,
        String.format("coinf must be between 0 and 1 but given {}", coinf),
    )
    var cdf = c_float(0.0)
    for i in range(n):
        var probs_i = probs[i]
        debug_assert(
            probs_i >= 0.0 and probs_i <= 1.0,
            String.format(
                "probs_i must be between 0 and 1 but given {}", probs_i
            ),
        )
        cdf += probs_i
        if coinf < cdf:
            return i

    return n - 1


fn tokenizer_encode(c: c_int) raises -> c_int:
    var newline = c_int(ord("\n"))
    debug_assert(
        c == newline or (c_int(ord("a")) <= c and c <= c_int(ord("z"))),
        "characters a-z are encoded as 1-26, and '\n' is encoded as 0",
    )
    return c_int(EOT_TOKEN) if c == newline else c_int(c) - c_int(ord("a")) + 1


fn tokenizer_decode(token: c_int) raises -> c_int:
    debug_assert(
        token >= 0 and token <= NUM_TOKENS,
        String.format(
            "token must be between 0 to NUM_TOKENS={} but given {}",
            NUM_TOKENS,
            token,
        ),
    )
    return (
        c_int(ord("\n")) if token
        == c_int(EOT_TOKEN) else c_int(ord("a")) + c_int(token) - 1
    )


struct NgramModel:
    var seq_len: c_int
    var vocab_size: c_int
    var smoothing: c_float
    var num_counts: c_size_t
    var counts: UnsafePointer[c_uint32_t]
    var ravel_buffer: UnsafePointer[c_int]

    fn __init__(
        inout self, vocab_size: c_int, seq_len: c_int, smoothing: c_float
    ) raises:
        debug_assert(vocab_size > 0, "vocab_size must be a positive integer.")
        debug_assert(
            seq_len >= 1 and seq_len <= 6,
            "seq_len must be an integer between (including) 1 to 6.",
        )
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.smoothing = smoothing
        self.num_counts = powi(vocab_size, seq_len)
        self.counts = UnsafePointer[c_uint32_t].alloc(self.num_counts)
        memset_zero(self.counts, self.num_counts)
        self.ravel_buffer = UnsafePointer[c_int].alloc(int(self.seq_len))

    fn __del__(owned self):
        self.counts.free()
        self.ravel_buffer.free()

    fn train(inout self, tape: UnsafePointer[c_int]) raises:
        var offset = ravel_index(tape, self.seq_len, self.vocab_size)
        debug_assert(
            offset >= 0 and offset < self.num_counts,
            String.format(
                "offset must be between 0 to num_counts={} but give {}",
                self.num_counts,
                offset,
            ),
        )
        self.counts[offset] += 1

    fn inference(
        self, tape: UnsafePointer[c_int], probs: UnsafePointer[c_float]
    ) raises:
        """
        Here, tape is of length `seq_len - 1`, and we want to predict the next token
        probs should be a pre-allocated buffer of size `vocab_size`.
        """
        # copy the tape into the buffer and set the last element to zero
        for i in range(self.seq_len - 1):
            self.ravel_buffer[i] = tape[i]

        self.ravel_buffer[int(self.seq_len) - 1] = 0
        # find the offset into the counts array based on the context
        var offset = ravel_index(
            self.ravel_buffer, self.seq_len, self.vocab_size
        )
        # seek to the row of counts for this context
        var counts_row = self.counts.offset(offset)

        # calculate the sum of counts in the row
        var row_sum = (self.vocab_size).cast[DType.float32]() * self.smoothing
        for i in range(self.vocab_size):
            row_sum += (counts_row[i]).cast[DType.float32]()

        if row_sum == c_float(0.0):
            # the entire row of counts is zero, so let's set uniform probabilities
            var uniform_prob = 1.0 / (self.vocab_size).cast[DType.float32]()
            for i in range(self.vocab_size):
                probs[i] = uniform_prob
        else:
            # normalize the row of counts into probabilities
            var scale = c_float(1.0) / row_sum
            for i in range(self.vocab_size):
                var counts_i = (counts_row[i]).cast[
                    DType.float32
                ]() + self.smoothing
                probs[i] = scale * counts_i


fn ravel_index(
    index: UnsafePointer[c_int], n: c_int, dim: c_int
) raises -> c_size_t:
    var index1d = 0
    var multiplier = 1
    for i in range(n - 1, 0, -1):
        var ix = index[i]
        debug_assert(
            ix >= 0 and ix < dim,
            String.format(
                "ix must be between 0 and dim={} but given {}", dim, ix
            ),
        )
        index1d += multiplier * int(ix)
        multiplier *= int(dim)

    return index1d


struct Tape:
    var n: c_int
    var length: c_int
    var buffer: UnsafePointer[c_int]

    fn __init__(inout self, length: c_int) raises:
        debug_assert(
            length >= 0,
            String.format(
                "length must a non-negative integer but given {}", length
            ),
        )
        self.length = length
        self.n = 0
        self.buffer = UnsafePointer[c_int]()
        if length > 0:
            self.buffer = UnsafePointer[c_int].alloc(int(length))

    fn __del__(owned self):
        self.buffer.free()

    fn set(inout self, val: c_int) raises:
        if not self.buffer:
            raise "length must be set to non-zero"

        for i in range(self.length):
            self.buffer[i] = val

    fn update(inout self, token: c_int) -> c_int:
        if self.length == 0:
            return 1

        for i in range(self.length - 1):
            self.buffer[i] = self.buffer[i + 1]

        self.buffer[int(self.length) - 1] = token

        if self.n < self.length:
            self.n += 1

        return (self.n == self.length).cast[DType.int32]()


struct FILE:
    pass


fn fopen(path: String, mode: String) raises -> UnsafePointer[FILE]:
    var path_ptr = _as_char_ptr(path)
    var mode_ptr = _as_char_ptr(mode)
    var stream = external_call[
        "fopen",
        UnsafePointer[FILE],
        UnsafePointer[c_char],
        UnsafePointer[c_char],
    ](path_ptr, mode_ptr)
    if stream == UnsafePointer[FILE]():
        raise Error("Cannot open the file")

    mode_ptr.free()
    path_ptr.free()
    return stream


fn _as_char_ptr(s: String) -> UnsafePointer[c_char]:
    var nelem = len(s)
    var ptr = UnsafePointer[c_char]().alloc(nelem + 1)
    for i in range(len(s)):
        ptr[i] = ord(s[i])

    ptr[nelem] = 0
    return ptr


fn fclose(stream: UnsafePointer[FILE]) raises:
    debug_assert(stream != UnsafePointer[FILE](), "File must be opened first")
    var ret = external_call["fclose", c_int, UnsafePointer[FILE]](stream)
    if ret:
        raise Error("File cannot be closed")

    return


fn fgetc(stream: UnsafePointer[FILE]) raises -> c_int:
    debug_assert(stream != UnsafePointer[FILE](), "File must be opened first")
    var ret = external_call["fgetc", c_int, UnsafePointer[FILE]](stream)
    if not ret:  # null on error
        raise Error("Error in fgetc")

    return ret


struct DataLoader:
    var file: UnsafePointer[FILE]
    var seq_len: c_int
    var tape: Tape

    fn __init__(inout self, path: String, seq_len: c_int) raises:
        self.file = fopen(path, mode="r")
        self.seq_len = seq_len
        self.tape = Tape(self.seq_len)

    fn __del__(owned self):
        try:
            fclose(self.file)
            _ = self.tape^

        except:
            return

    fn next(inout self) raises -> c_int:
        while True:
            var c = fgetc(self.file)
            if c == EOF:
                break

            var token = tokenizer_encode(c)
            var ready = self.tape.update(token)
            if ready:
                return 1

        return 0


fn error_usage():
    print("Usage: ./ngram [options]", end="\n")
    print("Options:", end="\n")
    print(" -n <int> n-gram model arity (default 5)", end="\n")
    print(" -s <float> smoothing factor (default 0.1)", end="\n")
    sys.exit(1)


fn main() raises:
    var args = sys.argv()
    var argc = len(args)
    var seq_len = c_int(5)
    var smoothing = c_float(0.1)
    for i in range(1, argc, 2):
        if i + 1 >= argc:
            return error_usage()
        if args[i][0] != "-":
            return error_usage()
        if len(args[i]) != 2:
            return error_usage()
        if args[i][1] == "n":
            seq_len = atol(args[i + 1])
        elif args[i][1] == "s":
            smoothing = atof(args[i + 1]).cast[DType.float32]()
        else:
            return error_usage()

    # train the model
    var model = NgramModel(NUM_TOKENS, seq_len, smoothing)
    var train_loader = DataLoader("data/train.txt", seq_len)
    while train_loader.next():
        model.train(train_loader.tape.buffer)

    # allocate probs buffer for inference
    var probs = UnsafePointer[c_float].alloc(NUM_TOKENS)
    var sample_tape = Tape(seq_len - 1)
    sample_tape.set(EOT_TOKEN) # fill with EOT tokens to init
    var rng = UnsafePointer[c_uint64_t].alloc(1)
    rng[0] = 1337
    for _ in range(200):
        model.inference(sample_tape.buffer, probs)
        var coinf = random_f32(rng)
        var token = sample_discrete(probs, NUM_TOKENS, coinf)
        _ = sample_tape.update(token)
        var c = tokenizer_decode(token)
        print(chr(int(c)), end="")

    print("\n")

    # evaluate the test split loss
    var test_loader = DataLoader("data/test.txt", seq_len)
    var sum_loss = c_float(0)
    var count = 0
    while test_loader.next():
        # note that `inference` will only use the first seq_len - 1 tokens in buffer
        model.inference(test_loader.tape.buffer, probs)
        # and the last token in the tape buffer is the label
        var target = test_loader.tape.buffer[int(seq_len) - 1]
        # negative log likelihood loss
        sum_loss += -logf(probs[int(target)])
        count += 1

    var mean_loss = sum_loss / count
    var test_preplexity = expf(mean_loss)
    print(
        String.format(
            "test_loss {}, test_preplexity {}", mean_loss, test_preplexity
        )
    )
    rng.free()
    probs.free()
    return
