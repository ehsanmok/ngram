# """
# Optimize

# Run:
#     mojo ngram_v02.mojo -n 5
# OR
#     mojo build ngram_v02.mojo && ./ngram_v02 -n 5
# """

# import math
# import sys
# from sys.ffi import external_call
# from utils import InlineArray

# alias c_char = UInt8
# alias c_int = Int32
# alias c_long = UInt64
# alias c_void = UInt8
# alias c_size_t = Int
# alias c_uint32_t = UInt32
# alias c_uint64_t = UInt64
# alias c_float = Float32

# alias NUM_TOKENS = 27
# alias EOF = -1
# alias EOT_TOKEN = 0

# struct RNG:
#     var rng: UnsafePointer[c_uint64_t]

#     @always_inline("nodebug")
#     fn __init__(inout self, seed: Int):
#         self.rng = UnsafePointer[c_uint64_t].alloc(1)
#         self.rng.init_pointee_copy(seed)

#     @always_inline("nodebug")
#     fn __del__(owned self):
#         self.rng.free()

#     @always_inline
#     fn random_u32(self) -> c_uint32_t:
#         self.rng[0] ^= self.rng[0] >> 12
#         self.rng[0] ^= self.rng[0] << 25
#         self.rng[0] ^= self.rng[0] >> 27
#         return ((self.rng[0] * 0x2545F4914F6CDD1D) >> 32).cast[DType.uint32]()

#     @always_inline
#     fn random_f32(self) -> c_float:
#         return (self.random_u32() >> 8).cast[DType.float32]() / c_float(16777216.0)

# @always_inline
# def sample_discrete(
#     probs: InlineArray[c_float, NUM_TOKENS], n: c_int, coinf: c_float
# ) -> c_int:
#     debug_assert(
#         coinf >= 0.0 and coinf < 1.0,
#         String.format("coinf must be between 0 and 1 but given {}", coinf),
#     )
#     cdf = c_float(0.0)
#     for i in range(n):
#         probs_i = probs[i]
#         debug_assert(
#             probs_i >= 0.0 and probs_i <= 1.0,
#             String.format(
#                 "probs_i must be between 0 and 1 but given {}", probs_i
#             ),
#         )
#         cdf += probs_i
#         if coinf < cdf:
#             return i

#     return n - 1

# @always_inline
# def tokenizer_encode(c: c_int) -> c_int:
#     newline = c_int(ord("\n"))
#     debug_assert(
#         c == newline or (c_int(ord("a")) <= c and c <= c_int(ord("z"))),
#         "characters a-z are encoded as 1-26, and '\n' is encoded as 0",
#     )
#     return c_int(EOT_TOKEN) if c == newline else c_int(c) - c_int(ord("a")) + 1

# @always_inline
# def tokenizer_decode(token: c_int) -> c_int:
#     debug_assert(
#         token >= 0 and token <= NUM_TOKENS,
#         String.format(
#             "token must be between 0 to NUM_TOKENS={} but given {}",
#             NUM_TOKENS,
#             token,
#         ),
#     )
#     return (
#         c_int(ord("\n")) if token
#         == c_int(EOT_TOKEN) else c_int(ord("a")) + c_int(token) - 1
#     )


# struct NgramModel:
#     var seq_len: c_int
#     var vocab_size: c_int
#     var smoothing: c_float
#     var num_counts: c_size_t
#     var counts: UnsafePointer[c_uint32_t]
#     var ravel_buffer: UnsafePointer[c_int]

#     @always_inline("nodebug")
#     fn __init__(
#         inout self, vocab_size: c_int, seq_len: c_int, smoothing: c_float
#     ) raises:
#         debug_assert(vocab_size > 0, "vocab_size must be a positive integer.")
#         debug_assert(
#             seq_len >= 1 and seq_len <= 6,
#             "seq_len must be an integer between (including) 1 to 6.",
#         )
#         self.vocab_size = vocab_size
#         self.seq_len = seq_len
#         self.smoothing = smoothing
#         self.num_counts = int(vocab_size**seq_len)
#         self.counts = UnsafePointer[c_uint32_t].alloc(self.num_counts)
#         memset_zero(self.counts, self.num_counts)
#         self.ravel_buffer = UnsafePointer[c_int].alloc(int(self.seq_len))

#     @always_inline("nodebug")
#     fn __del__(owned self):
#         self.counts.free()
#         self.ravel_buffer.free()

#     @always_inline
#     @staticmethod
#     fn _ravel_index(index: UnsafePointer[c_int], n: c_int, dim: c_int) raises -> c_size_t:
#         var index1d = 0
#         var multiplier = 1
#         for i in range(n - 1, 0, -1):
#             var ix = index[i] # ???assumes index has been initialized
#             debug_assert(
#                 ix >= 0 and ix < dim,
#                 String.format(
#                     "ix must be between 0 and dim={} but given {}", dim, ix
#                 ),
#             )
#             index1d += multiplier * int(ix)
#             multiplier *= int(dim)

#         return index1d

#     @always_inline
#     def train(inout self, tape: Span[c_int, _]):
#         offset = self._ravel_index(tape.unsafe_ptr(), self.seq_len, self.vocab_size)
#         debug_assert(
#             offset >= 0 and offset < self.num_counts,
#             String.format(
#                 "offset must be between 0 to num_counts={} but give {}",
#                 self.num_counts,
#                 offset,
#             ),
#         )
#         self.counts[offset] += 1

#     @always_inline
#     def inference(
#         self, tape: Span[c_int, _], probs: InlineArray[c_float, NUM_TOKENS]
#     ):
#         """
#         Here, tape is of length `seq_len - 1`, and we want to predict the next token
#         probs should be a pre-allocated buffer of size `vocab_size`.
#         """
#         # copy the tape into the buffer and set the last element to zero
#         for i in range(self.seq_len - 1):
#             # tape is already initialized for inference so it's safe to index tape[i]
#             self.ravel_buffer.offset(i).init_pointee_copy(tape[i])

#         self.ravel_buffer.offset(int(self.seq_len) - 1).init_pointee_copy(0)
#         # find the offset into the counts array based on the context
#         offset = self._ravel_index(
#             self.ravel_buffer, self.seq_len, self.vocab_size
#         )
#         # seek to the row of counts for this context
#         counts_row = self.counts.offset(offset)

#         # calculate the sum of counts in the row
#         row_sum = (self.vocab_size).cast[DType.float32]() * self.smoothing
#         for i in range(self.vocab_size):
#             row_sum += (counts_row[i]).cast[DType.float32]()

#         if row_sum == c_float(0.0):
#             # the entire row of counts is zero, so let's set uniform probabilities
#             uniform_prob = 1.0 / (self.vocab_size).cast[DType.float32]()
#             for i in range(self.vocab_size):
#                 probs.unsafe_ptr().offset(i).init_pointee_copy(uniform_prob)
#         else:
#             # normalize the row of counts into probabilities
#             scale = c_float(1.0) / row_sum
#             for i in range(self.vocab_size):
#                 counts_i = (counts_row[i]).cast[
#                     DType.float32
#                 ]() + self.smoothing
#                 probs.unsafe_ptr().offset(i).init_pointee_copy(scale * counts_i)


# struct Tape:
#     var n: c_int
#     var length: c_int
#     var buffer: UnsafePointer[c_int]

#     @always_inline("nodebug")
#     fn __init__(inout self, length: c_int) raises:
#         debug_assert(
#             length >= 0,
#             String.format(
#                 "length must a non-negative integer but given {}", length
#             ),
#         )
#         self.length = length
#         self.n = 0
#         self.buffer = UnsafePointer[c_int]()
#         if length > 0:
#             self.buffer = UnsafePointer[c_int].alloc(int(length))
#             # need to initialize to make sure ravel_index `index[i]` is valid for `train`
#             memset_zero(self.buffer, int(length))

#     @always_inline("nodebug")
#     fn __del__(owned self):
#         self.buffer.free()

#     @always_inline("nodebug")
#     fn span(ref [_] self) -> Span[c_int, __lifetime_of(self)]:
#         return Span[c_int, __lifetime_of(self)](unsafe_ptr=self.buffer, len=int(self.length))

#     @always_inline("nodebug")
#     def set(inout self, val: c_int):
#         if not self.buffer:
#             raise "length must be set to non-zero"

#         for i in range(self.length):
#             self.buffer.offset(i).init_pointee_copy(val)

#     @always_inline("nodebug")
#     fn update(inout self, token: c_int) -> c_int:
#         if self.length == 0:
#             return 1

#         for i in range(self.length - 1):
#             self.buffer[i] = self.buffer[i + 1]

#         self.buffer[int(self.length) - 1] = token

#         if self.n < self.length:
#             self.n += 1

#         return (self.n == self.length).cast[DType.int32]()


# struct FILE:
#     pass


# struct FileHandle:
#     var handle: UnsafePointer[FILE]

#     @always_inline("nodebug")
#     fn __init__(inout self, path: String, mode: String) raises:
#         # https://man7.org/linux/man-pages/man3/fopen.3.html
#         var handle = external_call["fopen", UnsafePointer[FILE]](
#             path.unsafe_cstr_ptr(), mode.unsafe_cstr_ptr()
#         )
#         if not handle:
#             raise Error("Error opening file")

#         self.handle = handle

#     @always_inline("nodebug")
#     fn __moveinit__(inout self, owned existing: Self):
#         self.handle = existing.handle

#     @always_inline("nodebug")
#     def fclose(inout self):
#         """Safe and idiomatic wrapper https://man7.org/linux/man-pages/man3/fclose.3.html.
#         """
#         debug_assert(
#             self.handle != UnsafePointer[FILE](), "File must be opened first"
#         )
#         var ret = external_call["fclose", c_int, UnsafePointer[FILE]](
#             self.handle
#         )
#         # Important to set handle to NULL ptr to prevent having dangling pointer
#         self.handle = UnsafePointer[FILE]()
#         if ret:
#             raise Error("Error in closing the file")

#         return

#     @always_inline("nodebug")
#     def fgetc(inout self) -> c_int:
#         """Safe and idiomatic wrapper https://man7.org/linux/man-pages/man3/fgetc.3.html.
#         """
#         debug_assert(
#             self.handle != UnsafePointer[FILE](), "File must be opened first"
#         )
#         var ret = external_call["fgetc", c_int, UnsafePointer[FILE]](
#             self.handle
#         )
#         if not ret:  # null on error
#             raise Error("Error in fgetc")

#         return ret

# @always_inline("nodebug")
# def fopen(path: String, mode: String = "r") -> FileHandle:
#     return FileHandle(path, mode)

# struct DataLoader:
#     var file: FileHandle
#     var seq_len: c_int
#     var tape: Tape

#     @always_inline("nodebug")
#     fn __init__(inout self, path: String, seq_len: c_int) raises:
#         self.file = fopen(path, mode="r")
#         self.seq_len = seq_len
#         self.tape = Tape(self.seq_len)

#     @always_inline("nodebug")
#     fn __del__(owned self):
#         try:
#             self.file.fclose()
#             _ = self.tape^

#         except:
#             return

#     @always_inline("nodebug")
#     def next(inout self) -> c_int:
#         while True:
#             c = self.file.fgetc()
#             if c == EOF:
#                 break

#             token = tokenizer_encode(c)
#             ready = self.tape.update(token)
#             if ready:
#                 return 1

#         return 0

# @always_inline("nodebug")
# fn error_usage():
#     print("Usage: ./ngram [options]", end="\n")
#     print("Options:", end="\n")
#     print(" -n <int> n-gram model arity (default 4)", end="\n")
#     print(" -s <float> smoothing factor (default 0.1)", end="\n")
#     sys.exit(1)


# def main():
#     args = sys.argv()
#     argc = len(args)
#     seq_len = c_int(4)
#     smoothing = c_float(0.1)
#     for i in range(1, argc, 2):
#         if i + 1 >= argc:
#             return error_usage()
#         if args[i][0] != "-":
#             return error_usage()
#         if len(args[i]) != 2:
#             return error_usage()
#         if args[i][1] == "n":
#             seq_len = atol(args[i + 1])
#         elif args[i][1] == "s":
#             smoothing = atof(args[i + 1]).cast[DType.float32]()
#         else:
#             return error_usage()

#     # train the model
#     model = NgramModel(NUM_TOKENS, seq_len, smoothing)
#     train_loader = DataLoader("/home/ubuntu/workspace/explore/ehsan-ngram/data/train.txt", seq_len)
#     while train_loader.next():
#         model.train(train_loader.tape.span())

#     # allocate probs buffer for inference
#     probs = InlineArray[c_float, NUM_TOKENS](NUM_TOKENS)
#     sample_tape = Tape(seq_len - 1)
#     sample_tape.set(EOT_TOKEN) # fill with EOT tokens to init
#     rng = RNG(1337)
#     for _ in range(200):
#         model.inference(sample_tape.span(), probs)
#         coinf = rng.random_f32()
#         token = sample_discrete(probs, NUM_TOKENS, coinf)
#         _ = sample_tape.update(token)
#         c = tokenizer_decode(token)
#         print(chr(int(c)), end="")

#     print("\n")

#     # evaluate the test split loss
#     test_loader = DataLoader("/home/ubuntu/workspace/explore/ehsan-ngram/data/test.txt", seq_len)
#     sum_loss = c_float(0)
#     count = 0
#     while test_loader.next():
#         # note that `inference` will only use the first seq_len - 1 tokens in buffer
#         test_tape = test_loader.tape.span()
#         model.inference(test_tape, probs)
#         # and the last token in the tape buffer is the label
#         target = test_tape[int(seq_len) - 1]
#         # negative log likelihood loss
#         sum_loss += -math.log(probs[int(target)])
#         count += 1

#     mean_loss = sum_loss / count
#     test_preplexity = math.exp(mean_loss)
#     print(
#         String.format(
#             "test_loss {}, test_preplexity {}", mean_loss, test_preplexity
#         )
#     )
#     return
