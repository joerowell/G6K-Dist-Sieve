/***\
 *
 *   Copyright (C) 2018-2021 Team G6K
 *
 *   This file is part of G6K. G6K is free software:
 *   you can redistribute it and/or modify it under the terms of the
 *   GNU General Public License as published by the Free Software Foundation,
 *   either version 2 of the License, or (at your option) any later version.
 *
 *   G6K is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with G6K. If not, see <http://www.gnu.org/licenses/>.
 *
 ****/

#ifndef G6K_SIMHASH_INL
#define G6K_SIMHASH_INL

#ifndef G6K_SIEVER_H
#error Do not include siever.inl directly
#endif

// choose the vectors sparse vectors r_i for the compressed representation
inline void SimHashes::reset_compress_pos(Siever const &siever) {
  n = siever.n;
  if (n < 30) {
    for (size_t i = 0; i < XPC_BIT_LEN; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        compress_pos[i][j] = 0;
      }
    }
    return;
  }

  size_t x, y;
  std::string const filename = siever.params.simhash_codes_basedir + "/sc_" +
                               std::to_string(n) + "_" +
                               std::to_string(XPC_BIT_LEN) + ".def";
  std::ifstream in(filename);
  std::vector<int> permut;

  if (!in) {
    std::string s = "Cannot open file ";
    s += filename;
    throw std::runtime_error(s);
  }

  // create random permutation of 0..n-1:
  permut.resize(n);
  std::iota(permut.begin(), permut.end(), 0);
  std::shuffle(permut.begin(), permut.end(), sim_hash_rng);

  for (y = 0; y < XPC_BIT_LEN; y++) {
    for (x = 0; x < 6; x++) {
      int v;
      in >> v;
      compress_pos[y][x] = permut[v];
    }
  }
  in.close();
}

template <long unsigned nr>
inline void SimHashes::vec_compress(const LFT *const v,
                                    std::array<CompressedVector, nr> &c) const {

  static_assert(nr == 4 || nr == 8);
  ATOMIC_CPUCOUNT(260);
  for (auto &x : c) {
    std::fill(x.begin(), x.end(), 0);
  }

  if (n < 30)
    return;

  // This function is actually quite straightforward. The idea is that we can
  // realise better speedups (in a distributed setting, mainly) when computing
  // multiple simhashes at once. Indeed, if we have an already transposed set of
  // incoming vectors, then most of the operations can be parallelised really
  // naturally.
  //
  // The idea here is as follows. v is a pointer to a vector (of size nr*n) that
  // contains nr vectors in a transposed order. Specifically, if the vectors
  // are b_0, b_1, ..., b_(nr-1), then v[nr*i+j] = b_j[i] for all j < nr. This
  // means that when we iterate through the compress_pos, we can load the values
  // for any x < n by simply loading v[nr*x]. This then means that the simhash
  // computation becomes quite cheap: conceptually, you can think of a[i] as
  // containing the result of the (current) simhash value for b_i at the end of
  // each iteration.
  //
  // This shouldn't
  // really have been an issue, but compilers seem to struggle with vectorising
  // the existing version of compress.
  std::array<uint64_t, nr> c_store;

  // Load the right vector types.
  using VecI = typename Simd::TypeDispatch<nr>::IntegerType;
  using VecF = typename Simd::TypeDispatch<nr>::FloatType;

  VecI c_tmp;
  VecF cau[6];

  const auto iter32 = [&](const unsigned j, unsigned lo, const unsigned hi) {
    for (; lo < hi; lo++) {
      const unsigned k = 64 * j + lo;
      for (unsigned m = 0; m < 6; m++) {
        const auto load_pos = nr * compress_pos[k][m];
        memcpy(&cau[m], &v[load_pos], sizeof(cau[m]));
      }

      const auto first = cau[0] + cau[1] + cau[2];
      const auto second = cau[3] + cau[4] + cau[5];
      const auto a = first - second;

      // N.B This is a little bit tricky. If you look at the regular compress
      // function, you'll see that at the end of each loop the following is
      // executed: c_tmp <<= 1; c_tmp |= (uint64_t)(a > 0).
      //
      // GCC's vector intrinsics do not allow shifting by bits (i.e a << 1
      // shifts each element in a left by one byte). However, because the
      // comparison > 0.0f returns 0xFFFFFFFF in case of truth and 0 in case of
      // failure, we can mask simply extract the bits "in order". This allows us
      // to avoid the shift and realise a small speedup.
      // Note: despite the fact that we end up having to unpack values twice,
      // this code is more than twice as fast as the 64 bit version. No idea
      // why.
      const auto masked = VecI(a > 0.0f) & (1 << (31 - (lo % 32)));
      c_tmp = c_tmp | masked;
    }
  };

  for (unsigned j = 0; j < XPC_WORD_LEN; j++) {
    c_tmp = c_tmp ^ c_tmp;
    iter32(j, 0, 32);

    // Unpack c_tmp into c_store.
    for (unsigned m = 0; m < nr; m++) {
      c_store[m] = c_tmp[m];
      c_store[m] <<= 32;
    }

    // Now do it again.
    c_tmp = c_tmp ^ c_tmp;
    iter32(j, 32, 64);

    // And now, finally, unpack and store.
    for (unsigned m = 0; m < nr; m++) {
      c[m][j] = c_store[m] | c_tmp[m];
    }
  }
}

inline void SimHashes::compress_four(const LFT *const v,
                                     std::array<CompressedVector, 4> &c) const {
  vec_compress(v, c);
}

// Compute the compressed representation of an entry
inline CompressedVector
SimHashes::compress(std::array<LFT, MAX_SIEVING_DIM> const &v) const {
  ATOMIC_CPUCOUNT(260);
  CompressedVector c = {0};
  if (n < 30)
    return c;
  for (size_t j = 0; j < XPC_WORD_LEN; ++j) {
    uint64_t c_tmp = 0;
    LFT a = 0;
    for (size_t i = 0; i < 64; i++) {
      size_t k = 64 * j + i;
      a = v[compress_pos[k][0]];
      a += v[compress_pos[k][1]];
      a += v[compress_pos[k][2]];
      a -= v[compress_pos[k][3]];
      a -= v[compress_pos[k][4]];
      a -= v[compress_pos[k][5]];

      c_tmp = c_tmp << 1;
      c_tmp |= (uint64_t)(a > 0);
    }
    c[j] = c_tmp;
  }
  return c;
}

#endif
