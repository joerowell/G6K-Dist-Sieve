// clang-format off

/***                                            \
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


#ifndef G6K_SIEVER_INL
#define G6K_SIEVER_INL

#ifndef G6K_SIEVER_H
#error Do not include siever.inl directly
#endif

#include "parallel_algorithms.hpp"
namespace pa = parallel_algorithms;

// a += c*b
template <typename Container, typename Container2>
inline void Siever::addmul_vec(Container &a, Container2 const &b, const typename Container::value_type c, int num)
{
    auto ita = a.begin();
    auto itb = b.cbegin();
    auto const ite = ita + num;

    for (; ita != ite; ++ita, ++itb)
    {
        *ita += c * (*itb);
    }
}

template <typename Container, typename Container2>
inline void Siever::addmul_vec(Container &a, Container2 const &b, const typename Container::value_type c)
{
    auto ita = a.begin();
    auto itb = b.cbegin();
    auto const ite = ita + n;

    for (; ita != ite; ++ita, ++itb)
    {
        *ita += c * (*itb);
    }
}

template <typename Container, typename Container2>
inline void Siever::addsub_vec(Container &a, Container2 const &b, const typename Container::value_type c)
{
    auto ita = a.begin();
    auto itb = b.cbegin();
    auto const ite = ita + n;
    assert( c == 1 || c == -1 );

    if (c == 1)
    {
        for (; ita != ite; ++ita, ++itb)
            *ita += *itb;
    } else {
        for (; ita != ite; ++ita, ++itb)
            *ita -= *itb;
    }
}


inline size_t Siever::histo_index(double l) const
{
    int const i = std::ceil((l - 1.) * size_of_histo - .001);
    if (i > static_cast<int>(size_of_histo-1)) return size_of_histo-1; // the static_cast is just to silence a warning.
    if (i < 0) return 0;
    return i;
}

template<class Functor>
void Siever::apply_to_all_entries(Functor const &functor)
{
    int th_n = std::min<int>(this->params.threads, 1 + db.size()/MIN_ENTRY_PER_THREAD);
    threadpool.run([this,functor](int th_i, int th_n)
        {
	    for (auto i : pa::subrange(db.size(), th_i, th_n))
		functor(this->db[i]);
        }, th_n);
}

template<class Functor>
void Siever::apply_to_all_compressed_entries(Functor const &functor)
{
    int th_n = std::min<int>(this->params.threads, 1 + cdb.size()/MIN_ENTRY_PER_THREAD);
    threadpool.run([this,functor](int th_i, int th_n)
        {
	    for (auto i : pa::subrange(cdb.size(), th_i, th_n))
		functor(this->cdb[i]);
        }, th_n);
}

template<unsigned int THRESHOLD>
inline bool Siever::is_reducible_maybe(const uint64_t *left, const uint64_t *right)
{

    // No idea if this makes a difference, mirroring previous code
    unsigned wa = unsigned(0) - THRESHOLD;
    for (size_t k = 0; k < XPC_WORD_LEN; ++k)
    {
        // NOTE return type of __builtin_popcountl is int not unsigned int
        wa += __builtin_popcountl(left[k] ^ right[k]);
    }
    return (wa > (XPC_BIT_LEN - 2 * THRESHOLD));
}


template<unsigned int THRESHOLD>
inline bool Siever::is_reducible_maybe(const CompressedVector &left, const CompressedVector &right)
{
    return is_reducible_maybe<THRESHOLD>(&left.front(), &right.front());
}

/*
    Same as the above is_reducible_maybe but for the inner most loop of triple sieve
    We look for pairs that are *far apart* (i.e. PopCnt is bigger than expected rather than smaller)
*/
template<unsigned int THRESHOLD>
inline bool Siever::is_far_away(const uint64_t *left, const uint64_t *right)
{
    unsigned w = unsigned(0);
    for (size_t k = 0; k < XPC_WORD_LEN; ++k)
    {
        // NOTE return type of __builtin_popcountl is int not unsigned int
        w += __builtin_popcountl(left[k] ^ right[k]);
    }
    return (w > THRESHOLD);
}


template<unsigned int THRESHOLD>
inline bool Siever::is_far_away(const CompressedVector &left, const CompressedVector &right)
{
    return is_far_away<THRESHOLD>(&left.front(), &right.front());
}

/**
  ENABLE_BITOPS_FOR_ENUM enables bit-operations on enum classes without having to static_cast
  This macro is defined in compat.hpp.
*/

ENABLE_BITOPS_FOR_ENUM(Siever::Recompute)

inline Siever::Recompute constexpr Siever::recompute_all_no_otf_lift() {
  return Recompute::recompute_all & (~Recompute::consider_otf_lift);
}

inline Siever::Recompute constexpr Siever::recompute_all_but_c_no_otf_lift() {
  return Recompute::recompute_all & (~Recompute::recompute_c) & (~Recompute::consider_otf_lift);
}

inline Siever::Recompute constexpr Siever::recompute_only_uid_no_otf_lift() {
  return Recompute::recompute_uid & Recompute::recompute_otf_helper;
}

inline Siever::Recompute constexpr Siever::recompute_after_redist() {
  return Siever::recompute_all_no_otf_lift();
}

inline Siever::Recompute constexpr Siever::recompute_recv() {
#if defined X_AND_C
  return Siever::recompute_all_but_c_no_otf_lift();
#elif defined XC_AND_Y
  // N.B This sends the length too because recompute_data_for_entry recomputes both yr and the length
  // if either is missing.
  return Siever::recompute_only_uid_no_otf_lift();
#else
  return Siever::recompute_all_no_otf_lift();
#endif
}

template<Siever::Recompute what_to_recompute, unsigned width>
inline void Siever::recompute_data_for_entry_vec(std::vector<Entry>& entries, const unsigned start, const unsigned end) {
  bool constexpr rec_yr = (what_to_recompute & Recompute::recompute_yr) != Recompute::none;
  bool constexpr rec_len = (what_to_recompute & Recompute::recompute_len) != Recompute::none;
  bool constexpr rec_c = (what_to_recompute & Recompute::recompute_c) != Recompute::none;
  bool constexpr rec_uid = (what_to_recompute & Recompute::recompute_uid) != Recompute::none;
  bool constexpr rec_otf_helper = (what_to_recompute & Recompute::recompute_otf_helper) != Recompute::none;

  if(end == start) return;
  
  using VecF = typename Simd::TypeDispatch<width>::FloatType;
  using VecD = typename Simd::TypeDispatch<width>::DoubleType;
  using VecQ = typename Simd::TypeDispatch<width>::QuadType;
  
  std::vector<int64_t> xs((rec_yr || rec_uid || rec_len || rec_otf_helper) ? width*n : 0);
  
  // N.B The conversion to double precision here is needed for maximum throughput in the loop
  // below for re-computing yr and len (because otherwise the calculation will not use the right
  // CPU instructions, at least on x86-64).
  std::vector<FT> ys((rec_yr || rec_uid || rec_len || rec_otf_helper) ? width*n : 0);
  std::vector<LFT> v((rec_c) ? (width * n) : 0);
  
  assert((end-start)%width == 0);
  
  for(unsigned pos = start; pos < end; pos += width) {
    if(rec_yr || rec_uid || rec_len || rec_otf_helper) {
      for(unsigned j = 0; j < n; j++) {
        for(unsigned m = 0; m < width; m++) {
          xs[width*j + m] = entries[pos+m].x[j];
          ys[width*j + m] = FT(entries[pos+m].x[j]);
        }
      }
    }

    if(rec_yr || rec_len) {
      VecD lens{};

      
      for(unsigned i = 0; i < n; i++) {        
        // Warning: understanding this code requires some low-level knowledge.
        // Essentially, most modern X86 CPUs have an instruction variant called fused-multiply add (FMA),
        // which allows you to compute ab+c in a single instruction. This is pretty neat, and it's exactly
        // what we need for many inner products.
        //
        // However, FMAs have an annoying cost: they have around four cycles of latency. On the other hand,
        // the instruction itself typically takes between 0.5-1 cycles to execute, meaning that we spend a lot
        // of time just waiting for results. This is painful to watch.
        //
        // Thankfully, we can do better if we just issue many such instructions at once. The compiler will not
        // do this for us, but we simulate this manually. We thus unroll manually by a factor of 4
        // and handle the other ones separately. Unrolling by more does not seem to help at all.

        const auto size = n-i;
        const auto size_eight = 8*(size/8);
        const auto size_four  = 4*((size-size_eight)/4);
        VecD tmp0{}, tmp1{}, tmp2{}, tmp3{}, tmp4{}, tmp5{}, tmp6{}, tmp7{};

        unsigned j = i;

        // the i + is to make sure that we don't skip this loop altogether.
        if(size_eight > 0) {
          for(;j < i + size_eight; j += 8) {  
            tmp0 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+0)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+0]);
            tmp1 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+1)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+1]);
            tmp2 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+2)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+2]);
            tmp3 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+3)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+3]);
            tmp4 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+4)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+4]);
            tmp5 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+5)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+5]);
            tmp6 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+6)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+6]);
            tmp7 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+7)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+7]);
          }
          tmp0 += tmp4;
          tmp1 += tmp5;
          tmp2 += tmp6;
          tmp3 += tmp7;
        }
        

        if(size_four > 0) {
          for(; j < i+size_eight + size_four; j+=4) {
            tmp0 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+0)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+0]);
            tmp1 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+1)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+1]);
            tmp2 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+2)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+2]);
            tmp3 += Simd::TypeDispatch<width>::to_dp(&ys[width*(j+3)]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j+3]);
          }
        }
        
        
        for(; j < n; j++) {
          tmp0 += Simd::TypeDispatch<width>::to_dp(&ys[width*j]) * Simd::TypeDispatch<width>::fill_dp(muT[i][j]);
        }
        
        const auto yr = (tmp0 + tmp1 + tmp2 + tmp3) * Simd::TypeDispatch<width>::fill_dp(sqrt_rr[i]);

        if(rec_yr) {
          for(unsigned j = 0; j < width; j++) {
            entries[pos + j].yr[i] = yr[j];
          }
        }

        if(rec_len) {
          lens += yr*yr;          
        }      
      }

      if(rec_len) {
        for(unsigned j = 0; j < width; j++) {
          entries[pos + j].len = lens[j];
        }
      }
    }

    if(rec_uid) {
      VecQ uids{};
      const auto& coeffs = uid_hash_table.get_uid_coeffs();
      for(unsigned j = 0; j < n; j++) {
        uids += Simd::TypeDispatch<width>::fill_qw(coeffs[j]) * Simd::TypeDispatch<width>::to_quad_type(&xs[width*j]);
      }
      
      for(unsigned j = 0; j < width; j++) {
        entries[pos+j].uid = uids[j];
      }
    }

    if(rec_c) {
      for(unsigned j = 0; j < n; j++) {
        for(unsigned m = 0; m < width; m++) {
          v[width*j + m] = entries[pos+m].yr[j];
        }
      }
      std::array<CompressedVector, width> c;
      sim_hashes.vec_compress<width>(v.data(), c);
      for(unsigned j = 0; j < width; j++) {
        entries[pos+j].c = c[j];
      }
    }

    if(rec_otf_helper) {
      // Similarly to the coefficient recomputation, we unroll the inner loop manually here.
      const auto size_four = 4*(n/4);
      
      for(unsigned k = 0; k < OTF_LIFT_HELPER_DIM; ++k) {
        int const i = (l - k + 1);
        if(i < static_cast<signed int>(ll)) break;
        VecD tmp0{}, tmp1{}, tmp2{}, tmp3{}, cau{};
        unsigned j{};
        
        for(j = 0; j < size_four; j+= 4) {
          tmp0 += Simd::TypeDispatch<width>::to_dp(&xs[width*(j+0)]) * 
            Simd::TypeDispatch<width>::fill_dp(full_muT[i][l+j+0]);
          tmp1 += Simd::TypeDispatch<width>::to_dp(&xs[width*(j+1)]) * 
            Simd::TypeDispatch<width>::fill_dp(full_muT[i][l+j+1]);
          tmp2 += Simd::TypeDispatch<width>::to_dp(&xs[width*(j+2)]) * 
            Simd::TypeDispatch<width>::fill_dp(full_muT[i][l+j+2]);
          tmp3 += Simd::TypeDispatch<width>::to_dp(&xs[width*(j+3)]) * 
            Simd::TypeDispatch<width>::fill_dp(full_muT[i][l+j+3]);          
        }

        for(; j < n; j++) {
           cau += Simd::TypeDispatch<width>::to_dp(&xs[width*j]) *
            Simd::TypeDispatch<width>::fill_dp(full_muT[i][l+j]);
        }

        cau += tmp0 + tmp1 + tmp2 + tmp3;
                
        for(unsigned j = 0; j < width; j++) {
          entries[pos+j].otf_helper[k] = cau[j];
        }
      }
    }    
  }
}

template<Siever::Recompute what_to_recompute>
inline void Siever::recompute_data_for_batch(std::vector<Entry> &entries, const unsigned start, const unsigned end) {
  // This function applies some (hopefully useful) vectorisation tricks to make
  // recomputing data for a collection of entries faster.
  ATOMIC_CPUCOUNT(214);  
  bool constexpr consider_lift = (what_to_recompute & Recompute::consider_otf_lift) != Recompute::none;

  assert(end <= entries.size());
  const auto size = (end - start);
  if(size == 0) return;

  // Note: recomputing yr, otf_lift_helper and len essentially boils down to just doing matrix multiplications.
  // To make this fast, we'll use BLAS.
  

  
  // Work out how many entries can be vectorised in sets of 8.
  // At the moment, at least, sets of 8 seem to be slower than just doing more sets of 4:
  // this is probably because the vectorisation then needs AVX512 / mimics AVX512, which
  // (on this CPU at least) will lower the clock frequency. 
  constexpr bool use_eight = false;
  const auto size_eight = (use_eight) ? 8 * (size/8) : 0;
  
  // Now work out how many can be vectorised in sets of 4 from the remaining amount.
  
  const auto size_four = 4 * ((size - size_eight)/4);  
  // We don't yet do parallelism more finely than sets of 4.
  unsigned processed{start};
  
  if(size_eight > 0) {
    const auto lower = processed;
    const auto upper = processed + size_eight;
    recompute_data_for_entry_vec<what_to_recompute, 8>(entries, lower, upper);
    // The otf lift stuff isn't so easily parallelisable.
    if(consider_lift && params.otf_lift) {
      for(unsigned i = lower; i < upper; i++) {
          Entry& e = entries[i];
          if(e.len < params.lift_radius) {            
            lift_and_compare(e);
          }
      }      
      processed += size_eight;   
    }
  }
  
  if(size_four > 0) {
    const auto lower = processed;
    const auto upper = processed + size_four;
    recompute_data_for_entry_vec<what_to_recompute, 4>(entries, lower, upper);

    if(consider_lift && params.otf_lift) {
      for(unsigned i = lower; i < upper; i++) {        
          Entry& e = entries[i];
          if(e.len < params.lift_radius) {
            lift_and_compare(e);
          }
      }
    }
    
    processed += size_four;
  }

  // And for the rest we just do the regular thing.
  for(; processed < end; processed++) {
    recompute_data_for_entry<what_to_recompute>(entries[processed]);
  }
}

// if you change one function template, you probably have to change all 2 (vanilla, babai)
template<Siever::Recompute what_to_recompute>
inline void Siever::recompute_data_for_entry(Entry &e)
{
    ATOMIC_CPUCOUNT(214);
    bool constexpr rec_yr = (what_to_recompute & Recompute::recompute_yr) != Recompute::none;
    bool constexpr rec_len = (what_to_recompute & Recompute::recompute_len) != Recompute::none;
    bool constexpr rec_c = (what_to_recompute & Recompute::recompute_c) != Recompute::none;
    bool constexpr rec_uid = (what_to_recompute & Recompute::recompute_uid) != Recompute::none;
    bool constexpr consider_lift = (what_to_recompute & Recompute::consider_otf_lift) != Recompute::none;
    bool constexpr rec_otf_helper = (what_to_recompute & Recompute::recompute_otf_helper) != Recompute::none;


    CPP17CONSTEXPRIF(rec_len) e.len = 0.;
    for (unsigned int i = 0; i < n; ++i)
    {
        if (rec_yr || rec_len)
        {
            FT const yri = std::inner_product(e.x.cbegin()+i, e.x.cbegin()+n, muT[i].cbegin()+i,  static_cast<FT>(0.)) * sqrt_rr[i];
            if (rec_yr) e.yr[i] = yri; // Note : conversion to lower precision
            if (rec_len) e.len+=yri * yri; // slightly inefficient if we only compute the lenght and not yr, but that does not happen anyway.
        }
    }

    // No benefit of merging those loops, I think.

    CPP17CONSTEXPRIF (rec_uid)
    {
        e.uid  = uid_hash_table.compute_uid(e.x);
    }

    CPP17CONSTEXPRIF (rec_c)
    {
        e.c = sim_hashes.compress(e.yr);
    }

    CPP17CONSTEXPRIF (rec_otf_helper)
    {
        for (int k = 0; k < OTF_LIFT_HELPER_DIM; ++k)
        {
            int const i = l - (k + 1);
            if (i < static_cast<signed int>(ll)) break;
            e.otf_helper[k] = std::inner_product(e.x.cbegin(), e.x.cbegin()+n, full_muT[i].cbegin()+l,  static_cast<FT>(0.));
        }
    }



    if (consider_lift && params.otf_lift && e.len < params.lift_radius)
    {
        lift_and_compare(e);
    }


    return;
}


template<Siever::Recompute what_to_recompute>
inline void Siever::recompute_data_for_entry_babai(Entry &e, int babai_index)
{
    ATOMIC_CPUCOUNT(215);
    bool constexpr rec_yr = (what_to_recompute & Recompute::recompute_yr) != Recompute::none;
    bool constexpr rec_len = (what_to_recompute & Recompute::recompute_len) != Recompute::none;
    bool constexpr rec_c = (what_to_recompute & Recompute::recompute_c) != Recompute::none;
    bool constexpr rec_uid = (what_to_recompute & Recompute::recompute_uid) != Recompute::none;
    bool constexpr consider_lift = (what_to_recompute & Recompute::consider_otf_lift) != Recompute::none;
    bool constexpr rec_otf_helper = (what_to_recompute & Recompute::recompute_otf_helper) != Recompute::none;

    CPP17CONSTEXPRIF(rec_len) e.len = 0.;

  // recompute y, yr, len for the other indices (if requested)
    for (int i = n-1; i >= babai_index; --i)
    {
        CPP17CONSTEXPRIF(rec_yr || rec_len)
        {
            FT const yri = std::inner_product(e.x.cbegin()+i, e.x.cbegin()+n, muT[i].cbegin()+i,  static_cast<FT>(0.)) * sqrt_rr[i];
            CPP17CONSTEXPRIF (rec_yr) e.yr[i] = yri;    // Note : conversion to lower precision
            CPP17CONSTEXPRIF (rec_len) e.len+=yri * yri; // slightly inefficient if we only compute the lenght and not yr, but that does not happen anyway.
        }
    }

    for (int i = babai_index -1 ; i >= 0; --i)
    {
        FT yi = std::inner_product(e.x.cbegin()+i+1, e.x.cbegin()+n, muT[i].cbegin()+i+1, static_cast<FT>(0.));
        int c = -std::floor(yi+0.5);
        e.x[i] = c;
        yi += c;
        yi *= sqrt_rr[i];
        e.yr[i] = yi; // ( original_value(yi) + c ) * sqrt_rr[i]. Note that assignment loses precision.
        CPP17CONSTEXPRIF (rec_len) e.len += yi * yi; // adds e.yr[i]^2 (but with higher precision)
    }

    // No benefit of merging those loops, I think.

    CPP17CONSTEXPRIF (rec_uid)
    {
        e.uid = uid_hash_table.compute_uid(e.x);
    }

    CPP17CONSTEXPRIF (rec_c)
    {
        e.c = sim_hashes.compress(e.yr);
    }

    CPP17CONSTEXPRIF (rec_otf_helper)
    {
        for (int k = 0; k < OTF_LIFT_HELPER_DIM; ++k)
        {
            int const i = l - (k + 1);
            
            if (i < static_cast<signed int>(ll)) break;
            e.otf_helper[k] = std::inner_product(e.x.cbegin(), e.x.cbegin()+n, full_muT[i].cbegin()+l,  static_cast<FT>(0.));
        }
    }

    if (consider_lift && params.otf_lift && e.len < params.lift_radius)
    {
        lift_and_compare(e);
    }

    return;
}

#endif

// clang-format on
