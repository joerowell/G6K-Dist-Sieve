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

#include "QEntry.hpp"
#include "fht_lsh.h"
#include "siever.h"
#include <algorithm>
#include <assert.h>
#include <chrono> // needed for wall time>
#include <cstring>
#include <ctime>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <type_traits>
#include <vector>

struct TimePoint {
  using T = decltype(std::chrono::steady_clock::now());
  std::string name;
  T start;

  explicit TimePoint(const std::string &name_)
      : name{name_}, start(std::chrono::steady_clock::now()) {}
  ~TimePoint() {
    std::cerr << "Time for " << name << " "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
  }
};

#define TIME_BDGL(x)
/*
#define TIME_BDGL(x)                                                           \
  TimePoint tp(std::string("[Rank " + std::to_string(mpi.get_rank()) + "] " +  \
                         std::string(x)))
*/
struct atomic_size_t_wrapper {
  atomic_size_t_wrapper() : val(0) {}
  atomic_size_t_wrapper(const size_t &v) : val(v) {}
  atomic_size_t_wrapper(const atomic_size_t_wrapper &v) : val(size_t(v.val)) {}
  std::atomic_size_t val;
  CACHELINE_PAD(pad);
};

inline bool compare_QEntry(QEntry const &lhs, QEntry const &rhs) {
  return lhs.len < rhs.len;
}

std::pair<LFT, int8_t> Siever::reduce_to_QEntry(CompressedEntry *ce1,
                                                CompressedEntry *ce2) {
  LFT inner =
      std::inner_product(db[ce1->i].yr.begin(), db[ce1->i].yr.begin() + n,
                         db[ce2->i].yr.begin(), static_cast<LFT>(0.));
  LFT new_l = ce1->len + ce2->len - 2 * std::abs(inner);
  int8_t sign = (inner < 0) ? 1 : -1;
  return {new_l, sign};
}

inline int Siever::bdgl_reduce_with_delayed_replace(
    const size_t i1, const size_t i2, LFT const lenbound,
    std::vector<Entry> &transaction_db, int64_t &write_index, LFT new_l,
    int8_t sign) {
  UidType new_uid = db[i1].uid;
  if (sign == 1) {
    new_uid += db[i2].uid;
  } else {
    new_uid -= db[i2].uid;
  }

  if (new_l < lenbound) {
    if (uid_hash_table.insert_uid(new_uid)) {
      std::array<ZT, MAX_SIEVING_DIM> new_x = db[i1].x;
      addsub_vec(new_x, db[i2].x, static_cast<ZT>(sign));
      int64_t index = write_index--; // atomic and signed!
      if (index >= 0) {
        Entry &new_entry = transaction_db[index];
        new_entry.x = new_x;
        recompute_data_for_entry<
            Recompute::recompute_all_and_consider_otf_lift>(new_entry);
        return 1;
      }
      return -2; // transaction_db full
    } else {
      // duplicate
      return 0;
    }
  }

  if (params.otf_lift && (new_l < params.lift_radius)) {
    bdgl_lift(i1, i2, new_l, sign);
  }

  return -1;
}

inline void Siever::bdgl_lift(const Entry &e1, const Entry &e2, LFT new_l,
                              int8_t sign) {
  ZT x[r];
  LFT otf_helper[OTF_LIFT_HELPER_DIM];
  std::fill(x, x + l, 0);
  std::copy(e1.x.cbegin(), e1.x.cbegin() + n, x + l);
  std::copy(e1.otf_helper.cbegin(),
            e1.otf_helper.cbegin() + OTF_LIFT_HELPER_DIM, otf_helper);

  if (sign == 1) {
    for (unsigned int i = 0; i < n; ++i) {
      x[l + i] += e2.x[i];
    }
    for (unsigned int i = 0; i < OTF_LIFT_HELPER_DIM; ++i) {
      otf_helper[i] += e2.otf_helper[i];
    }
  } else {
    for (unsigned int i = 0; i < n; ++i) {
      x[l + i] -= e2.x[i];
    }

    for (unsigned int i = 0; i < OTF_LIFT_HELPER_DIM; ++i) {
      otf_helper[i] -= e2.otf_helper[i];
    }
  }
  lift_and_compare(x, new_l * gh, otf_helper);
}

// assumed that sign is known
inline void Siever::bdgl_lift(const size_t i1, const size_t i2, LFT new_l,
                              int8_t sign) {
  const Entry &e1 = db[i1];
  const Entry &e2 = db[i2];
  bdgl_lift(e1, e2, new_l, sign);
}

// Replace the db and cdb entry pointed at by cdb[cdb_index] by e, unless
// length is actually worse
template <bool is_dist>
bool Siever::bdgl_replace_in_db(size_t cdb_index, Entry &e) {
  CompressedEntry &ce = cdb[cdb_index]; // abbreviation

  if (REDUCE_LEN_MARGIN_HALF * e.len >= ce.len) {
    uid_hash_table.erase_uid(e.uid);
    if (is_dist) {
      mpi.mark_as_unused(e.uid);
    }
    return false;
  }

  uid_hash_table.erase_uid(db[ce.i].uid);
  if (is_dist) {
    // We need to record the fact that we've overwritten db[ce.i].uid, so
    // that we don't end up with too many entries in the global uid table.
    mpi.mark_as_replaced(db[ce.i].uid);
    if (!mpi.owns_uid(e.uid)) {
      uid_hash_table.erase_uid(e.uid);
    }
  }

  ce.len = e.len;
  ce.c = e.c;
  db[ce.i] = e;
  return true;
}

void Siever::bdgl_bucketing_task(
    const size_t t_id, std::vector<uint32_t> &buckets,
    std::vector<atomic_size_t_wrapper> &buckets_index, ProductLSH &lsh) {
  CompressedEntry *const fast_cdb = cdb.data();
  const size_t S = cdb.size();
  const unsigned int nr_buckets = buckets_index.size();
  const size_t bsize = buckets.size() / nr_buckets;
  const size_t threads = params.threads;

  uint32_t i_start = t_id;
  const auto multi_hash = lsh.multi_hash;
  int32_t res[multi_hash];
  size_t bucket_index;
  for (uint32_t i = i_start; i < S; i += threads) {
    auto db_index = fast_cdb[i].i;
    lsh.hash(db[db_index].yr.data(), res);
    for (size_t j = 0; j < multi_hash; j++) {
      uint32_t b = res[j];
      assert(b < nr_buckets);
      bucket_index = buckets_index[b].val++; // atomic
      if (bucket_index < bsize) {
        buckets[bsize * b + bucket_index] = i;
      }
    }
  }
}

// assumes buckets and buckets_index are resized and resetted correctly.
void Siever::bdgl_bucketing(const size_t blocks, const size_t multi_hash,
                            const size_t nr_buckets_aim,
                            std::vector<uint32_t> &buckets,
                            std::vector<atomic_size_t_wrapper> &buckets_index) {
  // init hash
  const int64_t lsh_seed = rng();
  ProductLSH lsh(n, blocks, nr_buckets_aim, multi_hash, lsh_seed);
  const size_t nr_buckets = lsh.codesize;
  const size_t S = cdb.size();
  size_t bsize = 2 * (S * multi_hash / double(nr_buckets));
  buckets.resize(nr_buckets * bsize);
  buckets_index.resize(nr_buckets);
  for (size_t i = 0; i < nr_buckets; i++)
    buckets_index[i].val = 0;

  for (size_t t_id = 0; t_id < params.threads; ++t_id) {
    threadpool.push([this, t_id, multi_hash, &buckets, &buckets_index, &lsh]() {
      bdgl_bucketing_task(t_id, buckets, buckets_index, lsh);
    });
  }
  threadpool.wait_work();

  for (size_t i = 0; i < nr_buckets; ++i) {
    // bucket overflow
    if (buckets_index[i].val > bsize) {
      buckets_index[i].val = bsize;
    }
  }
}

void Siever::bdgl_process_buckets_task(
    const size_t t_id, const std::vector<uint32_t> &buckets,
    const std::vector<atomic_size_t_wrapper> &buckets_index,
    std::vector<QEntry> &t_queue) {

  const size_t nr_buckets = buckets_index.size();
  const size_t bsize = buckets.size() / buckets_index.size();

  const uint32_t *const fast_buckets = buckets.data();
  CompressedEntry *const fast_cdb = cdb.data();

  const size_t S = cdb.size();

  // todo: start insert earlier
  int64_t kk = S - 1 - t_id;

  LFT lenbound =
      fast_cdb[std::min(S - 1, size_t(params.bdgl_improvement_db_ratio * S))]
          .len;

  const size_t b_start = t_id;
  size_t B = 0;

  for (size_t b = b_start; b < nr_buckets; b += params.threads) {
    const size_t i_start = bsize * b;
    const size_t i_end = bsize * b + buckets_index[b].val;

    B += ((i_end - i_start) * (i_end - i_start - 1)) / 2;
    for (size_t i = i_start; i < i_end; ++i) {
      if (kk < .1 * S)
        break;

      uint32_t bi = fast_buckets[i];
      CompressedEntry *pce1 = &fast_cdb[bi];
      CompressedVector cv = pce1->c;
      for (size_t j = i_start; j < i; ++j) {
        uint32_t bj = fast_buckets[j];

        if (is_reducible_maybe<XPC_THRESHOLD>(cv, fast_cdb[bj].c)) {
          std::pair<LFT, int> len_and_sign =
              reduce_to_QEntry(pce1, &fast_cdb[bj]);
          if (len_and_sign.first < lenbound) {
            if (kk < .1 * S)
              break;
            kk -= params.threads;

            statistics.inc_stats_2redsuccess_outer();

            t_queue.push_back({pce1->i, fast_cdb[bj].i, len_and_sign.first,
                               (int8_t)len_and_sign.second});
          } else if (params.otf_lift and
                     len_and_sign.first < params.lift_radius) {
            bdgl_lift(pce1->i, fast_cdb[bj].i, len_and_sign.first,
                      len_and_sign.second);
          }
        }
      }
    }
  }
  statistics.inc_stats_xorpopcnt_inner(B);
  std::sort(t_queue.begin(), t_queue.end(), &compare_QEntry);
}

// Returned queue is sorted
void Siever::bdgl_process_buckets(
    const std::vector<uint32_t> &buckets,
    const std::vector<atomic_size_t_wrapper> &buckets_index,
    std::vector<std::vector<QEntry>> &t_queues) {

  for (size_t t_id = 0; t_id < params.threads; ++t_id) {
    threadpool.push([this, t_id, &buckets, &buckets_index, &t_queues]() {
      bdgl_process_buckets_task(t_id, buckets, buckets_index, t_queues[t_id]);
    });
  }

  threadpool.wait_work();
}

void Siever::bdgl_queue_dup_remove_task(std::vector<QEntry> &queue) {
  const size_t Q = queue.size();
  for (size_t index = 0; index < Q; index++) {
    size_t i1 = queue[index].i;
    size_t i2 = queue[index].j;
    UidType new_uid = db[i1].uid;
    if (queue[index].sign == 1) {
      new_uid += db[i2].uid;
    } else {
      new_uid -= db[i2].uid;
    }

    // if already present, use sign as duplicate marker
    if (uid_hash_table.check_uid_unsafe(new_uid))
      queue[index].sign = 0;
  }
}

void Siever::bdgl_queue_create_task(const size_t t_id,
                                    const std::vector<QEntry> &queue,
                                    std::vector<Entry> &transaction_db,
                                    int64_t &write_index) {
  const size_t S = cdb.size();
  const size_t Q = queue.size();

  const size_t insert_after = S - 1 - t_id - params.threads * write_index;
  for (unsigned int index = 0; index < Q; index++) {
    // use sign as skip marker
    if (queue[index].sign == 0) {
      continue;
    }

    bdgl_reduce_with_delayed_replace(
        queue[index].i, queue[index].j,
        cdb[std::min(S - 1, static_cast<unsigned long>(
                                insert_after + params.threads * write_index))]
                .len /
            REDUCE_LEN_MARGIN,
        transaction_db, write_index, queue[index].len, queue[index].sign);
    if (write_index < 0) {
      /*
      std::cerr << "Spilling full transaction db" << t_id << " " << Q - index
                << std::endl;
      */
      break;
    }
  }
}

template <bool is_dist>
size_t Siever::bdgl_queue_insert_task(const size_t t_id,
                                      std::vector<Entry> &transaction_db,
                                      int64_t write_index) {
  const size_t S = cdb.size();

  const auto to_sub = params.threads - (is_dist && params.threads != 1);
  const size_t insert_after =
      std::max(int(0), int(int(S) - 1 - t_id -
                           to_sub * (transaction_db.size() - write_index)));

  size_t kk = S - 1 - t_id;

  int i;
  for (i = transaction_db.size() - 1; i > write_index and kk >= insert_after;
       --i) {
    if (bdgl_replace_in_db<is_dist>(kk, transaction_db[i])) {
      kk -= to_sub;
    }
  }

  // Now purge the ones that, for whatever reason, we didn't insert.
  for (; i > write_index; --i) {
    uid_hash_table.erase_uid(transaction_db[i].uid);
    mpi.mark_as_unused(transaction_db[i].uid);
  }

  return kk + to_sub;
}

void Siever::bdgl_queue(std::vector<std::vector<QEntry>> &t_queues,
                        std::vector<std::vector<Entry>> &transaction_db) {

  for (size_t t_id = 0; t_id < params.threads; ++t_id) {
    threadpool.push([this, t_id, &t_queues]() {
      bdgl_queue_dup_remove_task(t_queues[t_id]);
    });
  }
  threadpool.wait_work();

  const size_t S = cdb.size();
  size_t Q = 0;
  for (unsigned int i = 0; i < params.threads; i++)
    Q += t_queues[i].size();
  size_t insert_after = std::max(0, int(int(S) - Q));

  for (unsigned int i = 0; i < params.threads; i++)
    transaction_db[i].resize(std::min(S - insert_after, Q) / params.threads +
                             1);

  std::vector<int> write_indices(params.threads, transaction_db[0].size() - 1);
  // Prepare transaction DB from queue

  for (size_t t_id = 0; t_id < params.threads; ++t_id) {
    threadpool.push([this, t_id, &t_queues, &transaction_db, &write_indices]() {
      int64_t write_index = write_indices[t_id];
      bdgl_queue_create_task(t_id, t_queues[t_id], transaction_db[t_id],
                             write_index);
      write_indices[t_id] = write_index;
      t_queues[t_id].clear();
    });
  }

  threadpool.wait_work();

  // Insert transaction DB
  std::vector<size_t> kk(params.threads);

  for (size_t t_id = 0; t_id < params.threads; ++t_id) {
    threadpool.push([this, &kk, t_id, &transaction_db, &write_indices]() {
      kk[t_id] = bdgl_queue_insert_task(t_id, transaction_db[t_id],
                                        write_indices[t_id]);
    });
  }
  threadpool.wait_work();

  size_t min_kk = kk[0];
  size_t inserted = 0;
  for (unsigned int i = 0; i < params.threads; i++) {
    min_kk = std::min(min_kk, kk[i]);
    inserted += (S - 1 - i - kk[i] - params.threads) / params.threads;
  }
  status_data.plain_data.sorted_until = min_kk;
}

bool Siever::bdgl_sieve(size_t nr_buckets_aim, const size_t blocks,
                        const size_t multi_hash) {
  switch_mode_to(SieveStatus::plain);
  if (mpi.is_root() && mpi.should_sieve(n, params.dist_threshold)) {
    mpi.start_bdgl(nr_buckets_aim, blocks, multi_hash);
  }

  mpi.in_sieving();

  const bool should_dist_sieve = mpi.should_sieve(n, params.dist_threshold);

  if (mpi.is_distributed_sieving_enabled() && !mpi.is_active() &&
      should_dist_sieve) {
    split_database();
  }

  if (mpi.is_active()) {
    const auto worked = bdgl_dist_sieve(nr_buckets_aim, blocks, multi_hash);
    if (mpi.is_root()) {
      mpi.write_stats(n, db.size());
    }
    return worked;
  }

  auto const S = cdb.size();
  parallel_sort_cdb();
  statistics.inc_stats_sorting_sieve();
  recompute_histo();

  size_t saturation_index = 0.5 * params.saturation_ratio *
                            std::pow(params.saturation_radius, n / 2.0);

  if (saturation_index > 0.5 * S) {
    std::cerr << "Saturation index larger than half of db size" << std::endl;
    saturation_index = std::min(saturation_index, S - 1);
  }

  std::vector<std::vector<Entry>> transaction_db(params.threads,
                                                 std::vector<Entry>());
  std::vector<uint32_t> buckets;
  std::vector<atomic_size_t_wrapper> buckets_i;
  std::vector<std::vector<QEntry>> t_queues(params.threads);

  size_t it = 0;
  while (true) {
    bdgl_bucketing(blocks, multi_hash, nr_buckets_aim, buckets, buckets_i);
    bdgl_process_buckets(buckets, buckets_i, t_queues);
    bdgl_queue(t_queues, transaction_db);
    parallel_sort_cdb();

    if (cdb[saturation_index].len <= params.saturation_radius) {
      assert(std::is_sorted(cdb.cbegin(), cdb.cend(), compare_CE()));
      invalidate_histo();
      recompute_histo();
      mpi.out_of_sieving();
      return true;
    }

    if (it > 10000) {
      std::cerr << "Not saturated after 10000 iterations" << std::endl;
      return false;
    }

    it++;
  }
}

void Siever::bdgl_distributed_bucketing(
    const size_t S, const size_t blocks, const size_t multi_hash,
    const size_t nr_buckets_aim, std::vector<uint32_t> &buckets,
    std::vector<atomic_size_t_wrapper> &buckets_index) {

  ProductLSH lsh = [&]() {
    if (mpi.is_root()) {
      const auto lsh_seed = rng();
      ProductLSH lsh(n, blocks, nr_buckets_aim, multi_hash, lsh_seed);
      mpi.bdgl_broadcast_lsh(lsh);
      return lsh;
    } else {
      return mpi.bdgl_build_lsh();
    }
  }();

  // We reserve these in-order as usual.
  const size_t nr_buckets = lsh.codesize;
  const size_t bsize = 2 * (S * multi_hash / double(nr_buckets));

  auto &owner = mpi.setup_bdgl_bucketing(n, nr_buckets, bsize, cdb.size());

  buckets.resize(nr_buckets * bsize);
  std::fill(buckets.begin(), buckets.end(), 0);
  buckets_index.resize(nr_buckets);

  for (size_t i = 0; i < nr_buckets; i++)
    buckets_index[i].val = 0;

  for (size_t t_id = 0; t_id < params.threads; ++t_id) {
    threadpool.push([this, t_id, multi_hash, &buckets, &buckets_index, &lsh]() {
      bdgl_bucketing_task(t_id, buckets, buckets_index, lsh);
    });
  }
  threadpool.wait_work();

  // Set up the various sizes etc.
  std::vector<uint64_t> sizes(nr_buckets);
  for (size_t i = 0; i < nr_buckets; ++i) {
    // bucket overflow
    if (buckets_index[i].val > bsize) {
      buckets_index[i].val = bsize;
    }
    sizes[i] = buckets_index[i].val;
  }

  mpi.bdgl_gather_sizes(sizes);
  GBL_max_len =
      cdb[std::min(cdb.size() - 1,
                   size_t(params.bdgl_improvement_db_ratio * cdb.size()))]
          .len;
}

void Siever::bdgl_sieve_bucket(const std::vector<CompressedVector> &cbucket,
                               const std::vector<Entry> &bucket,
                               const size_t bsize,
                               std::vector<Reduction> &t_queue, int64_t &kk) {

  size_t const bucket_size = cbucket.size();
  assert(cbucket.size() == bucket.size());
  size_t const S = cdb.size();

  CompressedVector const *const fast_bucket = cbucket.data();
  const auto *const fast_db = db.data();
  const auto n = this->n;

  const LFT lenbound = GBL_max_len.load(std::memory_order_relaxed);

  const auto qentry_func = [&bucket,
                            n](const unsigned lhs,
                               const unsigned rhs) -> std::pair<LFT, int8_t> {
    const auto left_len = static_cast<LFT>(bucket[lhs].len);
    const auto right_len = static_cast<LFT>(bucket[rhs].len);
    const LFT inner =
        std::inner_product(bucket[lhs].yr.begin(), bucket[lhs].yr.begin() + n,
                           bucket[rhs].yr.begin(), static_cast<LFT>(0.));
    LFT new_l = left_len + right_len - 2 * std::abs(inner);
    int8_t sign = (inner < 0) ? 1 : -1;
    return {new_l, sign};
  };

  const auto start_pos = t_queue.size();

  for (unsigned block = 0; block < bucket_size; block += CACHE_BLOCK) {
    for (unsigned i = block + 1; i < bucket_size; ++i) {
      const auto jmin = block;
      const auto jmax = std::min(i, block + CACHE_BLOCK);
      for (unsigned j = jmin; j < jmax; ++j) {
        if (is_reducible_maybe<XPC_THRESHOLD>(fast_bucket[i], fast_bucket[j])) {
          // Reduce.
          const auto len_and_sign = qentry_func(i, j);
          const auto &e1 = bucket[i];
          const auto &e2 = bucket[j];

          if (len_and_sign.first < lenbound) {
            if (kk < .1 * S) {
              break;
            }

            // Note: this is another area of difference from the regular BDGL
            // code. Rather than insert every time we find
            // a potential reduction, we actually check first if we've already
            // seen it. We only insert if so: this is so we aren't storing many
            // potential duplicates locally.
            const auto new_uid =
                (len_and_sign.second == 1) ? e1.uid + e2.uid : e1.uid - e2.uid;

            if (!uid_hash_table.insert_uid(new_uid)) {
              continue;
            }

            // Now we have to eagerly compute the new vector.
            std::array<ZT, MAX_SIEVING_DIM> new_x = e1.x;
            addsub_vec(new_x, e2.x, static_cast<ZT>(len_and_sign.second));
            Entry new_entry;
            new_entry.x = new_x;
            recompute_data_for_entry<Recompute::recompute_uid>(new_entry);
            if (new_uid != new_entry.uid) {
              // We'll undo everything.
              uid_hash_table.erase_uid(new_uid);
              continue;
            }

            kk -= params.threads;

            // Just add it to the table.
            const unsigned pos =
                mpi.add_to_outgoing_bdgl(new_entry, thread_pool::id, n);
            t_queue.emplace_back(Reduction{new_uid, len_and_sign.first, pos});
          } else if (params.otf_lift &&
                     len_and_sign.first < params.lift_radius) {
            bdgl_lift(e1, e2, len_and_sign.first, len_and_sign.second);
          }
        }
      }
    }
  }

  const auto cmp = [](const Reduction &r1, const Reduction &r2) {
    return r1.len < r2.len;
  };

  // Sort the new insertions and, then, merge into a single sorted list.
  std::sort(t_queue.begin() + start_pos, t_queue.end(), cmp);
  std::inplace_merge(t_queue.begin(), t_queue.begin() + start_pos,
                     t_queue.end(), cmp);
  assert(std::is_sorted(t_queue.cbegin(), t_queue.cend(), cmp));
}

void Siever::bdgl_sieve_batch(
    const unsigned index, const std::vector<uint32_t> &buckets,
    const std::vector<atomic_size_t_wrapper> &bsizes) {

  const auto size = mpi.get_bdgl_batch_size(index);

  // NOTE: this must be done _here_, because then the copy to the lambda is
  // guaranteed to be thread-safe.
  const auto scratch_index = mpi.get_batch_index(index);
  if (UNLIKELY(size == 0)) {
    // NOTE: this must be here. Essentially, this prevents us from
    // holding a scratch buffer hostage if we've processed all of the buckets
    // for us. Equally, it prevents buffer re-use if we're aggressively
    // sending out other batches.
    mpi.dec_batch_use(scratch_index);
    mpi.bdgl_dec_bucket_use(index);
    return;
  }

  const auto bsize = buckets.size() / bsizes.size();
  {
    std::unique_lock<std::mutex> lock(threadpool.get_mut());
    auto &condition = threadpool.get_cond();
    auto &tasks = threadpool.get_tasks();
    for (unsigned i = 0; i < size; i++) {
      // Work out which bucket the thread will process.
      const auto bucket = mpi.get_bdgl_bucket(index, i);
      tasks.emplace(
          [this, i, index, bucket, bsize, &buckets, &bsizes, scratch_index]() {
            auto &thread_entry = mpi.get_thread_entry_bdgl(thread_pool::id);
            auto kk = mpi.get_bdgl_insert_pos(thread_pool::id);

            // Before we labour ourselves through all of this work, we'll check
            // if we actually have any space left to insert or not. If we don't
            // we'll just terminate.
            const auto S = cdb.size();

            if (kk < .1 * S) {
              mpi.dec_batch_use(scratch_index);
              mpi.bdgl_dec_bucket_use(index);
              mpi.mark_bdgl_bucket_as_finished();
              return;
            }

            // We now need to unpack the bucket from the scratch space. We do
            // this based on
            // the size we have locally.
            const auto our_size = bsizes[bucket].val.load();

            // N.B This decrements the batch's reference count, so it acts as a
            // freeing operation if
            // needed.
            mpi.deal_with_finished(index, n, i, our_size, scratch_index,
                                   thread_entry.bucket);

            // We now need to work out the difference in sizes between the
            // number of buckets that we have
            // in the buffer vs the number we have locally.
            const auto full_size = thread_entry.bucket.size();
            const auto size_diff = full_size - our_size;

            // We now resize the cbucket appropriately.
            thread_entry.cbucket.resize(full_size);

            // First we copy over the old entries we had before.
            for (unsigned j = 0; j < our_size; j++) {
              auto cdb_index = buckets[bucket * bsize + j];
              const Entry &e = db[cdb[cdb_index].i];
              const auto pos = j + size_diff;
              thread_entry.bucket[pos] = e;
              thread_entry.cbucket[pos] = e.c;
            }

            // And then recompute the rest. This is mostly carried out in SIMD.
            recompute_data_for_batch<Siever::Recompute::recompute_all>(
                thread_entry.bucket, 0, size_diff);

            // Now we finally just tidy up with rest of the cbucket.
            for (unsigned j = 0; j < size_diff; j++) {
              thread_entry.cbucket[j] = thread_entry.bucket[j].c;
            }

            // Now we can quadratically sieve over the bucket.
            bdgl_sieve_bucket(thread_entry.cbucket, thread_entry.bucket, bsize,
                              thread_entry.t_queue, kk);

            // And now we finally free everything.
            thread_entry.bucket.clear();
            thread_entry.bucket.shrink_to_fit();
            thread_entry.cbucket.clear();
            thread_entry.cbucket.shrink_to_fit();

            // Mark it all as done.
            mpi.mark_bdgl_bucket_as_finished();
            mpi.bdgl_dec_bucket_use(index);
            mpi.bdgl_update_insert_pos(thread_pool::id, kk);
          });
    }

    // Only notify after unlocking: this prevents a race condition.
    lock.unlock();
    // Chances are it'll be a large enough batch that all threads need to do
    // something.
    condition.notify_all();
  }

  if (params.threads == 1) {
    threadpool.wait_work();
  }
}

void Siever::bdgl_dist_queue_create_task(std::vector<Entry> &transaction_db,
                                         int64_t &write_index) {
  const size_t S = cdb.size();
  const auto t_id = thread_pool::id;
  auto &thread_entry = mpi.get_thread_entry_bdgl(t_id);

  const auto Q = thread_entry.t_queue.size();
  auto &queue = thread_entry.t_queue;

  const auto cleanup = [&thread_entry]() {
    thread_entry.t_queue.clear();
    thread_entry.t_queue.shrink_to_fit();
    thread_entry.insert_db.clear();
    thread_entry.insert_db.shrink_to_fit();
  };

  if (Q == 0) {
    cleanup();
    return;
  }

  const auto to_sub = params.threads - params.threads != 1;

  const size_t insert_after = S - 1 - t_id - to_sub * write_index;

  unsigned index;
  for (index = 0; index < Q && write_index >= 0; index++) {
    // Unlike the regular skip code, we detect duplicates by checking if the
    // entry was removed from the hash table (or not).
    if (!uid_hash_table.check_uid(queue[index].uid)) {
      continue;
    }

    const auto lenbound =
        cdb[std::min(S - 1, static_cast<unsigned long>(insert_after +
                                                       to_sub * write_index))]
            .len /
        REDUCE_LEN_MARGIN;

    if (queue[index].len < lenbound) {
      int64_t w_index = write_index--;
      mpi.bdgl_extract_entry(thread_pool::id, transaction_db[w_index],
                             queue[index].pos, n);
      // We'll batch recompute everything later.
      transaction_db[w_index].uid = queue[index].uid;
      transaction_db[w_index].len = queue[index].len;
    } else {
      mpi.mark_as_unused(queue[index].uid);
      uid_hash_table.erase_uid(queue[index].uid);
    }
  }

  // Recompute all of the auxiliary stuff in parallel. This uses SIMD
  // parallelism.
  if (write_index != transaction_db.size() - 1) {
    recompute_data_for_batch<Recompute::recompute_all_and_consider_otf_lift &
                             ~Recompute::recompute_uid &
                             ~Recompute::recompute_len>(
        transaction_db, write_index + 1, transaction_db.size());
  }

  for (; index < Q; index++) {
    uid_hash_table.erase_uid(queue[index].uid);
    mpi.mark_as_unused(queue[index].uid);
  }

  cleanup();
}

void Siever::bdgl_dist_insertions(
    std::vector<std::vector<Entry>> &transaction_db) {
  // This function works by carefully removing any of the insertions that we
  // can't make (by globally checking) and then updating all of the various
  // writes that we need to do.
  mpi.bdgl_run_queries(uid_hash_table, threadpool);

  // Now we'll run the insertion loop. At a high-level, this works almost the
  // same way that the regular variant (that operates on QEntrys) does, but we
  // detect duplicates by looking at their presence in the underlying hash
  // table. If they've been removed, then we know that they're duplicates
  // globally.
  const size_t S = cdb.size();
  const size_t Q = mpi.get_bdgl_queue_size();
  const size_t insert_after = std::max(0, int(int(S) - Q));

  // This parameter requires some explanation. Essentially, there's two
  // situations in the distributed variant of BDGL that we need to care about.
  // 1) The unlikely setting that we have a single threaded node. In that
  // setting the thread
  //    has access to the entire database, and does all of the work.
  // 2) The setting that we have more than one thread. In this setting, one
  // thread does all of the serialisation
  //    work, and the other threads do the sieving work. This means that there's
  //    a single thread that will never have any valid insertions to do;
  //    however, this means that allowing that thread to claim any of the
  //    insertion space will lead to missed insertions. Thus, we adjust slightly
  //    to account for this "off by one" in the insertion space.

  const auto divide_by = params.threads - (params.threads != 1);

  for (unsigned int i = 0; i < params.threads; i++)
    transaction_db[i].resize(std::min(S - insert_after, Q) / divide_by + 1);

  const auto g_write_index = int64_t(transaction_db[0].size() - 1);
  std::vector<std::atomic<size_t>> kk(params.threads);

  threadpool.run([this, &transaction_db, g_write_index, &kk]() {
    auto write_index = g_write_index;
    bdgl_dist_queue_create_task(transaction_db[thread_pool::id], write_index);
    kk[thread_pool::id] = bdgl_queue_insert_task<true>(
        thread_pool::id, transaction_db[thread_pool::id], write_index);
  });

  size_t min_kk = kk[0];
  size_t inserted = 0;
  for (unsigned int i = 0; i < params.threads; i++) {
    min_kk = std::min(min_kk, kk[i].load());
    inserted += (S - 1 - i - kk[i] - params.threads) / params.threads;
  }

  status_data.plain_data.sorted_until = min_kk;
  mpi.bdgl_remove_speculatives(uid_hash_table);
}

bool Siever::bdgl_dist_sieve(const size_t nr_buckets_aim, const size_t blocks,
                             const size_t multi_hash) {

#ifndef G6K_MPI
  // If MPI isn't enabled then calling this is a strict error.
  assert(false);
  std::abort();
#endif

  // Start the timer.
#ifdef MPI_TIME
  const auto _time = mpi.time_bdgl();
#endif

  // The very first thing we do is sort the database and rebuild the
  // histogram, keeping in sync with the original file.
  parallel_sort_cdb();
  recompute_histo(false);
  statistics.inc_stats_sorting_sieve();
  mpi.initialise_thread_entries(params.threads, true);

  // Unlike in the original function, we consider the size of the whole
  // database.
  const auto S = mpi.get_cdb_size(cdb.size());
  size_t saturation_index = 0.5 * params.saturation_ratio *
                            std::pow(params.saturation_radius, n / 2.0);

  if (saturation_index > 0.5 * S) {
    if (mpi.is_root()) {
      std::cerr << "Saturation index larger than half of db size" << std::endl;
    }
    saturation_index = std::min(saturation_index, S - 1);
  }

  // Start the initial bucketing work.
  std::vector<uint32_t> buckets;
  std::vector<atomic_size_t_wrapper> buckets_i;
  bdgl_distributed_bucketing(S, blocks, multi_hash, nr_buckets_aim, buckets,
                             buckets_i);

  size_t it = 0;
  thread_pool::id = threadpool.size();
  std::vector<std::vector<Entry>> transaction_dbs(params.threads);

  // Check (globally) that the database starts in a consistent state.
  assert(mpi.bdgl_gbl_uniques(db));

  while (true) {
    mpi.test();

    if (UNLIKELY(mpi.has_bdgl_finished_sieving())) {
      // Check the uniqueness before the insert.
      // assert(mpi.bdgl_gbl_uniques(db));
      mpi.bdgl_pass_stop();

      // There's no database integrity here because we insert the uids that are
      // pending insertions: thus, we'd expect to have many more entries in the
      // hash table compared to the database.

      bdgl_dist_insertions(transaction_dbs);
      parallel_sort_cdb();
      threadpool.run(
          [this]() { std::atomic_thread_fence(std::memory_order_seq_cst); });

      // Check the uniqueness after the insertions, too.
      // assert(mpi.bdgl_gbl_uniques(db));

      // And check that everything agrees.
      // assert(mpi.bdgl_inserts_consistent(db, uid_hash_table));

      const auto norm = mpi.get_norm_bdgl(saturation_index, *this);

      if (norm <= params.saturation_radius) {
        assert(std::is_sorted(cdb.cbegin(), cdb.cend(), compare_CE()));
        invalidate_histo();
        recompute_histo(false);
        mpi.finish_bdgl();

        // We also need to gather the best lifts at the root too.
        if (mpi.is_root()) {
          best_lifts_so_far =
              mpi.receive_best_lifts_as_root(best_lifts_so_far, full_n);
        } else {
          mpi.share_best_lifts_with_root(best_lifts_so_far, full_n);
        }

        return true;
      }

      // Otherwise, redistribute the buckets and continue.
      if (it > 10000) {
        if (mpi.is_root()) {
          std::cerr << "Not saturated after 10000 iterations" << std::endl;
        }
        return false;
      }
      // Gather the lift bounds too.
      mpi.grab_lift_bounds(lift_bounds, lift_max_bound);
      ++it;

      // Now redist.
      bdgl_distributed_bucketing(S, blocks, multi_hash, nr_buckets_aim, buckets,
                                 buckets_i);
    }

    const auto finished_buckets = mpi.finished_buckets();
    for (const auto batch : finished_buckets) {
      bdgl_sieve_batch(batch, buckets, buckets_i);
    }

    mpi.bdgl_decrement_outstanding_batches(finished_buckets.size);
    mpi.forward_and_gather_all(n, buckets, cdb, db);

    if (mpi.bdgl_has_finished()) {
      mpi.issue_stop_barrier();
    }
  }
}
