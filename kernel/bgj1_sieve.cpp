/***
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

// clang-format off

#include "siever.h"
#include "iostream"
#include "fstream"
#include <numeric>
#include <atomic>
#include <thread>
#include <mutex>

/**
    Threaded Bucketed NV Sieve
*/

/**
    bgj1_sieve realizes a threaded bucketed NV sieve.
    The algorithm roughly works as follows:

    It chooses random bucket centers aux from (c)db.
    It then creates a "bucket" of vectors that are (up to sign) close to it. (Bucketing phase)
    Then, it searches for reductions within elements from the bucket.
    Since the bucket centers are lattice points, we may also find reductions
    during the bucketing phase and we use them as well.
    For better concurrency, newly found reductions are first put into a (thread-local) database
    of pending db insertions and only inserted later.
    Insertion is performed by overwriting the (presumed) longest elements from the db (i.e. at the end of cdb).
    After a certain amount of insertions, we resort.
    The parameter alpha controls what vectors we put into a bucket (up to missing vectors due to concurrency issues or imperfect simhashes):
    We put x into the bucket with center aux if |<x, aux>| > alpha * |x| * |aux|
    We do not grow the database inside this algorithm. This has to be done by the caller.
*/


/**
    Siever::bgj1_sieve is what is called from outside.

    This function just sets up "global" parameters for the bgj1_sieve, dispatches to worker threads and cleans up afterwards.
*/
void Siever::bgj1_sieve(double alpha)
{
    CPUCOUNT(100);
    switch_mode_to(SieveStatus::bgj1);
    assert(alpha >= 0); // negative alpha would work, but not match the documentation.

    // Needs to be done before everything else so the attached ranks all call
    // into the cdb size function.
    if(mpi.is_root() && mpi.should_sieve(n, params.dist_threshold)) {
      mpi.start_bgj1(alpha);
    }

    mpi.in_sieving();
    
    const bool should_dist_sieve = mpi.should_sieve(n, params.dist_threshold);
    // g_size is the global DB size.
    size_t const g_size = (should_dist_sieve) ? mpi.get_cdb_size(cdb.size()) : cdb.size();    
   
    if(g_size == 0) {
      mpi.out_of_sieving();
      return;
    }
        
    if(mpi.is_distributed_sieving_enabled() && !mpi.is_active()
       && should_dist_sieve) {
      split_database();
    }
    
    parallel_sort_cdb();
    statistics.inc_stats_sorting_sieve();
    size_t const S = cdb.size();
      
    // TODO: Its possible a particular rank may have no vectors here. This could cause hangs if (say) the collective calls
    // are expecting all ranks to participate. This should be fixed. 
    if(S == 0) {
      mpi.out_of_sieving();
      return;
    }

    // This is explained in more detail in bgj1_round_sieve, but essentially we add an extra portion
    // on to the end of the cdb and store the bucket centers from other processes there.
    // This variable simply makes sure that certain database operations (e.g sorting) do not
    // integrate received centers. 
    stop_pos = S;
    assert(cdb.cbegin() + stop_pos == cdb.cend());

    // initialize global variables: GBL_replace_pos is where newly found elements are inserted.
    // This variable is reset after every sort.
    // For better concurrency, when we insert a bunch of vectors, we first (atomically) decrease GBL_replace_pos
    // (this reserves the portion of the database for exclusive access by the current thread)
    // then do the actual work, then (atomically) decrease GBL_replace_done_pos.
    // GBL_replace_done_pos is used to synchronize (c)db writes with resorting.
    GBL_replace_pos = S - 1;
    GBL_replace_done_pos = S - 1;

    // GBL_max_len is the (squared) length bound below which we consider vectors as good enough for
    // potential insertion. This formula ensures some form of meaningful progress:
    // We need to at least improve by a constant fraction REDUCE_LEN_MARGIN
    // and we need to improve the rank (in sorted order) by at least a constant fraction.
    GBL_max_len = cdb[params.bgj1_improvement_db_ratio * (cdb.size()-1)].len / REDUCE_LEN_MARGIN;
    ENABLE_IF_STATS_GBL_ML_VARIANCE(if(mpi.is_active()) {
        const auto len = mpi.gather_gbl_ml_variance(GBL_max_len);
        if(mpi.is_root()) {
          statistics.set_gbl_max_len_variance(len);
        }
    })

    // maximum number of buckets that we try in-between sorts.
    // If we do not trigger sorting after this many buckets, we give up / consider our work done.
    // NOTE: in distributed sieving it doesn't appear to matter much if this is S or g_size.
    // This is primarily because we quit whenever we've saturated or whenever we hit 0 trials
    // on any node. However, for those edge cases where this does matter, we use g_size anyway. 
    GBL_max_trial = 100 + std::ceil(4 * (std::pow(g_size, .65)));
    GBL_remaining_trial = GBL_max_trial; // counts down to 0 to realize this limit.

    // points to the beginning of cdb. Note that during sorting, we first sort into a copy
    // and then swap. Publishing the new cdb is then done essentially by atomically changing this pointer.
    GBL_start_of_cdb_ptr = cdb.data();

    // We terminate bgj1 if we have (roughly) more than the initial value of GBL_saturation_count
    // vectors in our database that are shorter than params.saturation_radius (radius includes gh normalization).
    // We decrease this until 0, at which point we will stop.
    // Remark: When overwriting a vector that was already shorter than the saturation_radius, we might double-count.
    // For parameters in the intended ranges, this is not a problem.
    GBL_saturation_count = std::pow(params.saturation_radius, n/2.) * params.saturation_ratio / 2.;

    // current number of entries below the saturation bound. We use the fact that cdb is sorted.
    size_t cur_sat = std::lower_bound(cdb.cbegin(), cdb.cend(), params.saturation_radius, [](CompressedEntry const &ce, double const &bound){return ce.len < bound; } ) - cdb.cbegin();
    ENABLE_IF_STATS_GBL_SAT_VARIANCE(if(mpi.is_active()) {
        const auto sat = mpi.gather_gbl_sat_variance(cur_sat);
        if(mpi.is_root()) {
          statistics.set_gbl_sat_variance(sat);
        }
    })
    
    // Now we'll collect the saturation count globally, if applicable.
    cur_sat = mpi.get_global_saturation(cur_sat);
    if (cur_sat >= GBL_saturation_count) {
      mpi.out_of_sieving();
      return;
    }

   
    GBL_saturation_count -= cur_sat;
    if(!mpi.is_active()) {
      // We dispatch to worker threads. UNTEMPLATE_DIM overwrites task by bgj1_sieve_task<dim>
      // i.e. a pre-compiled version of the worker task with the dimension hardwired.
      auto task = &Siever::bgj1_sieve_task<-1>;
      UNTEMPLATE_DIM(&Siever::bgj1_sieve_task, task, n);
      
      for (size_t c = 0; c < params.threads; ++c)
      {
          threadpool.push([this,alpha,task](){((*this).*task)(alpha);});
      }
      threadpool.wait_work();
    } else {
      auto task = &Siever::bgj1_round_sieve<-1>;
      UNTEMPLATE_DIM(&Siever:bgj1_round_sieve, task, n);

      ((*this).*task)(alpha);
      threadpool.wait_work();
      
      if(mpi.is_root()) {
        best_lifts_so_far = mpi.receive_best_lifts_as_root(best_lifts_so_far, full_n);
      } else {
        mpi.share_best_lifts_with_root(best_lifts_so_far, full_n);
      }

      // Collect up the number of unique sends and receives.
      // This only does something if this is compile-time enabled. 
      mpi.collect_statistics(statistics);
    }

    mpi.out_of_sieving();

    invalidate_sorting();
    statistics.inc_stats_sorting_sieve();
    
    status_data.plain_data.sorted_until = 0;

    if(mpi.is_distributed_sieving_enabled()) {
      invalidate_histo();
    }
    
    // we set histo for statistical purposes
    if(mpi.is_root() || !mpi.is_distributed_sieving_enabled()) {
      recompute_histo(); // TODO: Remove?
    }

    // Finally, gather the stats if sieving is enbaled.
    if(mpi.is_root() && mpi.is_active()) {
      mpi.write_stats(n, db.size());      
    }
 
    return;
}

template<int tn, bool ours>
void Siever::bgj1_bucket(const double alpha,
                         const size_t fullS,
                         const CompressedEntry& ce,                         
                         std::vector<unsigned>& buckets,
                         std::vector<Entry>& transaction_db) {
  ATOMIC_CPUCOUNT(110);
  assert(tn < 0 || static_cast<unsigned int>(tn) == n);
  
  ENABLE_IF_STATS_REDSUCCESS ( auto &&local_stat_successful_red_outer  =
                               merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_2redsuccess_outer(val); }); )
  

  const auto yr1 = db[ce.i].yr;
  CompressedVector cv;
  if(ours) {
    cv = sim_hashes.compress(yr1);
  } else {
    cv = ce.c;
  }
  
  LFT maxlen = GBL_max_len;
  
  CompressedEntry* const fast_cdb = GBL_start_of_cdb_ptr; // atomic load
  LFT const alpha_square_times_len = alpha * alpha * ce.len;

  for(size_t j = 0; j < fullS; ++j) {
      if(UNLIKELY(is_reducible_maybe<XPC_BUCKET_THRESHOLD>(cv, fast_cdb[j].c))) {
        if(ours && UNLIKELY(cdb[j].i == ce.i)) continue;
        LFT const inner = std::inner_product(yr1.cbegin(), yr1.cbegin() + (tn < 0 ? n : tn),
                                             db[fast_cdb[j].i].yr.cbegin(), static_cast<LFT>(0.));
        #if (COLLECT_STATISTICS_OTFLIFTS >= 2) || (COLLECT_STATISTICS_REDSUCCESS >= 2) || (COLLECT_STATISTICS_DATARACES >= 2)
        if (bgj1_reduce_already_computed<tn>(inner, maxlen, ces[i].len, fast_cdb[j].len, db[ces[i].i], db[fast_cdb[j].i], transaction_db, db.data(), true)) {
            ENABLE_IF_STATS_REDSUCCESS ( ++local_stat_successful_red_outer; )
            if (UNLIKELY(transaction_db.size() > params.bgj1_transaction_bulk_size))
            {
              if (bgj1_execute_delayed_replace(transaction_db, false))
                maxlen = GBL_max_len;
             }  
         }
         #else
         if(bgj1_reduce_already_computed<tn>(inner, maxlen, ce.len, fast_cdb[j].len, db[ce.i], db[fast_cdb[j].i], transaction_db)) {
            if (UNLIKELY(transaction_db.size() > params.bgj1_transaction_bulk_size))
            {
              if (bgj1_execute_delayed_replace(transaction_db, false))
                maxlen = GBL_max_len;
             }
          }
          #endif
           
           // Test for bucketing
           if (UNLIKELY(inner * inner > alpha_square_times_len * fast_cdb[j].len))
           {
             buckets.emplace_back(fast_cdb[j].i);
           }
       }
  }  
}

template<int tn>
void Siever::bgj1_sieve_over_bucket(const std::vector<CompressedVector> &cbucket,
                                const std::vector<Entry> &bucket,
                                std::vector<Entry> &transaction_db) {

  
  ATOMIC_CPUCOUNT(112);
  LFT maxlen = GBL_max_len;
  size_t const S = cbucket.size();
  auto db  = bucket.data();

      // statistics are collected per-thread and only merged once at the end. These variables get merged automatically at scope exit.
    #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
        // number of successful xorpopcnt tests in the second phase
        auto &&local_stat_successful_xorpopcnt_reds = merge_on_exit<unsigned long long>( [this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_pass_inner(val);
            statistics.inc_stats_fullscprods_inner(val);
        } );
    #endif
    #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
        // number of 2-reduction attempts in the second phase
        auto &&local_stat_2red_attempts = merge_on_exit<unsigned long long>([this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_inner(val);
            statistics.inc_stats_2reds_inner(val);
        } );
    #endif
    ENABLE_IF_STATS_REDSUCCESS ( auto &&local_stat_successful_reductions = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_2redsuccess_inner(val); }); )

  CompressedVector const * const fast_bucket = cbucket.data();    
  
  for(size_t block = 0; block < S; block += CACHE_BLOCK) {
      for(size_t i = block + 1; i < S; ++i) {
       // We don't need to skip here because there's no synchronisation.
       size_t const jmin = block;
       size_t const jmax = std::min(i, block + CACHE_BLOCK);     
       for(size_t j = jmin; j < jmax; ++j) {
          if(UNLIKELY(is_reducible_maybe<XPC_THRESHOLD>(fast_bucket[i], fast_bucket[j]))) {    
              #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
                     ++local_stat_successful_xorpopcnt_reds; // adds to successful xorpopcnt and also to scalar product computations (done inside bgj1_reduce_with_delayed_replace)
              #endif
              ATOMIC_CPUCOUNT(103);
              // We compute this here because we do not use CompressedEntries.
              LFT const inner = std::inner_product(db[i].yr.cbegin(), db[i].yr.cbegin() + (tn < 0 ? n : tn), db[j].yr.cbegin(),
                                                   static_cast<LFT>(0.));
              bool const red = bgj1_reduce_already_computed<tn>(inner, maxlen, db[i].len, db[j].len, db[i], db[j], transaction_db);              
              ENABLE_IF_STATS_REDSUCCESS(if(red) {++local_stat_successful_reductions; } )
              if (UNLIKELY(red) && transaction_db.size() > params.bgj1_transaction_bulk_size)
              {
                if (bgj1_execute_delayed_replace(transaction_db, false))
                {
                  if (GBL_remaining_trial.load(std::memory_order_relaxed) < 0)
                  {
                    #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
                       local_stat_2red_attempts += (1 + j - jmin); // statistics for half-finished loop
                    #endif
                    return;
                  }
                  maxlen = GBL_max_len;
                }
              }
          }
      }
      #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
         local_stat_2red_attempts += jmax - jmin;
      #endif
     }
  }
}

template<int tn>
void Siever::bgj1_bucket_ours(const double alpha, const size_t fullS,
                              const unsigned batch, 
                              const std::vector<CompressedEntry>& centers,
                              std::vector<std::vector<unsigned>>& positions) {
                              
  const auto size = centers.size();
  for(unsigned i = 0; i < size; i++) {
    threadpool.push([this, batch, i, alpha, fullS, &centers, &positions]() {
      ATOMIC_CPUCOUNT(108);
      thread_pool::batch = batch;
      auto& tdb = mpi.get_thread_entry(thread_pool::id).transaction_db;
      positions[i].clear();
      bgj1_bucket<tn, true>(alpha, fullS, centers[i], positions[i], tdb);  
      mpi.inc_bucketed_count(batch);
      mpi.batch_insertions(thread_pool::id, batch, n);
    });
  }
       
  if(params.threads == 1) {
    threadpool.wait_work();
  }
}

template<int tn>
void Siever::bgj1_sieve_our_buckets(const unsigned index, const std::vector<std::vector<unsigned>>& positions) {
  const auto size = mpi.get_bucket_size(mpi.get_rank());

  // NOTE: this must be done _here_, because then the copy to the lambda is
  // guaranteed to be thread-safe.
  const auto scratch_index = mpi.get_batch_index(index);
  
  // Note: profiling indicates that if we use the regular
  // threadpool.push() here that we spend a lot of time
  // sleeping. To make this easier, we manaully do
  // the entire thing in one go.
  mpi.start_sieving_bucket(index);
  {
    std::unique_lock<std::mutex> lock(threadpool.get_mut());
    auto& condition = threadpool.get_cond();
    auto& tasks = threadpool.get_tasks();
    for(unsigned i = 0; i < size; i++) {
      tasks.emplace([this, index, i, &positions, scratch_index]() {
        ATOMIC_CPUCOUNT(111);
        thread_pool::batch = index;
        if (GBL_remaining_trial.load(std::memory_order_relaxed) < 0) {
          // WARNING: this call _must_ be here.
          // This call prevents the batch from being marked as used long after
          // it is done. Essentially, for performance reasons we don't unpack a
          // particular bucket if the sieve is finished. However, because batch
          // buffers have their lifetimes managed based on the number of
          // "unpackings" that we do, we need to mark that we've done an
          // unpacking, even if we haven't. You can view this as decrementing a
          // reference counter.
          mpi.dec_batch_use(scratch_index);
          mpi.inc_sieved_buckets(index);
          return;
        }

        auto& thread_entry = mpi.get_thread_entry(thread_pool::id);

        {
          // We now need to unpack the bucket from the scratch space. We do this based on
          // the size we have locally.
          const auto our_size = positions[i].size();
        
          // N.B This decrements the batch's reference count, so it acts as a freeing operation if
          // needed. 
          mpi.deal_with_finished(index, n, i, our_size, scratch_index, thread_entry.bucket);

          // We now need to work out the difference in sizes between the number of buckets that we have
          // in the buffer vs the number we have locally.
          const auto full_size = thread_entry.bucket.size();
          const auto size_diff = full_size - our_size;

          // We now resize the cbucket appropriately.        
          thread_entry.cbucket.resize(full_size);

          // First we copy over the old entries we had before.
          for(unsigned j = 0; j < our_size; j++) {
            const Entry& e = db[positions[i][j]];
            const auto pos = j + size_diff;
            thread_entry.bucket[pos] = e;          
            thread_entry.cbucket[pos] = e.c;
          }

        
        
          // And then recompute the rest. We use the SIMD code for this.
          recompute_data_for_batch<Siever::recompute_all_no_otf_lift()>(thread_entry.bucket, 0, size_diff);

          for(unsigned j = 0; j < size_diff; j++) {
            Entry& e = thread_entry.bucket[j];
            thread_entry.cbucket[j] = thread_entry.bucket[j].c;
          }

          mpi.count_uniques(thread_entry.bucket);          
        }
        // And now finally we can sieve.
        if (GBL_remaining_trial.load(std::memory_order_relaxed) > 0) {
          // With that done, we can just sieve and be on our way.
          bgj1_sieve_over_bucket<tn>(thread_entry.cbucket, thread_entry.bucket, thread_entry.transaction_db);
        }

        // Free the memory we've allocated.
        thread_entry.bucket.clear();
        thread_entry.bucket.shrink_to_fit();
        thread_entry.cbucket.clear();
        thread_entry.cbucket.shrink_to_fit();
        
        // N.B This must be before the inc_sieved_buckets: this makes it so that the reading thread will
        // not grab the mutex to serialise sizes for insertions before the insertions themselves have finished. 
        mpi.batch_insertions(thread_pool::id,index, n);
        mpi.inc_sieved_buckets(index);        
      });
    }

    // Only notify after unlocking: this prevents a race condition.  
    lock.unlock();
    // Now we should be in the case where the number of buckets is at least the number of threads.
    assert(size >= params.threads);
    // And so we can just wake them all, rather than repeatedly notifying just one.
    condition.notify_all();
  }
      
  if(params.threads == 1) {
    threadpool.wait_work();
  }
}
                              
template<int tn>
void Siever::bgj1_bucket_others(const unsigned batch, const double alpha,
                                const size_t fullS, std::vector<std::vector<std::vector<unsigned>>>&buckets) {

  // This function builds the buckets for a particular batch. This is done across multiple threads
  // at once to maximize parallelism.
  // Recall that each other process writes their centers into an extra portion of our database.
  // Thus, we just invoke the normal bucketing code on those centers and build the results.
  // N.B To prevent lock contention, we insert all of these tasks into the queue in one go.
  {
    std::unique_lock<std::mutex> lock(threadpool.get_mut());
    auto& condition = threadpool.get_cond();
    auto& tasks = threadpool.get_tasks();
   
    for(unsigned rank = 0; rank < buckets.size(); rank++) {      
      if(rank == static_cast<unsigned>(mpi.get_rank())) continue;
      auto& i_buckets = buckets[rank];

      const auto size_and_start = mpi.get_size_and_start(rank, batch);
      const auto size = size_and_start[0];
      const auto start = size_and_start[1];

      for(unsigned j = 0; j < size; j++) {
        tasks.emplace([this, rank, size, start, &i_buckets, fullS, batch, alpha, j]() {
          ATOMIC_CPUCOUNT(109);
          auto& tdb = mpi.get_thread_entry(thread_pool::id).transaction_db;
          thread_pool::batch = batch;
          Entry& e = db[j+start];
          CompressedEntry& ce = cdb[j+start];
          recompute_data_for_entry<Siever::recompute_recv()>(e);
          ce.c = e.c;
          ce.len = e.len;
          i_buckets[j].clear();
          bgj1_bucket<tn, false>(alpha, fullS, cdb[start+j], i_buckets[j], tdb);
          statistics.inc_stats_xorpopcnt_outer(fullS);
          mpi.inc_bucketed_count(batch);
          statistics.inc_stats_filter_pass(i_buckets[j].size());          
          mpi.batch_insertions(thread_pool::id, batch, n);
        });        
      }
    }
    lock.unlock();
    condition.notify_all();
  }
    
  if(params.threads == 1) {
    threadpool.wait_work();
  }
}

// Worker function for bgj1 distributed sieving. 
template<int tn>
void Siever::bgj1_round_sieve(double alpha) {
#ifndef G6K_MPI
  // If MPI isn't enabled then calling this is a strict error.
  assert(false);
  std::abort();  
#endif
  CPUCOUNT(107);
  assert(tn < 0 || static_cast<unsigned int>(tn)==n);

  // Start the timer.
#ifdef MPI_TIME
  const auto _time = mpi.time_bgj1();
#endif

  mpi.reset_bandwidth();


  /*
    Distributed sieving for BGJ1.

    The algorithm and code used in this function (and the system more generally)
    has been the result of many, many small optimisations and improvements. As a
    result, the code is a little bit difficult to follow: this comment block is
    meant to try to demystify how some of the code works at a high-level, but
    some of the details are better understood by looking at the code itself.

    ## Terminology
    Before continuing, we shall fix some terminology.

    1. We speak of a "bucket" as all vectors in the global database that are
    within a fixed angle of some vector v. In particular, the bucket defined by
    `v` contains all vectors `u` such that v \cdot u >= alpha * |u|.

    2. We speak of a "centre" as the vector that defines a bucket. For example,
    in the above case, we would refer to `v` as a centre.

    3. We speak of a "batch" as a collection of buckets that are processed as
    one job. For example, consider the scenario where a node chooses v_1, v_2 as
    bucket centres and broadcasts them to the rest of the network. In this
    setting, each node must process both v_1 and v_2 before any progress is
    made, and the batch size is 2.

    The size of any given batch is a trade-off. On the one hand, increasing the
    number of buckets per batch increases the amount of work per thread and
    decreases the overhead of sending a single centre. On the other hand, each
    bucket requires some amount of storage. Moreover, G6K's sieving algorithms
    iteratively improve the database as the program runs: thus, choosing too
    many buckets at any given time can increase the number of sieved buckets
    without adding much in extra improvements. This choice stacks with other
    decisions, and we discuss this in more detail later.

    4. We speak of the "number of batches" as the number of batches that are in
    flight at once. In particular, the number of batches that are being
    processed at once can be chosen at runtime. This allows us to establish a
    pipelining effect in the code, reducing latency. We discuss this in more
    detail later.

    5. We define |buckets_per_rank| as the maximum number of buckets that any
    one rank will ask for in a batch. In the same vein, we define
    |number_of_buckets|_i as the number of buckets that rank `i` will issue in
    any particular batch.

    ## Operation

    The algorithm presented here is an iterative algorithm that has 4 distinct
    stages.

    ### Setup

    The setup stage consists of allocating various pieces of auxiliary storage
    that are needed for the algorithm. This stage is actually quite involved: to
    save on certain overheads, we try to allocate as much of the extra memory
    that we will need at once, and then only re-allocate later if we will need
    to (this will always happen).

    In particular, we allocate:

    1. Extra space in the database and the compressed database. This extra space
       is used to hold the incoming centres. This is placed here to make
    bucketing easier: as all of the information is already in the database, we
    can use a modified variant of the existing bgj1 bucketing code. To prevent
    these from centres fromm being sorted into the database, we store the
    original size of the database and consider only that portion as active.
    Practically speaking, we reserve an extra |buckets_per_rank| *
    |bucket_batches| entries in both the cdb and the db.

    2. Extra space for outgoing and incoming insertions. Because we split the
    uid hash table globally for insertions, we need to maintain extra storage
    for outgoing reads and writes. In practice, we found it faster to allocate
    multiple extra buffers for writing: each thread maintains a local buffer for
    each other rank, and we merge the insertion buffer for each rank `i` into a
    global insertion buffer for rank `i` at certain opportune moments. Whilst
    this requires more memory, this also saves on lock contention for
    insertions. On the other hand, we maintain a single incoming insertion
    vector from each other rank. Put differently, rank `i` allocates memory to
    receive insertions from ranks `0`, ..., `i-1`, `i+1`, etc.

       As a further optimisation, we use MPI persistent requests for these
    insertions, which should lower the overhead of doing many large insertions.

    3. The space for our centres. Here, rank `i` reserves  |bucket_batches| * |number_of_buckets|_i extra
    CDB elements in some vectors. These are used to store the current set of
    centres that are being processed by this rank.

    4. The space for the partially built buckets. During bucketing each rank
    produces a partial bucket that represent all vectors in the rank's local
    database that are close to a particular centre. These buckets are later
    combined to form complete buckets. For space saving purposes, we represent
    each vector in the local bucket by their index in the uncompressed database
    (i.e if v = db[i] is close to a bucket centre u, then we store `i` in the
    partial buckets). Here, each rank allocates |ranks| * |bucket_batches| *
    \sum_{i=0}^{m-1} |number_of_buckets|_i lists of 32 bit integers. In
    practice, this requires little storage: if we assume that each vector
    requires 24 bytes of storage as overhead, then the list structure itself
    would require |ranks| * |bucket_batches| * \sum_{i=0}^{ranks-1}
    |number_of_buckets|_i. If we assume that |number_of_buckets|_i is at most
       200, then this structure applied to 30 ranks and 3 batches would take 3 *
    200 * 30 = 18000 lists, which would take us up to 432 KB of storage per
    node. The main cost is the buckets themselves, although this is still not
    too expensive: if we assume the default bucket size of G6K's BGJ1 sieve
    (i.e 3.2 * sqrt(db_size)) then in dimension 150 we would need 3.2 * 4 *
    2^(0.2075d/2) = 620KB per bucket in total. This is not too expensive
    relative to the estimated global database size.

    5. A set of used centres. This is used to prevent duplicates during the
    bucketing stage. This is rather cheap.

    6. Space for the incoming centres. For better performance, we serialise all
    incoming (and outgoing centres) into a single buffer. Note that this cannot
    be the space in the database, because we also send other synchronisation
    information at the same time (see bucketing below). This is not too
    expensive, taking around |full_dimension - active_dimension| * sizeof(FT) +
    |active_dimension| * |bucket_batches| * |buckets_per_rank| * |ranks| *
    sizeof(ZT) bytes in total. For dimension 149 (with full_dimension = 150) and
    using the same numbers as above, we have that this would take 149 * 3 * 200
    * 30 * 2 = ~5MB per rank.

    7. Space for the gathered buckets. This is the largest storage cost of this
    entire section.

       Essentially, for networking efficiency reasons we copy all of the buckets
       into temporary storage and then serialise them in one go. We also need to
    allocate the space for each incoming bucket piece too, which can be quite
    expensive. To lower this cost, we serialise buckets in the short term in
    their coefficient representation. Doing this naively would lead to a high
    cost cost: using the same numbers as before, we would expect each bucket to
    require N = 3.2 * 2^(0.2075d/2) * 2 * d bytes to serialise in total. For d =
    150 and with the same numbers as before, we would expect a total storage
    usage of around 2 * N * |buckets_per_rank|
       * |bucket_batches| = 278.7 GB in total across the entire cluster. This
    works out at about 10GB per node. This is rather expensive, but we would
    expect a comparative sieving database for the same dimension to
    take 3.2*2^(0.2075*150) * 1KB = ~7.5TB of memory in total, so this is around
    a 3% growth in storage. Naturally, this cost can be lowered by changing
    certain parameters.

    We can do even better, though, if we use a pipelined job system. Indeed, we
    can note that we might not expect to have many batches in use at once. This
    presents a good opportunity for re-use. Specifically, we use a system that
    means we re-use these buffers whilst the batch is being sieved, provided
    that we have fully unpacked it. Experiments in dimension 90 (bucket_batches
    = 5, scale_factor = 2, 2 * 10 threads) indicate that this means we use
    around 57MB of peak storage for these buffers: various improvements are
    made by freeing the memory after it is used. To allow this to be customised
    by any user, this can be controlled via the scratch_buffers parameter. Note that this
    is a trade-off: too low of a value here will mean that many batches will be contesting
    the same scratch buffers, which will cause a slowdown. On the other hand, this allows
    the memory use to be somewhat configured. As an added benefit, this also allows the
    CPU utilisation to go up: if one sets scratch_buffers = bucket_batches, then nodes have
    a tendency to spend time waiting for work, as the batches seem to be processed at approximately
    the same speed over a slow network. On the other hand, having more bucketing requests on the go
    gives a natural pipelined system that works quite well. 

       As we process buckets using threads, we allow each thread to allocate
    their own vector of entries, compressed entries, and a transaction db. This
    cost can be tailored per node: as each thread only processes a single bucket
    at once, the total memory cost per node is |threads| * 3.2*2^(0.2075d/2) *
    ~1KB. Again, taking d = 150 and (say) threads = 200, we get a total cost of
    around 200 * 3.2*2^(0.2075*75) * 1KB ~= 31GB per node. This cost translates
    to about 150MB per thread in d = 150 in a massively parallel setting. This
    is a rather large cost: it takes around 930GB across a 30 node cluster at
    these parameter sizes. On the other hand, this is still only just over an
    additional 10% in storage cost compared to the global database. Indeed, this
    ratio will drop to 0 as n->\infinity, (|number of threads| * |ranks| *
    2^(0.2075d/2)) / 2^(0.2075d) is dominated by the denominator, and |number of
    threads| * |ranks| will be a constant for a particular runtime
    configuration. However, other tweaks such as e.g the one above should help
    reduce the total amount of memory used at once in practice.

    ### Bucketing

       Bucketing is quite straightforward. For a given batch `j`,  each rank `i`
    chooses a set of |number_of_buckets|_i centres and broadcasts these to the
    rest of the network. This operation is carried out collectively using an
    Iallgather operation, which is asymptotically optimal for this sort of
    message from a communication perspective. In order to convey extra
    information, each rank also sends both a 64-bit header and the
    max_lift_bounds vector. In the first case, the header contains 1. a 32-bit
    integer denoting if the rank has finished sieving, and 2. a 32-bit integer
    containing how many short vectors the rank has inserted since the last
    broadcast. This is necessary for the algorithm to terminate correctly (see
    the "termination" stage). Note that each rank also tracks how often they've broadcast
    a new set of centres for a particular batch. This is very important for termination later. 

       Upon receiving the batch, each rank checks if any rank has terminated and
    returns the minimum lift bounds globally. Once this is done (and if no other
    work is waiting) each rank produces the partial buckets for the batch `j`.
                                                                    
       Once the partial buckets have been finished, each rank collects the total
    size of the buckets that they will hold. This executed using an Ialltoll
    operation, which is not particularly expensive in this setting. Finally,
    once the size request has been completed, each rank distributes their
    partial buckets to the appropriate other ranks. This is carried out using an
    Ialltoallv operation. Note that Ialltoallv scales differently depending on
    your MPI implementation, so this can quickly become a bottleneck on a slow
    implementation. 

       Note that during the bucketing process each rank may fill up their
    insertion buffers with short vectors. These short vectors are serialised in
    an adhoc manner, with many requests potentially active at once. These
    requests are handled asynchronously with the rest of the program.

    ### Sieving

       When a completed batch is received, each rank sieves their batches. This
    happens in a bucket-by-bucket manner: each bucket is unpacked from the
    temporary storage into the thread-local entry vectors, reconstructed, and
    then sieved. This is the simplest part of the sieve. When a particular batch
    `j` is finished, then we re-issue a new batch and start the whole thing all
    over again.

    ### Teardown

       All good things must come to an end. This sieve does when either 1.
    saturation is reached or 2. any particular rank hits a trial count below 0.
    At this stage, all ranks issue an unblocking MPI barrier to signal that they
    are done. We then make sure that each communicator is in the same state before exiting. 
  */

  size_t const fullS = cdb.size();
  // N.b This is here solely so that when the cdb gets resorted we don't mix in short vectors from other databases.
  // Essentially, we're just making sure we only sort our part of the CDB, and not (potentially) nicer entries from
  // other parts of the distributed sieve. 
  stop_pos = fullS;

  auto const total_nr_buckets = params.bucket_batches * mpi.get_nr_buckets();
         
  // To allow us to use the CDB / DB centric functions already written for BGJ1, we
  // grow the database to allow us to store the received centers somewhere easy: namely,
  // right at the end of the database. 
  // To make sure these aren't overwritten, we don't adjust the GBL_replace_pos or
  // GBL_replace_done_pos, and use the stop_pos (defined above) to stop the sorting routine
  // from touching these entries too. 
  // Note: we don't add extra space for ourselves, since that's unnecessary. 
  const auto total_buckets = params.bucket_batches * mpi.get_total_buckets();
  const auto scratch_space = total_buckets - total_nr_buckets;
     
  // This should be genuinely impossible. 
  assert(scratch_space > 0);
    
  // These will always have a 1:1 correspondence.
  cdb.resize(fullS + scratch_space);
  db.resize(fullS + scratch_space);

  // Make sure the correspondence is exactly 1:1. This means we don't need to reset
  // this each time.
  for(size_t i = fullS; i < fullS + scratch_space; i++) {
    cdb[i].i = static_cast<IT>(i);
  }

  // This sets up the insertion requests, alongside the space for the sync
  // objects. 
  mpi.setup_insertion_requests(n, lift_bounds.size());

  // Let the mpi object know where it should store centers.
  mpi.setup_bucket_positions(fullS);
    
  // This is needed to make sure the pointer can actually be used.
  GBL_start_of_cdb_ptr = cdb.data();

  // These just inform what we should do and when.
  const auto ranks = mpi.number_of_ranks();
  const auto my_rank = mpi.get_rank();

  // Build the centers for the first iteration.
  std::vector<std::vector<CompressedEntry>> centers(params.bucket_batches);
  const auto nr_buckets = mpi.get_bucket_size(my_rank);
  
  for(auto& set : centers) {
    set.resize(nr_buckets);
  }

  // WARNING: What follows is very important.
  // The distributed sieve suffers from a potentially curious issue: if the
  // number of buckets (and the number of buckets) per batch is too high relative to the size of the
  // local databases then the probability of choosing the same center twice is rather high.
  // In particular, this is essentially an instance of the birthday problem in action.
  // This can cause saturation issues if sieving in low dimensions (e.g 50) relative to the number of
  // active buckets at a given time. On the other hand, it's certainly undesirable to have this sort of 
  // collision at all, as it means that we are essentially repeated work for no gain.
  //
  // We try to mitigate this issue as follows. For any particular batch we allow no duplicates:
  // each batch has its own hash table that is uses to maintain these invariants. Notably, we do not
  // try and ensure this sort of consistency across batches. The only exception to this is the
  // initial set of batches that are sent: in that case, we do ensure consistency, as repetitions there
  // are not useful i.e. the repeated buckets are likely to vary little, if at all, which can lead to
  // a significant cost.
  // Even this has a caveat, though: if the number of initial buckets is larger than the range, then
  // we warn and continue. This can happen in situations where the distribution threshold is simply set too        
  // low.     
  std::vector<std::unordered_set<size_t>> c_sets(centers.size());
  for(auto& set : c_sets) { set.reserve(nr_buckets); }
  
  // We'll re-use this lambda throughout. 
  auto choose_centers = [&](std::vector<CompressedEntry>& center_set, unsigned batch) {
    auto& seen = c_sets[batch];    
    for(auto& v : center_set) {
      size_t j_aux;

      do {
        j_aux = fullS / 4 + (rng() % fullS/4);
      } while(!seen.insert(j_aux).second);

      v = cdb[j_aux];
    }
    seen.clear();
  };

  // This contains the buckets for all other ranks on the network.
  // This vector is indexed by batch, then by rank, and then by set.
  // For example, cbuckets[i] contains the buckets in batch `i`, cbuckets[i][j] contains the buckets for rank `j` in batch
  // `i`, and finally cbuckets[i][j][k] contains all the vectors in our DB that are close to the k-th center of rank j in batch
  //  i.
   
  std::vector<std::vector<std::vector<std::vector<unsigned>>>> cbuckets(params.bucket_batches);
    
  {
    for(auto& v : cbuckets) {
      // We have `ranks` sub buckets in each batch.
      v.resize(ranks);
      for (unsigned i = 0; i < static_cast<unsigned>(ranks); i++) {
        v[i].resize(mpi.get_bucket_size(i));
      }
    }    
  }
  
  
  // Set up the thread-local variables. Each thread has its own set of (cbucket, bucket, transaction_db)
  // that it re-uses, alongside a series of individual vectors for distributed insertions.
  // Because C++ threads seem to sometimes change their IDs after sleeping (!!!)
  // we set a unique identifier and then use that to access the parameters directly. 
  thread_pool::id = threadpool.size();
  mpi.initialise_thread_entries(params.threads);                                                


  // Produce the initial set of buckets.
  {
    const auto trial = GBL_remaining_trial.load();
    std::unordered_set<unsigned> initial_set;
    initial_set.reserve(centers.size() * nr_buckets);

    // See the large comment block above.
    const auto abandon_safe_guard = (fullS / 2) < (centers.size() * nr_buckets);
    if(abandon_safe_guard) {
      std::cerr << "Rank " << my_rank << " is requesting more unique buckets than it can allocate. " << std::endl;
    }
    
    const auto size = centers.size();
    for(unsigned i = 0; i < size; i++) {
      auto& center_set = centers[i];
      for(auto& v : center_set) {
        size_t j_aux;

        do {
          j_aux = fullS / 4 + (rng() % fullS/4);         
        } while(!initial_set.insert(j_aux).second && !abandon_safe_guard);

        v = cdb[j_aux];
      }

      mpi.start_batch(db, center_set, i, trial, lift_bounds);
      bgj1_bucket_ours<tn>(alpha, fullS, i, center_set, cbuckets[i][my_rank]);
      ENABLE_IF_STATS_BUCKETS(statistics.inc_stats_buckets(nr_buckets);)
    }
  }
  
  // Warning: this loop terminates in a way that might be a little bit confusing if you aren't familiar with MPI.
  // Essentially, MPI's collective communications (e.g. broadcasting) need to be called in the same order across all processes
  // using a particular communicator. The bucketing and sieving loop below ensures that this happens whilst the sieve is running.
  // However, at termination time we need to make sure that all processes end in the same position across all communicators, which
  // is tricky.
  // To solve this issue, we "run forwards" with a barrier: once the global stopping condition has been met, each process continues as normal,
  // issuing a non-blocking barrier once they've finished sieving their own local buckets. Once the barrier has been cleared (i.e. all processes have
  // finished the particular bucketing and sieving iteration) the loop terminates.
  unsigned iter = 0;
  while(!mpi.is_done()) {
    // This call actually carries out all of the work: for performance reasons, we process
    // all requests in a single call (other than the synchronisation call for now).
    // Each call to the mpi object below queries and acts on already produced data. 
    mpi.test();
 
    // The first thing we do is check if the stop barrier has been triggered.
    // This only happens if we've finished sieving across all processes, which
    // requires all other processes to issue a barrier.
    if(UNLIKELY(mpi.is_initial_barrier_completed())) {
      mpi.process_stop();
    }
      
    {
      // This function call simply processes incoming syncs.
      // The syncs have already arrived, and this call just updates
      // the lift bounds.
      const auto change = mpi.process_incoming_syncs(lift_bounds, db, lift_max_bound);
      // The leading guard here is to stop us from doing this if we're
      // already in a stopped state. This is because there may be
      // an additional bucketing round if we've stopped.
      // The second case is a little bit stranger. It should read "
      // if any other process has a trial count below 0, then stop."      
      if(UNLIKELY(mpi.can_issue_barrier() && (change.any_sub_zero ||
                 (change.sat_drop &&
                  GBL_saturation_count.fetch_sub(change.sat_drop, std::memory_order_relaxed) <= change.sat_drop)))) {        
        GBL_remaining_trial = -1;
        mpi.issue_stop_barrier();
      }

      for(const auto batch : mpi.finished_syncs()) {
        bgj1_bucket_others<tn>(batch, alpha, fullS, cbuckets[batch]);        
      }
    }

    {
      const auto finished_buckets = mpi.finished_processed_buckets();
      for (const auto batch : finished_buckets) {
        mpi.send_sizes(batch, cbuckets[batch]);
      }
    }

    // The serialisation code for the buckets requires a consistent ordering everywhere to prevent
    // deadlocks. We handle this complexity inside the MPI wrapper.
    mpi.forward_and_gather_all(n, cbuckets, db);
          
    
    // And now we can check if any incoming requests have finished.
    {
      const auto finished_buckets = mpi.finished_buckets();
      for(auto batch : finished_buckets) {
        mpi.bucketing_request_finished(batch);
        for(unsigned i = 0; i < cbuckets[batch].size(); i++) {
          if(i == my_rank) continue;
          for(auto& v : cbuckets[batch][i]) {
            std::vector<unsigned> tmp;
            v.swap(tmp);            
          }
        }
        bgj1_sieve_our_buckets<tn>(batch, cbuckets[batch][my_rank]);
      }
    }

    // Now we check which ones have finished sieving. Those that have finished
    // are used for insertion sizes.
    {
      const auto finished_sieving = mpi.finished_sieving();
      if(UNLIKELY(finished_sieving.size > 0)) {
        GBL_remaining_trial.fetch_sub(nr_buckets*finished_sieving.size);
      }

      for(const auto batch : finished_sieving) {
        for(auto& ent : cbuckets[batch]) {
          for(auto& v : ent) {
            v.clear();
            v.shrink_to_fit();
          }
        }
        
        mpi.send_insertion_sizes(batch, n);
      }
    }

    // Now we check which incoming insertions sizes have finished.
    {
      const auto finished_insertion_sizes = mpi.finished_insertion_sizes();
      for(const auto batch : finished_insertion_sizes) {
        mpi.start_insertions(batch, n);
      }
    }

    // And finally we check which insertions have actually finished.
    {
      const auto finished_inc_insertions = mpi.finished_incoming_insertions();
      for(const auto batch : finished_inc_insertions) {
        bgj1_insert_others(batch);
      }
    }

    // And now we check which of these insertions have actually finished and,
    // depending on if we can, we re-issue centres.
    {
      const auto finished_insertions = mpi.finished_insertions();
      for(const auto batch : finished_insertions) {
        mpi.mark_batch_as_finished(batch);

        if(params.bucket_batches == 1) {
          mpi.print_uniques(iter++);
        }
        
        if(LIKELY(mpi.can_issue_more(batch))) {
          choose_centers(centers[batch], batch);
          // N.B we load on each iteration here so that we don't fall out of sync with          
          // regards to termination on another node.                        
          mpi.start_batch(db, centers[batch], batch, GBL_remaining_trial.load(), lift_bounds);
          bgj1_bucket_ours<tn>(alpha, fullS, batch, centers[batch], cbuckets[batch][my_rank]);          
          ENABLE_IF_STATS_BUCKETS(statistics.inc_stats_buckets(nr_buckets);)
        } else if (mpi.can_issue_stop_barrier()) {         
          // If all of our buckets are done we issue a barrier. 
          // N.B This actually only happens once per sieving iteration.           
          mpi.issue_stop_barrier();        
        }
      }
    }
  }

  // We need to wait for any outstanding tasks to be done. This is primarily because we may have
  // work that we haven't finished yet, and some of this might be useful. 
  threadpool.wait_work();
  
  // Now we estimate how many extra bytes we've used in this portion of the code.
  // We do this using getrusage, since we aggressively free memory whenever we can.
  // This detail in handled in the MPI object, though.
  mpi.set_extra_memory_used(fullS * (sizeof(CompressedEntry) + sizeof(Entry)));

  // Cancel any requests that might be ready to go. This is to prevent
  // us from having overlaps later.  
  mpi.cancel_outstanding();
  
  // Finally we shrink down to the old sizes to prevent the DB growing endlessly.
  // Note that this will only throw away the "extra" portion, which means this won't
  // have an impact on any parts of the database we actually want to preserve. 
  cdb.resize(fullS);
  db.resize(fullS);
}

void Siever::bgj1_add_to_tdb(std::vector<Entry>& tdb, Entry& entry, const FT maxlen) {
  recompute_data_for_entry<Recompute::recompute_uid>(entry);
  if(mpi.owns_uid(entry.uid) && uid_hash_table.check_uid(entry.uid)) {
    recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift & (~Recompute::recompute_uid)>(entry);
    if(entry.len < maxlen && uid_hash_table.insert_uid(entry.uid)) {
      tdb.emplace_back(entry);
    }
  }
}

void Siever::bgj1_insert_others(const unsigned batch, const bool force) {  
  // N.B As insertions has a lifetime that's tied to this function, we need to
  // move it into the std::function.
  mpi.start_insertions(batch);
  threadpool.push([batch, this, force]() {
    auto& tdb = mpi.get_thread_entry(thread_pool::id).transaction_db;    
    auto& from_other = mpi.get_insertion_candidates(batch);
    thread_pool::batch = batch;
    FT maxlen = GBL_max_len;
    unsigned curr{};
    Entry e;
    unsigned size = from_other.size()/n;
    for(unsigned i = 0; i < size; i++) {
      std::copy(from_other.cbegin() + curr, from_other.cbegin() + curr + n, e.x.begin());
      curr += n;        
      bgj1_add_to_tdb(tdb, e, maxlen);
      // N.B This is done on each iteration to stop the tdb from getting too large.
      if(tdb.size() > params.bgj1_transaction_bulk_size) {
        bgj1_execute_delayed_replace(tdb, force);
        maxlen = GBL_max_len;
      }
    }
    mpi.mark_insertion_as_done(batch);
    if(force) {
      bgj1_execute_delayed_replace(tdb, force);
    }
    /*
    std::cerr << "[Rank " << mpi.get_rank() << "] Batch " << batch << " Already had:" << already_had << " nr:" << size << " too_long:" << too_long << " perc used:" << 100.f * (size - already_had - too_long) / size << std::endl;
    already_had = 0;
    too_long = 0;
    */
  });
    
  if(params.threads == 1) {
    threadpool.wait_work();
  }
}

// Worker task for bgj1 sieve. The parameter tn is equal to n if it is >= 0.
template <int tn>
void Siever::bgj1_sieve_task(double alpha)
{
    ATOMIC_CPUCOUNT(101);

    assert(tn < 0 || static_cast<unsigned int>(tn)==n);

    // statistics are collected per-thread and only merged once at the end. These variables get merged automatically at scope exit.
    #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
        // number of successful xorpopcnt tests in the create-buckets phase
        auto &&local_stat_successful_xorpopcnt_bucketing = merge_on_exit<unsigned long long>( [this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_pass_outer(val);
            statistics.inc_stats_fullscprods_outer(val);
        } );
        // number of successful xorpopcnt tests in the second phase
        auto &&local_stat_successful_xorpopcnt_reds = merge_on_exit<unsigned long long>( [this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_pass_inner(val);
            statistics.inc_stats_fullscprods_inner(val);
        } );
    #endif
    #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
        // number of 2-reduction attempts in the second phase
        auto &&local_stat_2red_attempts = merge_on_exit<unsigned long long>([this](unsigned long long val)
        {
            statistics.inc_stats_xorpopcnt_inner(val);
            statistics.inc_stats_2reds_inner(val);
        } );
    #endif
    ENABLE_IF_STATS_REDSUCCESS ( auto &&local_stat_successful_reductions = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_2redsuccess_inner(val); }); )
    ENABLE_IF_STATS_REDSUCCESS ( auto &&local_stat_successful_red_outer  = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_2redsuccess_outer(val); }); )
    ENABLE_IF_STATS_BUCKETS(     auto &&local_stats_number_of_buckets    = merge_on_exit<unsigned long>([this](unsigned long val){ statistics.inc_stats_buckets(val); }); )

    size_t const fullS = cdb.size();
    std::vector<CompressedEntry> bucket; // bucket that this thread uses.

    std::vector<Entry> transaction_db; // stores pending transactions, i.e. points to be added to the DB.
    transaction_db.reserve(params.bgj1_transaction_bulk_size);

    for (; GBL_remaining_trial > 0; GBL_remaining_trial.fetch_sub(1))
    {
        ENABLE_IF_STATS_BUCKETS(++local_stats_number_of_buckets;)
        ///////////// preparing a bucket
        bucket.clear();

        CompressedEntry* const fast_cdb = GBL_start_of_cdb_ptr; // atomic load
        size_t const j_aux = fullS/4 + (rng() % (fullS/4));
        CompressedEntry const aux = fast_cdb[j_aux];
        std::array<LFT,MAX_SIEVING_DIM> const yr1 = db[aux.i].yr;
        CompressedVector const cv = sim_hashes.compress(yr1);

        LFT const alpha_square_times_len = alpha * alpha * aux.len;
        LFT maxlen = GBL_max_len;

        for (size_t j = 0; j < fullS; ++j)
        {
            if (UNLIKELY(is_reducible_maybe<XPC_BUCKET_THRESHOLD>(cv, fast_cdb[j].c)))
            {
                #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
                    ++local_stat_successful_xorpopcnt_bucketing; //adds to successful xorpopcnt and to #scalar product computations
                #endif
                if (UNLIKELY(j == j_aux))
                {
                    statistics.dec_stats_fullscprods_outer();
                    continue;
                }
                LFT const inner = std::inner_product(yr1.begin(), yr1.begin()+(tn < 0 ? n : tn), db[fast_cdb[j].i].yr.begin(),  static_cast<LFT>(0.));

                // Test for reduction while bucketing.
                LFT const new_l = aux.len + fast_cdb[j].len - 2 * std::abs(inner);
                if (UNLIKELY(new_l < maxlen))
                {
                    #if (COLLECT_STATISTICS_OTFLIFTS >= 2) || (COLLECT_STATISTICS_REDSUCCESS >= 2) || (COLLECT_STATISTICS_DATARACES >= 2)
                    if (bgj1_reduce_with_delayed_replace<tn>(aux, fast_cdb[j], maxlen, transaction_db, nullptr, true))
                    {
                        ENABLE_IF_STATS_REDSUCCESS ( ++ local_stat_successful_red_outer; )
                    }
                    #else
                    bgj1_reduce_with_delayed_replace<tn>(aux, fast_cdb[j], maxlen, transaction_db);
                    #endif
                    if (UNLIKELY(transaction_db.size() > params.bgj1_transaction_bulk_size))
                    {
                        if (bgj1_execute_delayed_replace(transaction_db, false))
                            maxlen = GBL_max_len;
                    }
                }

                // Test for bucketing
                if (UNLIKELY(inner * inner > alpha_square_times_len * fast_cdb[j].len))
                {
                    bucket.push_back(fast_cdb[j]);
                }
            }
        }
        // no-ops if statistics are not actually collected
        statistics.inc_stats_xorpopcnt_outer(fullS);
        statistics.inc_stats_2reds_outer(fullS -1);
        statistics.inc_stats_filter_pass(bucket.size());

        // bucket now contains points that are close to aux, and so are hopefully close to each other (up to sign)

        if (GBL_remaining_trial.load(std::memory_order_relaxed) < 0) break;
        size_t const S = bucket.size();

        ////////////// Sieve the bucket
        maxlen = GBL_max_len;
        ATOMIC_CPUCOUNT(102);
        CompressedEntry const * const fast_bucket = bucket.data();
        for (size_t block = 0; block < S; block+=CACHE_BLOCK)
        {
            for (size_t i = block+1; i < S; ++i)
            {
                // skip it if not up to date
                if (UNLIKELY(fast_bucket[i].c[0] != db[fast_bucket[i].i].c[0]))
                {
                    continue;
                }

                size_t const jmin = block;
                size_t const jmax = std::min(i, block+CACHE_BLOCK);
                CompressedEntry const * const pce1 = &fast_bucket[i];
                uint64_t const * const cv = &(pce1->c.front());
                for (size_t j = jmin; j < jmax; ++j)
                {
                    if (UNLIKELY(is_reducible_maybe<XPC_THRESHOLD>(cv, &fast_bucket[j].c.front())))
                    {
                        #if COLLECT_STATISTICS_XORPOPCNT_PASS || COLLECT_STATISTICS_FULLSCPRODS
                        ++local_stat_successful_xorpopcnt_reds; // adds to successful xorpopcnt and also to scalar product computations (done inside bgj1_reduce_with_delayed_replace)
                        #endif
                        ATOMIC_CPUCOUNT(103);
                        bool const red = bgj1_reduce_with_delayed_replace<tn>(*pce1, fast_bucket[j], maxlen, transaction_db);
                        ENABLE_IF_STATS_REDSUCCESS(if(red) {++local_stat_successful_reductions; } )
                        if (UNLIKELY(red) && transaction_db.size() > params.bgj1_transaction_bulk_size)
                        {
                            if (bgj1_execute_delayed_replace(transaction_db, false))
                            {
                                if (GBL_remaining_trial.load(std::memory_order_relaxed) < 0)
                                {
                                    #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
                                    local_stat_2red_attempts += (1 + j - jmin); // statistics for half-finished loop
                                    #endif
                                    return;
                                }
                                maxlen = GBL_max_len;
                            }
                        }
                    }
                }
                #if COLLECT_STATISTICS_XORPOPCNT || COLLECT_STATISTICS_REDS
                local_stat_2red_attempts += jmax - jmin;
                #endif
            }
        }
        bgj1_execute_delayed_replace(transaction_db, false);
    }
    bgj1_execute_delayed_replace(transaction_db, true, true);
}

#if (COLLECT_STATISTICS_OTFLIFTS >= 2) || (COLLECT_STATISTICS_REDSUCCESS >= 2) || (COLLECT_STATISTICS_DATARACES >= 2)
template<int tn>
inline bool Siever::bgj1_reduce_already_computed(LFT const inner, LFT const lenbound, LFT const e1len, LFT const e2len, Entry const & e1, Entry const & e2, std::vector<Entry>& transaction_db, bool const reduce_while_bucketing) {
#else
  template<int tn>
    inline bool Siever::bgj1_reduce_already_computed(LFT const inner, LFT const lenbound, LFT const e1len, LFT const e2len, Entry const& e1, Entry const& e2, std::vector<Entry>& transaction_db) {
  constexpr bool reduce_while_bucketing = false; // The actual value does not matter.
#endif
  LFT const new_l = e1len + e2len - 2 * std::abs(inner);    
  int const sign = inner < 0 ? 1 : -1;
  
    if (UNLIKELY(new_l < lenbound))
    {
        UidType new_uid = e1.uid;
        if(inner < 0)
        {
            new_uid += e2.uid;
        }
        else
        {
            new_uid -= e2.uid;
        }

        // If the vector belongs to another process, don't add it to our db and instead
        // forward it on.        
        if(mpi.is_active() && !mpi.owns_uid(new_uid)) {
          std::array<ZT, MAX_SIEVING_DIM> new_x = e1.x;
          addsub_vec(new_x, e2.x, static_cast<ZT>(sign));
          Entry new_entry;
          new_entry.x = new_x;
          recompute_data_for_entry<Recompute::recompute_uid>(new_entry);
          if(UNLIKELY(new_entry.uid != new_uid)) {
            // This can happen if the various points were out-of-sync
            if(reduce_while_bucketing) statistics.inc_stats_dataraces_2outer();
            else statistics.inc_stats_dataraces_2inner();
            uid_hash_table.erase_uid(new_uid);
            return false;
          }

          // Delegate to MPI to forward it.
          mpi.add_to_outgoing(new_entry, thread_pool::batch, this->n);
          return true;
          
        } else if(uid_hash_table.insert_uid(new_uid)) {
            std::array<ZT,MAX_SIEVING_DIM> new_x = e1.x;
            addsub_vec(new_x,  e2.x, static_cast<ZT>(sign));

            
            
            transaction_db.emplace_back();
            Entry& new_entry = transaction_db.back();
            new_entry.x = new_x;
            recompute_data_for_entry<Recompute::recompute_all_and_consider_otf_lift>(new_entry); // includes recomputing uid !
            if (UNLIKELY(new_entry.uid != new_uid)) // this may happen only due to data races, if the uids and the x-coos of the points we used were out-of-sync
            {
              if (reduce_while_bucketing) 
                    statistics.inc_stats_dataraces_2outer();
              else
                    statistics.inc_stats_dataraces_2inner();
                uid_hash_table.erase_uid(new_uid);
                transaction_db.pop_back();
                return false;
            }
            return true;
        }
        else
        {
          if (reduce_while_bucketing) 
              statistics.inc_stats_collisions_2outer();
          else 
                statistics.inc_stats_collisions_2inner();
            return false;
        }
    }
    else if (params.otf_lift && (new_l < params.lift_radius))
    {
        ZT x[r];
        LFT otf_helper[OTF_LIFT_HELPER_DIM];
        std::fill(x, x+l, 0);
        std::copy(e1.x.cbegin(), e1.x.cbegin()+(tn < 0 ? n : tn), x+l);
        std::copy(e1.otf_helper.cbegin(), e1.otf_helper.cbegin()+OTF_LIFT_HELPER_DIM, otf_helper);
        if(sign == 1)
        {
          for(unsigned int i=0; i < (tn < 0 ? n : tn); ++i)
          {
            x[l+i] += e2.x[i];
          }
          for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
          {
            otf_helper[i] += e2.otf_helper[i];
          }

        }
        else
        {
          for(unsigned int i=0; i < (tn < 0 ? n : tn); ++i)
          {
            x[l+i] -= e2.x[i];
          }
          for(unsigned int i=0; i < OTF_LIFT_HELPER_DIM; ++i)
          {
            otf_helper[i] -= e2.otf_helper[i];
          }
        }
        if (reduce_while_bucketing)
            statistics.inc_stats_otflifts_2outer();
        else
            statistics.inc_stats_otflifts_2inner();
        lift_and_compare(x, new_l * gh, otf_helper);
//        lift_and_compare(db[ce1.i], sign, &(db[ce2.i]));
    }
    return false;
}


// Attempt reduction between ce1 and c2. lenbound is a bound on the lenght of the result.
// If successful, we store the result in transaction_db.
// Templated by tn=n to hardwire the dimension.
// reduce_while_bucketing indicates whether we call this function during the bucketing phase (rather than when working inside a bucket)
// This is only relevant for statistics collection.
#if (COLLECT_STATISTICS_OTFLIFTS >= 2) || (COLLECT_STATISTICS_REDSUCCESS >= 2) || (COLLECT_STATISTICS_DATARACES >= 2)
template <int tn>
inline bool Siever::bgj1_reduce_with_delayed_replace(CompressedEntry const &ce1, CompressedEntry const &ce2, LFT const lenbound, std::vector<Entry>& transaction_db,
                                                     const Entry* db,
                                                     bool const reduce_while_bucketing)
{
#else
template <int tn>
  inline bool Siever::bgj1_reduce_with_delayed_replace(CompressedEntry const &ce1, CompressedEntry const &ce2, LFT const lenbound, std::vector<Entry>& transaction_db, const Entry* db)
{
    [[maybe_unused]] constexpr bool reduce_while_bucketing = false; // The actual value does not matter.
#endif

    // This is a little bit of a hack to allow bgj1_reduce_with_delayed_replace to be used for distributed sieving.
    // Essentially, because each received bucket is stored in its own memory, it makes no sense for the inner product
    // check below to look in the global siever database for reductions. This simply allows us to pass in a particular sieving
    // database to this function for reduction. 
    if(!db) {
      db = this->db.data();
    }
    
    // statistics.inc_fullscprods done at call site
    LFT const inner = std::inner_product(db[ce1.i].yr.cbegin(), db[ce1.i].yr.cbegin()+(tn < 0 ? n : tn), db[ce2.i].yr.cbegin(), static_cast<LFT>(0.));
    #if (COLLECT_STATISTICS_OTFLIFTS >= 2) || (COLLECT_STATISTICS_REDSUCCESS >= 2) || (COLLECT_STATISTICS_DATARACES >= 2)
    return bgj1_reduce_already_computed<tn>(inner, lenbound, ce1.len, ce2.len, db[ce1.i], db[ce2.i], transaction_db, reduce_while_bucketing);
    #else
    return bgj1_reduce_already_computed<tn>(inner, lenbound, ce1.len, ce2.len, db[ce1.i], db[ce2.i], transaction_db);
    #endif
}


bool Siever::bgj1_execute_delayed_replace(std::vector<Entry>& transaction_db, bool force, bool nosort /* =false*/)
{
    if (UNLIKELY(!transaction_db.size())) return true;
    ATOMIC_CPUCOUNT(104);
    
    std::unique_lock<std::mutex> lockguard(GBL_db_mutex, std::defer_lock_t());
    size_t resortthres = params.bgj1_resort_ratio * stop_pos;
    size_t maxts = resortthres;

    
    if (UNLIKELY(nosort == true))
    {
        // compute maximum transaction count that can be processed without triggering resort
        maxts = GBL_replace_pos.load();
        // if sort is current happening then return false if force = true or else wait
        while (maxts < resortthres)
        {
          if(!force) return false;
          lockguard.lock(); lockguard.unlock();
          maxts = GBL_replace_pos.load();
        }
        maxts = (maxts - resortthres) / params.threads;
    }
    
    // maximum size that can be safely processed without out-of-bounds errors
    if (UNLIKELY(transaction_db.size() > maxts))
    {
        std::sort(transaction_db.begin(), transaction_db.end(), [](const Entry& l, const Entry& r){return l.len < r.len;});
        transaction_db.resize(maxts);
    }
    size_t ts = transaction_db.size();
    assert(ts <= maxts);

    size_t rpos = 0;
    LFT oldmaxlen = 0;
    // try to claim a replacement interval in cdb: rpos-ts+1,...,rpos
    while (true)
    {
        LFT maxlen = GBL_max_len.load();
        // prune transaction_db based on current maxlen
        if (LIKELY(maxlen != oldmaxlen))
        {
            for (size_t i = 0; i < transaction_db.size(); )
            {
                if (transaction_db[i].len < maxlen)
                    ++i;
                else
                {
                    if (i != transaction_db.size()-1)
                        std::swap(transaction_db[i], transaction_db.back());
                    transaction_db.pop_back();
                }
            }
            oldmaxlen = maxlen;
            ts = transaction_db.size();
        }
        // check & wait if another thread is in the process of sorting cdb
        while (UNLIKELY((rpos = GBL_replace_pos.load()) < resortthres))
        {
            if (!force) return (ts==0); // Note: ts might have become 0 due to pruning. In this case, we return true
            lockguard.lock(); lockguard.unlock();
        }
        // decrease GBL_replace_pos with #ts if it matches the expected value rpos, otherwise retry
        if (LIKELY(GBL_replace_pos.compare_exchange_weak(rpos, rpos-ts)))
            break;
    }

    // now we can replace cdb[rpos-ts+1,...,rpos]
    // if we need to resort after inserting then already grab the lock so other threads can block
    if (UNLIKELY( rpos-ts < resortthres ))
    {
        lockguard.lock();
        // set GBL_remaining_trial to MAXINT, but if it was negative then keep it negative
        if (UNLIKELY(GBL_remaining_trial.exchange( std::numeric_limits<decltype(GBL_remaining_trial.load())>::max()) < 0))
            GBL_remaining_trial = -1;
    }

    // update GBL_max_len already
    GBL_max_len = cdb[params.bgj1_improvement_db_ratio * (rpos-ts)].len / REDUCE_LEN_MARGIN;

    // we can replace cdb[rpos-ts+1,...,rpos]
    size_t cur_sat = 0;
    for (size_t i = 0; !transaction_db.empty(); ++i)
    {
        if (bgj1_replace_in_db(rpos-i, transaction_db.back())
            && transaction_db.back().len < params.saturation_radius)
        {
            ++cur_sat;
        }
        transaction_db.pop_back();
    }

    if(mpi.is_active()) {
      mpi.set_sat_drop(unsigned(cur_sat));
    }
    
    statistics.inc_stats_replacements_list(cur_sat);
    // update GBL_saturation_count
    if (UNLIKELY(GBL_saturation_count.fetch_sub(cur_sat) <= cur_sat))
    {
        GBL_saturation_count = 0;
        GBL_remaining_trial = -1;
    }

    // update GBL_replaced_pos to indicate we're done writing to cdb
    GBL_replace_done_pos -= ts;
    // if we don't need to resort then we're done
    if (LIKELY(rpos-ts >= resortthres))
        return true;
    // wait till all other threads finished writing to db/cdb
    // TODO: a smarter way to sleep so it gets activated when other threads have finished
    while (GBL_replace_done_pos.load() != GBL_replace_pos.load()) // Note : The value of GBL_replace_pos actually cannot change here, so the order of the reads does not matter.
        std::this_thread::yield();

    // sorting always needs to happen, threads could be waiting on GBL_replace_pos
    CPUCOUNT(105);
    size_t improvindex = params.bgj1_improvement_db_ratio * (stop_pos-1);

    if (params.threads == 1)
    {
        const auto end_iter = cdb.begin() + stop_pos;
        std::sort(cdb.begin()+GBL_replace_pos+1, end_iter, compare_CE());
        std::inplace_merge(cdb.begin(), cdb.begin()+GBL_replace_pos+1, end_iter, compare_CE());
    }
    else
    {
        cdb_tmp_copy=cdb;
        const auto end_iter = cdb_tmp_copy.begin() + stop_pos;
        std::sort(cdb_tmp_copy.begin()+GBL_replace_pos+1, end_iter, compare_CE());
        std::inplace_merge(cdb_tmp_copy.begin(), cdb_tmp_copy.begin()+GBL_replace_pos+1, end_iter, compare_CE());
        cdb.swap(cdb_tmp_copy);
        GBL_start_of_cdb_ptr = &(cdb.front());
    }
    statistics.inc_stats_sorting_sieve();
    // reset GBL_remaining_trial to GBL_max_trial, unless it was set to < 0 in the meantime.
    if (UNLIKELY(GBL_remaining_trial.exchange(GBL_max_trial) < 0))
       GBL_remaining_trial = -1;
    
    GBL_replace_done_pos = stop_pos - 1;
    GBL_replace_pos = stop_pos - 1;
    GBL_max_len = cdb[improvindex].len / REDUCE_LEN_MARGIN;
    return true;
}

// Replace the db and cdb entry pointed at by cdb[cdb_index] by e, unless
// length is actually worse
bool Siever::bgj1_replace_in_db(size_t cdb_index, Entry &e)
{
    ATOMIC_CPUCOUNT(106);
    CompressedEntry &ce = cdb[cdb_index]; // abbreviation

    if (REDUCE_LEN_MARGIN_HALF * e.len >= ce.len)
    {
        statistics.inc_stats_replacementfailures_list();
        uid_hash_table.erase_uid(e.uid);
        return false;
    }
    uid_hash_table.erase_uid(db[ce.i].uid);
    if(ce.len < params.saturation_radius) { statistics.inc_stats_dataraces_replaced_was_saturated(); } // saturation count becomes (very) wrong if this happens (often)
    ce.len = e.len;
    ce.c = e.c;
    // TODO: FIX THIS PROPERLY. We have a race condition here.
    // NOTE: we tried std::move here (on single threaded), but it ended up slower...
    db[ce.i] = e;
    return true;
}

// clang-format on
