#ifndef INCLUDED_UID_HASH_TABLE_HPP
#define INCLUDED_UID_HASH_TABLE_HPP

#include "compat.hpp"
#include "constants.hpp"
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <vector>

class Siever;

/**
    UidHashTable class

    The class UidHashTable encapsulates the hash table that we use to detect
   collisions. Each point that we create has a hash, called uid, of type UidType
   and we ensure that the uids of all point in existance at a given point of
   time are distinct.

    The UidHashTable class stores the both description of the hash function and
   the hash tables and contains functions to compute hashes and insert / remove
   them from the database.

    For implementation reasons, our hashes are linear and the hash table takes
   care about +/- symmetry. 0 is always a member of the hash table that cannot
   be removed.

    Usage notes:

    Before any usage, reset_hash_function(siever) has to be run, which sets up a
   random hash function. The hash function assumes that the input has the
   correct size, which is set during reset_hash_function(siever). In particular,
   the hash function has to be reset after a change of dimension.

    To compute the uid / hash of a given Entry e, use compute_uid(e.x). The hash
   function operates on the x-coos of Entries. It is possible to insert such
   hashes, checks whether a hash is already present and erase hashes. All
   operations are thread-safe unless explicitly said otherwise.

    uid received by compute_uid should not be used for anything other than for
   calls into the uid database or to make use of linearity of compute_uid. We
   guarantee that the hash function is invariant under negation in the following
   sense: After performing uid = compute_uid(x), uid' = compute_uid(-x) and a
   successful hash_table.insert(uid), hash_table.check(uid') will return true. [
   Note: To preserve linearity, we cannot have uid == uid'.]
*/

class UidHashTable {
public:
  // creates a dummy hash table. Note that reset_hash_function has to be called
  // at least once before it can be used.
  explicit UidHashTable() : db_uid(), n(0), uid_coeffs() { insert_uid(0); }
  // resets the hash_function used. Siever is changed, because it uses it as a
  // randomness source. This also clears the database. NOT THREAD-SAFE
  // (including the randomness calls).
  inline void reset_hash_function(Siever &siever);

  // Compute the uid of x using the current hash function.
  inline UidType compute_uid(std::array<ZT, MAX_SIEVING_DIM> const &x) const;

  inline bool check_uid_unsafe(
      UidType uid); // checks whether uid is present without locks. Unsafe if
                    // other threads are writing to the hash table.
  inline bool
  check_uid(UidType uid); // checks whether uid is present in the table. Avoid
                          // in multi-threaded contexts.
  inline bool
  insert_uid(UidType uid); // inserts uid into the hash table. Return value
                           // indicates success (i.e. returns false if uid was
                           // present beforehand)
  inline bool
  erase_uid(UidType uid); // removes uid from the hash table. return value
                          // indicates success (i.e. if it was present at all)
  inline size_t hash_table_size(); // returns the number of stored hashes
                                   // (excluding 0). NOT THREAD-SAFE

  inline size_t
  hash_table_size_safe(); // returns the number of stored hashes
                          // (excluding 0). Thread-safe, but can only be
                          // used safely if no other threads are running.
  inline bool
  erase_uid_unsafe(UidType uid); /// removes the uid from the hash table if it
                                 /// was present at all. NOT THREAD-SAFE.

  // If possible, atomically {removes removed_uid and inserts new_uid}:
  // If both new_uid is not yet present and removed_uid is present, performs the
  // change, otherwise does nothing. Return indicates success. If removed_uid ==
  // new_uid, returns false [This should only happen with false positive
  // collisions and that way is simpler to implement]
  inline bool replace_uid(UidType removed_uid, UidType new_uid);

  inline std::vector<UidCoeffType> &get_uid_coeffs();

  inline void
  clear_hash_table(Siever &siever); // Empties the hash table without resetting
                                    // the hash function used. This inserts 0.

  static inline unsigned get_slot(UidType uid) {
    // Returns the slot in the hash table for "uid".  This
    // is used in distributed sieving.
    normalize_uid(uid);
    return uid % DB_UID_SPLIT;
  }

  inline void print_state(std::ostream &os = std::cerr);

private:
  // Note : Implementation is subject to possible changes
  // obtain the sub-table for a given uid.
  // Note that since uid is linear, we have to somehow deal with +/- invariance
  // on the inserting / erasure / checking layer. We can either insert both uid
  // and -uid into the database or we preprocess every incoming uid by a +/-
  // invariant function. Since we do not want to lock twice as many mutexes
  // (with a deadlock-avoiding algorithm!), we opt to preprocess all incoming
  // uids rather than insert both +uid and -uid, because we need some form of
  // preprocessing for the sub-table selection anyway. Furthermore, in
  // multi-threaded contexts, we do not use check anyway (but try to insert and
  // insert and handle the case where this fails), so optimizing for checks is
  // pointless.
  static void normalize_uid(UidType &uid) {
    static_assert(std::is_unsigned<UidType>::value, "");
    if (uid > std::numeric_limits<UidType>::max() / 2 + 1) {
      uid = -uid;
    }
  }

  // Note: The splitting is purely to allow better parallelism.
  struct padded_map : std::unordered_set<UidType> {
    cacheline_padding_t pad;
  };
  std::array<padded_map, DB_UID_SPLIT>
      db_uid; // Sets of the uids of all vectors of the database. db_uid[i]
              // contains only vectors with (normalized) uid % DB_UID_SPLIT ==
              // i.
  struct padded_mutex : std::mutex {
    cacheline_padding_t pad;
  };
  std::array<padded_mutex, DB_UID_SPLIT>
      db_mut; // array of mutexes. db_mut[i] protects all access to db_uid[i].
              // unsigned int n; // dimension of points in the domain of the
              // hash function.
  unsigned int n; // dimension of points in the domain of the hash function.
                  // The hash function only works on vectors of this size
  std::vector<UidCoeffType> uid_coeffs; // description of the hash function.
};

#endif
