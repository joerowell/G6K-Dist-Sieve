#ifndef INCLUDED_SHUFFLESIEVER_HPP
#define INCLUDED_SHUFFLESIEVER_HPP

#ifdef G6K_MPI
#include "siever.h"     // Needed for sieving!
#include "mpi.h"        // Needed for most of MPI.
#include "mpi_obj.hpp"  // Needed for other MPI things.
#include "topology.hpp" // Needed for inheritance hierarchy.

/**
   ShuffleSiever. This class realises the necklace sieving algorithm described
in:

   Quantum Algorithms for the Approximate k-List Problem and Their Application
to Lattice Sieving,
   Elena Kirshanova, Erik Mårtensson, Eamonn W. Postlethwaite, and Subhayan Roy
Moulik, Asiacrypt 2019, https://eprint.iacr.org/2019/1016, Algorithm 7.1, and
(in spirit) the Ducas-Kirchner ring sieve described in,
   https://eurocrypt.iacr.org/2018/Slides/Monday/TrackB/01-01.pdf, slide 22.

   Essentially, the idea is to shuffle vectors around a ring, at each step
applying some reduction. There are some minor differences between the strategies
(i.e whether the vectors are only shifted in one direction, or if they flow back
and forth between two nodes).

   It turns out that this sieving strategy maps quite nicely to distributed
hardware: the "flow" allows one to use non-blocking calls, which allows hardware
to be theoretically fully saturated.
**/
class ShuffleSiever : public Topology {
public:
  /**
     ShuffleSiever. This the default constructor.
     @param[in] comm: the MPI communicator to use.
  **/
  ShuffleSiever(MPI_Comm &comm);

  /**
     receive_entries. This function receives the entries from the `curr` node
     and stores the results in `entries`. This function may resize `entries` to
     ensure that enough storage is present. This function does not throw (as
  Entry is nothrow constructible). This particular instantiation relies upon a
  call to MPI_Shift.
     @param[in] entries: the vector of entries to overwrite.
     @param[in] l: the (inclusive) lower coefficient of each entry.
     @param[in] r: the (exclusive) upper coefficient of each entry.
     @remarks The received entries are _not_ initialised. You will need to
  compute the values directly in the caller.
  **/
  void receive_entries(std::vector<Entry> &entries, const unsigned l,
                       const unsigned r) noexcept;

  /**
     send_entries. This function sends the `entries` to the `curr` node. This
     function does not throw.

     This particular function pre-filters the vectors that it sends against the
  hash table provided by each node. This acts as (essentially) a Bloom filter
  that allows an approximate membership test. If the vector is present in the
  recipient's hash table, then it is not sent to the `curr` node.

     @param[in] entries: the vector of entries to overwrite.
     @param[in] l: the (inclusive) lower coefficient of each entry.
     @param[in] r: the (exclusive) upper coefficient of each entry.
  **/
  void send_entries(const std::vector<Entry> &entries, const unsigned l,
                    const unsigned r) noexcept;

  void send_entries_via_cdb(const std::vector<CompressedEntry> &cdb,
                            const std::vector<Entry> &db, const unsigned l,
                            const unsigned r) noexcept;

  void receive_entries_via_cdb(std::vector<CompressedEntry> &cdb,
                               std::vector<Entry> &db, const unsigned l,
                               const unsigned r);

  /**
     setup. This function sets up shared state across all nodes in the
   hierarchy. This function does not throw.

     The goal here is to establish shared state across all nodes. At present,
   this function simply copies the state from the root node and overwrites the
   state in all other attached nodes.

     @remarks Note that the caller of this function _must_ update their
   databases to make sure everything matches correctly.
   @param[in] is_g6k: true if this node is the root node, false otherwise.
   **/
  void setup(const bool is_g6k) noexcept;

  /**
     get_type. This function returns the type of this topology as a
     DistSieverType. This function only ever returns
  DistSieverType::ShuffleSiever. This function does not throw or modify this
  object.
     @return the type of the topology.
  **/
  DistSieverType get_type() const noexcept;

private:
  /**
     right. This is the rank of the node "to the right" of us.
  **/
  int right;

  /**
     left. This is the rank of the node "to the left" of us.
  **/
  int left;

  /**
     left_uid_table. This is the hash table of the node on the "left" of us.
     This is used to prevent unnecessary sends: it acts like a distributed Bloom
   filter.
   **/
  UidHashTable left_uid_table;

  /**
     right_uid_table. This is the hash table of the node on the "right" of us.
     This is used to prevent unnecessary sends: it acts like a distributed Bloom
   filter.
   **/
  UidHashTable right_uid_table;

  /**
     comm. A copy of the comm held by the owner of this object.
  **/
  MPI_Comm &comm;
};

#endif
#endif
