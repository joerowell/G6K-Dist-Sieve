#ifndef INCLUDED_TOPOLOGY_HPP
#define INCLUDED_TOPOLOGY_HPP

#ifdef G6K_MPI
#include "siever_types.hpp" // Needed for the types of sievers.
#include <memory>
#include <mpi.h>
#include <vector>

// This is a forward declaration so we can include the topology directly from
// siever.h
struct Entry;
struct CompressedEntry;

/**
   Topology. This class realises a generalisation of a networking topology in
MPI. Essentially, to make life easier for the caller in G6K, we abstract away
certain MPI details behind an abstract sending and receiving interface. This
interface should be interacted with via the MPIObj object.

@remarks Note that _all_ functions in this class are pure virtual. You will need
to implement them yourselves.
**/
class Topology {
public:
  /**
     ~Topology. Virtual destructor for this class. This destructor does nothing.
   **/
  virtual ~Topology();

  /**
     build_topology. This returns a unique ptr to a
     new topology. This function never throws.
     @param[in] comm: the MPI communicator. May be overwritten by this Topology.
     @param[in] type: the type of topology to instantiate. See SieverTypes.hpp
  for more.
  @remarks The caller should take care of the fact that `comm` is potentially
  changed by this function.
  **/
  static std::unique_ptr<Topology>
  build_topology(MPI_Comm &comm, const DistSieverType type) noexcept;

  /**
     send_entries. This function sends the entries in `entries` to the next
  selected receiver node from [l,r). This function does not throw. Note that the
  exact way that this is implemented strongly depends on the type of toplogy
  that is used.
     @param[in] entries: the entries to send.
     @param[in] l: the (inclusive) lower coefficient of each entry.
     @param[in] r: the (exclusive) upper coefficient of each entry.
  **/
  virtual void send_entries(const std::vector<Entry> &entries, const unsigned l,
                            const unsigned r) noexcept = 0;

  /**
     receive_entries. This function receives the entries from the previous node
  and stores the results in `entries`. This function may resize `entries` to
  ensure that enough storage is present. This function does not throw (as Entry
  is nothrow constructible). Note that the exact way that this is implemented
  strongly depends on the type of topology that is used.
     @param[in] entries: the array of entries to overwrite.
     @param[in] l: the (inclusive) lower coefficient of each entry.
     @param[in] r: the (exclusive) upper coefficient of each entry.
     @remarks The received entries are _not_ initialised. You will need to
     compute the values of these entries directly.
  **/
  virtual void receive_entries(std::vector<Entry> &entries, const unsigned l,
                               const unsigned r) noexcept = 0;

  /**
     send_entries_via_cdb. This function sends every vector from cdb to the next
  node. It does this by walking through `db` using the entries in `cdb`. This
  function does this without copying the elements of `db`: we send them directly
  using MPI routines. This function does not throw.
     @param[in] cdb: the compressed database. Used as a `key` into the database.
     @param[in] db: the full database. Only a handful of entries will be
  touched.
     @param[in] l: the (inclusive) lower coefficient of each entry.
     @param[in] r: the (exclusive) upper coefficient of each entry.
  **/
  virtual void send_entries_via_cdb(const std::vector<CompressedEntry> &cdb,
                                    const std::vector<Entry> &db,
                                    const unsigned l,
                                    const unsigned r) noexcept = 0;

  /**
     receive_entries_via_cdb. This function receives every vector from the next
   node and stores the result in `db`, threading through `db`.  You can few this
   function as a dual of `send_entries_via_db`.

     Note that this function may throw if `cdb` does not have enough elements in
   it to account for the new batch of vectors. This is primarily because of the
   1:1 correspondence between cdb and db. In those situations, we recommend
   trying again with a larger cdb.

     @param[in] cdb: the entries to overwrite.
     @param[in] db: the database of elements. These are indexed by the cdb.
     @param[in] l: the (inclusive) lower coefficient of each entry.
     @param[in] r: the (exclusive) upper coefficient of each entry.
     @remarks The received entries are _not_ initialised. You will need to
     compute the values of these entries directly.
   **/
  virtual void receive_entries_via_cdb(std::vector<CompressedEntry> &cdb,
                                       std::vector<Entry> &db, const unsigned l,
                                       const unsigned r) = 0;

  /**
     setup. This function sets up a particular sieving instance across many
  nodes. This function does not throw.

  @param[in] is_g6k: true if this node is the root node, false otherwise.
  **/
  virtual void setup(const bool is_g6k) noexcept = 0;

  /**
     get_type. This function returns the type of this topology as a
  DistSieverType. This function does not throw or modify this object.
     @return the type of topology.
  **/
  virtual DistSieverType get_type() const noexcept = 0;
};

#endif
#endif
