#ifndef INCLUDED_PACKED_SYNC_STRUCT_HPP
#define INCLUDED_PACKED_SYNC_STRUCT_HPP

/**
   PackedSyncStruct. This structure contains all of the information that is
synced across the cluster when doing sieving operations. You can view this as a
header that informs other nodes if the trial count has dropped below 0 and the
number of short vectors discovered since the last synchronisation.
**/
struct SyncHeader {
  unsigned trial_count;
  unsigned sat_drop;
};

#endif
