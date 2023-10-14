#ifndef INCLUDED_SIEVERTYPES_HPP
#define INCLUDED_SIEVERTYPES_HPP

/**
   DistSieverType. This is the enum for the type of sievers that are available
to be used. See the respective .hpp files for a more complete description.
**/
enum class DistSieverType : unsigned {
  Null = 0,
  ShuffleSiever = 1,
};

#endif
