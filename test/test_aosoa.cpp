#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include "ppMemUsage.hpp"
#include <Omega_h_mesh.hpp>
#include <particle_structs.hpp>
#include "pumipic_adjacency.hpp"
#include "pumipic_mesh.hpp"
#include "pumipic_ptcl_ops.hpp"
#include "pumipic_profiling.hpp"
#include "pseudoXGCmTypes.hpp"

using DataTypes = Cabana::MemberTypes<double[3],  // position
                                        double[3],   // computed position  
                                        int,          // id
                                        float,
                                        float,
                                        bool>;     // mask

using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace>;

void count(AoSoA_t& aosoa) {
  pumipic::kkLidView count("count", 1);
  auto mask = Cabana::slice<DataTypes::size-1>(aosoa, "mask");
  Kokkos::parallel_for("count_active", aosoa.size(), KOKKOS_LAMBDA(const int i) {
      if (mask(i)) {
          Kokkos::atomic_fetch_add(&count(0), 1);
      }
  });
  int active_count = pumipic::getLastValue(count);
  std::cout << "Number of active particles: " << active_count << std::endl;
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  
  {
    const int size = std::atoi(argv[1]);
    
    AoSoA_t aosoa("ParticleData", size);
    auto positions = Cabana::slice<0>(aosoa, "position");
    auto computed = Cabana::slice<1>(aosoa, "computed");
    auto pid = Cabana::slice<2>(aosoa, "id");
    auto ellipse_b = Cabana::slice<3>(aosoa, "ellipse_b");
    auto angle = Cabana::slice<4>(aosoa, "angle");
    auto mask = Cabana::slice<DataTypes::size-1>(aosoa, "mask");
    
    std::cout << "Initializing AoSoA with " << size << " elements..." << std::endl;
    
    // Initialize elements using parallel_for
    Kokkos::parallel_for("initialize", 
      Kokkos::RangePolicy<>(0, size),
      KOKKOS_LAMBDA(const int i) {
        positions(i, 0) = 1.0;      // x
        positions(i, 1) = 2.0;      // y  
        positions(i, 2) = 3.0;      // z
        computed(i, 0) = 1.0;  // vx
        computed(i, 1) = 2.0;  // vy
        computed(i, 2) = 3.0;  // vz
        pid(i) = i;            // id
        ellipse_b(i) = 0.5;   // ellipse b
        angle(i) = 0.25;      // angle
        mask(i) = true; // all active
      }
    );

    count(aosoa);
    std::cout << "\nModifying elements" << std::endl;
    const auto soa_len = AoSoA_t::vector_length;
    std::cout << "AoSoA vector length: " << soa_len << std::endl;

    Cabana::SimdPolicy<soa_len,Kokkos::DefaultExecutionSpace> simd_policy(0, size);
    Cabana::simd_parallel_for(simd_policy, KOKKOS_LAMBDA( const lid_t soa, const lid_t ptcl ) {
      if (mask.access(soa, ptcl)) {
        positions.access(soa, ptcl, 0) = 1.0;
        positions.access(soa, ptcl, 1) = 2.0;
        positions.access(soa, ptcl, 2) = 3.0;
      }
    });
    count(aosoa);
    // fprintf(stderr, "capacity %d size %d numSoA %d\n", aosoa.capacity(), aosoa.size(), aosoa.numSoA());
    std::cout << "capacity " << aosoa.capacity() << " size " << aosoa.size() << " numSoA " << aosoa.numSoA() << std::endl;
  }
  
  // Finalize Kokkos
  Kokkos::finalize();
  
  return 0;
}