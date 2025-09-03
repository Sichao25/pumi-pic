#include <Omega_h_mesh.hpp>
#include <particle_structs.hpp>
#include "pumipic_adjacency.hpp"
#include "pumipic_mesh.hpp"
#include "pumipic_ptcl_ops.hpp"
#include "pumipic_profiling.hpp"
#include "pseudoXGCmTypes.hpp"
#include "gyroScatter.hpp"
#include <fstream>
#include "ellipticalPush.hpp"
#include "psMemberTypeCabana.h"
#include <random>
#include <ppTiming.hpp>
#include "ppMemUsage.hpp"
#define ELEMENT_SEED 1024*1024
#define PARTICLE_SEED 512*512

void getMemImbalance(int hasptcls) {
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  size_t free, total;
  getMemUsage(&free, &total);
  const long used=total-free;
  long maxused=0;
  long totused=0;
  int rankswithptcls=0;
  MPI_Allreduce(&used, &maxused, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&used, &totused, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&hasptcls, &rankswithptcls, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const double avg=static_cast<double>(totused)/rankswithptcls;
  const double imb=maxused/avg;
  if(!comm_rank) {
    printf("ranks with particles %d memory usage imbalance %f\n",
        rankswithptcls, imb);
  }
  if( used == maxused ) {
    printf("%d peak mem usage %ld, avg usage %f\n", comm_rank, maxused, avg);
  }
}

void getPtclImbalance(lid_t ptclCnt) {
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  long ptcls=ptclCnt;
  long max=0;
  long tot=0;
  int hasptcls = (ptclCnt > 0);
  int rankswithptcls=0;
  MPI_Allreduce(&ptcls, &max, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&ptcls, &tot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&hasptcls, &rankswithptcls, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const double avg=static_cast<double>(tot)/rankswithptcls;
  const double imb=max/avg;
  if(!comm_rank) {
    printf("ranks with particles %d particle imbalance %f\n",
        rankswithptcls, imb);
  }
  if( ptcls == max ) {
    printf("%d peak particle count %ld, avg usage %f\n", comm_rank, max, avg);
  }
}

void render(p::Mesh& picparts, int iter, int comm_rank) {
  std::stringstream ss;
  ss << "pseudoPush_r" << comm_rank<<"_t"<<iter;
  std::string s = ss.str();
  Omega_h::vtk::write_parallel(s, picparts.mesh(), picparts.dim());
}

void printTiming(const char* name, double t) {
  fprintf(stderr, "kokkos %s (seconds) %f\n", name, t);
}

void printTimerResolution() {
  Kokkos::Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  fprintf(stderr, "kokkos timer reports 1ms as %f seconds\n", timer.seconds());
}

void setPtclIds(PS* ptcls) {
  auto pid_d = ptcls->get<2>();
  auto setIDs = PS_LAMBDA(const int& eid, const int& pid, const bool& mask) {
    pid_d(pid) = pid;
  };
  ps::parallel_for(ptcls, setIDs);
}

int setSourceElements(p::Mesh& picparts, PS::kkLidView ppe,
    const int mdlFace, const int numPtclsPerRank) {
  //Deterministically generate random number of particles on each element with classification less than mdlFace
  int comm_rank = picparts.comm()->rank();
  const auto elm_dim = picparts.dim();
  o::Mesh* mesh = picparts.mesh();
  auto face_class_ids = mesh->get_array<o::ClassId>(elm_dim, "class_id");
  auto face_owners = picparts.entOwners(elm_dim);
  o::Write<o::LO> isFaceOnClass(face_class_ids.size(), 0);
  o::parallel_for(face_class_ids.size(), OMEGA_H_LAMBDA(const int i) {
    if( face_class_ids[i] <= mdlFace && face_owners[i] == comm_rank)
      isFaceOnClass[i] = 1;
  });
  o::LO numMarked = o::get_sum(o::LOs(isFaceOnClass));
  if(!numMarked)
    return 0;

  int nppe = numPtclsPerRank / numMarked;
  o::HostWrite<o::LO> rand_per_elem(mesh->nelems());

  //Gaussian Random generator with mean = number of particles per element
  std::default_random_engine generator(ELEMENT_SEED);
  std::normal_distribution<double> dist(nppe, nppe / 4);

  Omega_h::HostWrite<o::LO> isFaceOnClass_host(isFaceOnClass);
  int total = 0;
  int last = -1;
  for (int i = 0; i < mesh->nelems(); ++i) {
    rand_per_elem[i] = 0;
    if (isFaceOnClass_host[i] && total < numPtclsPerRank ) {
      last = i;
      rand_per_elem[i] = Kokkos::round(dist(generator));
      if (rand_per_elem[i] < 0)
        rand_per_elem[i] = 0;
      total += rand_per_elem[i];
      //Stop if we hit the number of particles
      if (total > numPtclsPerRank) {
        int over = total - numPtclsPerRank;
        rand_per_elem[i] -= over;
      }
    }
  }
  //If we didn't put all particles in, fill them in the last element we touched
  if (total < numPtclsPerRank) {
    int under = numPtclsPerRank - total;
    rand_per_elem[last] += under;
  }
  o::Write<o::LO> ppe_write(rand_per_elem);


  int np = o::get_sum(o::LOs(ppe_write));
  o::parallel_for(mesh->nelems(), OMEGA_H_LAMBDA(const o::LO& i) {
    ppe(i) = ppe_write[i];
  });
  return np;
}

void setInitialPtclCoords(p::Mesh& picparts, PS* ptcls, bool output) {
  pumipic::kkLidView count("count", 1);
  //set particle positions and parent element ids
  auto x_ps_d = ptcls->get<0>();
  const auto vector_length = 32;
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask) {
      x_ps_d(pid,0) = 1.7;
      x_ps_d(pid,1) = 0.05;
      x_ps_d(pid,2) = 0;
      Kokkos::atomic_fetch_add(&count(0), 1);
      if (output)
        printf("pid %d: %.3f %.3f %.3f\n", pid, x_ps_d(pid,0), x_ps_d(pid,1), x_ps_d(pid,2));
    }
  };
  ps::parallel_for(ptcls, lamb);
  lid_t num_init = pumipic::getLastValue(count);
  fprintf(stderr, "[INFO] setInitialPtclCoords initialized %d particles\n", num_init);
}

o::Mesh readMesh(char* meshFile, o::Library& lib) {
  const auto rank = lib.world()->rank();
  (void)lib;
  std::string fn(meshFile);
  auto ext = fn.substr(fn.find_last_of(".") + 1);
  if( ext == "msh") {
    if(!rank)
      std::cout << "reading gmsh mesh " << meshFile << "\n";
    return Omega_h::gmsh::read(meshFile, lib.self());
  } else if( ext == "osh" ) {
    if(!rank)
      std::cout << "reading omegah mesh " << meshFile << "\n";
    return Omega_h::binary::read(meshFile, lib.self(), true);
  } else {
    if(!rank)
      std::cout << "error: unrecognized mesh extension \'" << ext << "\'\n";
    exit(EXIT_FAILURE);
  }
}

void countPIDs(PS* structure, int identifier) {
  pumipic::kkLidView mask_count("count", 1);
  auto pid_d = structure->get<2>();
  // count atomic 1 if pid is new
  auto countPIDs = PS_LAMBDA(const lid_t& e, const lid_t& pid, const bool& mask) {
    if (mask) {
      Kokkos::atomic_increment<lid_t>(&(mask_count[0]));
    }
  };
  ps::parallel_for(structure, countPIDs, "countPIDs");
  lid_t mcnt = pumipic::getLastValue(mask_count);
  fprintf(stderr, "[DEBUG = %d ] counted pids %d \n", identifier, mcnt);
}

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  const int numargs = 7;
  if( argc != numargs ) {
    printf("numargs %d expected %d\n", argc, numargs);
    auto args = "<ppm mesh> <numPtcls> "
      "<max initial model face> <maxIterations> "
      "<degrees per elliptical push>"
      "<enable prebarrier>";
    std::cout << "Usage: " << argv[0] << args << "\n";
    exit(1);
  }

  getMemImbalance(1);
  if (comm_rank == 0) {
    printf("world ranks %d\n", comm_size);
    printf("particle_structs floating point value size (bits): %zu\n", sizeof(fp_t));
    printf("omega_h floating point value size (bits): %zu\n", sizeof(Omega_h::Real));
    printf("Kokkos execution space memory %s name %s\n",
           typeid (Kokkos::DefaultExecutionSpace::memory_space).name(),
           typeid (Kokkos::DefaultExecutionSpace).name());
    printf("Kokkos host execution space %s name %s\n",
           typeid (Kokkos::DefaultHostExecutionSpace::memory_space).name(),
           typeid (Kokkos::DefaultHostExecutionSpace).name());
    printTimerResolution();
  }

  pumipic::SetTimingVerbosity(0);
  if (comm_rank == comm_size / 2) {
    pumipic::EnableTiming();
  }

  p::Mesh picparts;
  pumipic::read(&lib, lib.world(), argv[1], &picparts);
  o::Mesh* mesh = picparts.mesh();
  mesh->ask_elem_verts(); //caching adjacency info

  int nBuffers = picparts.numBuffers(picparts.dim());
  int* buffered_ranks = new int[nBuffers];
  auto buffers = picparts.bufferedRanks(picparts.dim());
  buffered_ranks[0] = comm_rank;
  for (int i = 0; i < nBuffers - 1; ++i)
    buffered_ranks[i+1] = buffers[i];
  p::Distributor<> dist(nBuffers, buffered_ranks);
  delete [] buffered_ranks;

  /* Particle data */
  const long int numPtcls = atol(argv[2]);
  const int numPtclsPerRank = numPtcls / comm_size;
  const bool output = numPtclsPerRank <= 30;

  long int totNumReqPtcls = 0;
  const long int numPtclsPerRank_li = numPtclsPerRank;
  MPI_Allreduce(&numPtclsPerRank_li, &totNumReqPtcls, 1, MPI_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  fprintf(stderr, "particles requested %ld %ld\n", numPtcls, totNumReqPtcls);

  Omega_h::Int ne = mesh->nelems();

  {
    PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
    PS::kkGidView element_gids("element_gids", ne);
    Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
    Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
        element_gids(i) = mesh_element_gids[i];
      });
    const int mdlFace = atoi(argv[3]);
    int actualParticles = setSourceElements(picparts,ptcls_per_elem,mdlFace,numPtclsPerRank);
    Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
        const int np = ptcls_per_elem(i);
        if (output && np > 0)
          printf("ppe[%d] %d\n", i, np);
      });

    long int totNumPtcls = 0;
    long int actualParticles_li = actualParticles;
    MPI_Allreduce(&actualParticles_li, &totNumPtcls, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (!comm_rank)
      fprintf(stderr, "particles created %ld\n", totNumPtcls);

    Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy = pumipic::TeamPolicyAuto(10000, 32);

    ps::CabM_Input<Particle> cabm_input(policy, ne, actualParticles,
                                      ptcls_per_elem, element_gids);
    ps::ParticleStructure<Particle>* ptcls = new ps::CabM<Particle>(cabm_input);

    countPIDs(ptcls, 0);
    setInitialPtclCoords(picparts, ptcls, output);
    countPIDs(ptcls, 1);
    setPtclIds(ptcls);
  
    // countPIDs(ptcls, 2);

    //cleanup
    delete ptcls;

  }
  pumipic::SummarizeTimeAcrossProcesses(pumipic::SORT_ORDER);
  if (!comm_rank)
    fprintf(stderr, "done\n");
  return 0;
}
