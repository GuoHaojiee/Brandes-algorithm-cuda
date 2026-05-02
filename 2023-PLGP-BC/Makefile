# defines
MPICXX=mpicxx
SPEEDFLAGS=-O3
CXXFLAGS=-Wall -std=c++11 $(SPEEDFLAGS)
LDFLAGS=$(SPEEDFLAGS)

# CUDA 配置（P100，sm_60）
NVCC=nvcc
CUDA_ARCH=-arch=sm_60
CUDA_PATH?=/usr/local/cuda
CUDA_LDFLAGS=-L$(CUDA_PATH)/lib64 -lcudart -Wl,-rpath,$(CUDA_PATH)/lib64

TARGET = gen_valid_info validation gen_RMAT gen_random solution_mpi

all: $(TARGET)

# your own distributed implementation（加入 rcm.o, brandes_gpu.o 和 CUDA 链接）
solution_mpi: main_mpi.o solution_mpi.o rcm.o brandes_gpu.o graph_tools.o gen_RMAT_mpi.o gen_random_mpi.o
	$(MPICXX) $^ -o $@ $(LDFLAGS) $(CUDA_LDFLAGS)

# CUDA kernel 编译规则（nvcc -arch=sm_60）
brandes_gpu.o: brandes_gpu.cu
	$(NVCC) $(CUDA_ARCH) -std=c++11 -O3 -c $< -o $@

# RMAT generator
gen_RMAT: gen_RMAT.o graph_tools.o
	$(MPICXX) $^ -o $@ $(LDFLAGS)

# Erdos-Renyi (random) graph generator
gen_random: gen_random.o graph_tools.o
	$(MPICXX) $^ -o $@ $(LDFLAGS)

# gen_valid_info
gen_valid_info: gen_valid_info.o reference_bfs.o graph_tools.o
	$(MPICXX) $^ -o $@ $(LDFLAGS)

# validation
validation: validation.o
	$(MPICXX) $^ -o $@ $(LDFLAGS)

.cpp.o:
	$(MPICXX) $(CXXFLAGS) -o $@ -c $

clean:
	rm -rf *.o $(TARGET)