
#see params.h for the parameters


#####################################
#Build binaries for the code in the paper
#see params.h for the parameters

SOURCES = main.cu CPU.cu GPU.cu kernel.cu
CUDAOBJECTS = CPU.o GPU.o kernel.o main.o
CC = nvcc
EXECUTABLE = main



#update your compute capability here
COMPUTE_CAPABILITY = 80
COMPUTE_CAPABILITY_FLAGS = -arch=compute_$(COMPUTE_CAPABILITY) -code=sm_$(COMPUTE_CAPABILITY)


FLAGS = -std=c++17 -O3 -Xcompiler -fopenmp -lcuda -lineinfo -g
CFLAGS = -c -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES


all: $(EXECUTABLE)

main.o: main.cu
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) main.cu 

kernel.o: kernel.cu
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) kernel.cu 		

GPU.o: GPU.cu
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) GPU.cu

CPU.o: CPU.cu
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) CPU.cu 	



$(EXECUTABLE): $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@

clean:
	rm $(CUDAOBJECTS)
	rm main