#define MODE 2  // 0 for cpu, 1 for gpu batched, 2 for gpu uvm

#define BLOCKSIZE 256

#define GPUBUFFERSIZE 100000000

#define SAMPLERATE 0.01

#define GPUSTREAMS 3

#define OUTPUTUVMBUFFERSIZE 200 // GB

//probe-and-store
#define PROBEANDSTORE 0 // 0 for no probe-and-store, 1 for probe-and-store

#define SLEEPSEC 1 //Number of seconds to wait until probing the count produced on the GPU

#define PAGESIZE 4 //Unified Memory page size (in KiB)