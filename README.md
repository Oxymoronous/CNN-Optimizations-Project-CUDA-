# ECE408_Fall22
UIUC - Applied Parallel Programming 

These are code snippets implemented by myself while taking UIUC/NVIDIA's parallel programming class in Fall 2022.
The kernel functionalities all obtained full credit for the class assignments, although I dropped MP4: 3D convolution.

The final project was about optimizing a forward convolutional neural network (while using CUDA parallel programming) with our own choice of optimizations.
The list of optimizations that I chose to implement with their corresponding points awarded are:
1 Tiled shared memory convolution (2 points)
2 Tuning with restrict and loop unrolling (considered as one optimization only if you do both) (3 points)
3 Fixed point (FP16) arithmetic. (note this can modify model accuracy slightly) (4 points)
4 Weight matrix (kernel values) in constant memory (1 point)
5 Using Streams to overlap computation with data transfer (4 points)

The optimizations all cap up to a maximum of 14 points (course awarded 10 points max), with the fastest optimization being the second one listed.

