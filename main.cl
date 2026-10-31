// THRESHOLD is passed as a build option (-DTHRESHOLD=...)
#ifndef THRESHOLD
#define THRESHOLD 200
#endif

/*
    > Image to mask matrix.
    Each pixel becomes its own index if it is part of a cell, -1 otherwise.
 */
float calculate_brightness(uchar3 pixel) {
    return (convert_float(pixel.x) + convert_float(pixel.y) + convert_float(pixel.z)) / 3.0f;
}

__kernel void image_to_mask_matrix(
    __global const uchar* inputImage,
    __global int* outputMask,
    int width,
    int height
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    uchar3 pixel = vload3(y * width + x, inputImage);
    float brightness = calculate_brightness(pixel);
    if (brightness < THRESHOLD)
        outputMask[y * width + x] = y * width + x;
    else
        outputMask[y * width + x] = -1;
}

/*
    > Union find.
    Only OpenCL 1.1 atomics (atomic_cmpxchg, atomic_or on volatile ints),
    so it also builds on OpenCL 1.2 devices such as Apple GPUs.
 */

enum Position {
    TOP_LEFT = 0,
    TOP = 1,
    TOP_RIGHT = 2,
    LEFT = 3,
    RIGHT = 4,
    BOTTOM_LEFT = 5,
    BOTTOM = 6,
    BOTTOM_RIGHT = 7
};

int get_neighbour(int x, int y, int width, int height, enum Position pos) {
    switch (pos) {
        case TOP_LEFT: return (x > 0 && y > 0) ? (y-1)*width + (x-1) : -1;
        case TOP: return (y > 0) ? (y-1)*width + x : -1;
        case TOP_RIGHT: return (x < width-1 && y > 0) ? (y-1)*width + (x+1) : -1;
        case LEFT: return (x > 0) ? y*width + (x-1) : -1;
        case RIGHT: return (x < width-1) ? y*width + (x+1) : -1;
        case BOTTOM_LEFT: return (x > 0 && y < height-1) ? (y+1)*width + (x-1) : -1;
        case BOTTOM: return (y < height-1) ? (y+1)*width + x : -1;
        case BOTTOM_RIGHT: return (x < width-1 && y < height-1) ? (y+1)*width + (x+1) : -1;
    }
    return -1;
}

// Atomic read: a plain load can return a stale value from a GPU cache,
// an atomic operation always sees the latest write.
int load(volatile __global int* p) {
    return atomic_or(p, 0);
}

// Root of x, with path splitting: each visited node is pointed to its grandparent.
int find(volatile __global int* parent, int x) {
    int px = load(&parent[x]);
    while (px != x) {
        int next = load(&parent[px]);
        atomic_cmpxchg(&parent[x], px, next);
        x = px;
        px = load(&parent[x]);
    }

    return x;
}

// Always link the larger root under the smaller one. Parents only point to
// smaller indices, so no cycle can form even when unions race. The root is
// replaced with a compare-and-swap: if another work-item changed it, retry.
void union_cells(volatile __global int* parent, int x, int y) {
    while (1) {
        int rootX = find(parent, x);
        int rootY = find(parent, y);

        if (rootX == rootY) {
            break;
        }

        int low  = min(rootX, rootY);
        int high = max(rootX, rootY);
        if (atomic_cmpxchg(&parent[high], high, low) == high) {
            break;
        }
    }
}

__kernel void merge_cells(
    __global const int* maskMatrix,
    volatile __global int* parent,
    int width,
    int height
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int index = y * width + x;

    if (maskMatrix[index] == -1) return;

    // 8-connectivity: diagonal pixels belong to the same cell
    for (int i = 0; i < 8; i++) {
        int neighbourIndex = get_neighbour(x, y, width, height, i);
        if (neighbourIndex != -1 && maskMatrix[neighbourIndex] != -1) {
            union_cells(parent, index, neighbourIndex);
        }
    }
}
