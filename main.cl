#define THRESHOLD 200


/*
    >  An image to a mask matrix implementation. 
 */
float calculate_brightness(uchar3 pixel) {
    return (convert_float(pixel.x) + convert_float(pixel.y) + convert_float(pixel.z)) / 3.0f;
}

bool is_part_of_cell(uchar3 pixel) {
    return (calculate_brightness(pixel) < THRESHOLD);
}



__kernel void image_to_mask_matrix(
    __global const uchar* inputImage,  // Pointeur générique
    __global int* outputMask,
    int width,
    int height
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    // Accès correct pour uchar3
    uchar3 pixel = vload3(y * width + x, inputImage);
    float brightness = calculate_brightness(pixel);
    if (brightness < THRESHOLD)
        outputMask[y * width + x] = y * width + x;
    else 
        outputMask[y * width + x] = -1;
}

/*
    > Union find implementation. 
 */

enum Position {
    TOP_LEFT = 0,
    TOP = 1,
    TOP_RIGHT = 2,
    LEFT = 3,
    RIGHT = 4,       // Ajouté
    BOTTOM_LEFT = 5, // Ajouté
    BOTTOM = 6,      // Ajouté
    BOTTOM_RIGHT = 7 // Ajouté
};

int get_neighbour(int x, int y, int width, int height, enum Position pos) {
    switch (pos) {
        case TOP_LEFT: return (x > 0 && y > 0) ? (y-1)*width + (x-1) : -1;
        case TOP: return (y > 0) ? (y-1)*width + x : -1;
        case TOP_RIGHT: return (x < width-1 && y > 0) ? (y-1)*width + (x+1) : -1;
        case LEFT: return (x > 0) ? y*width + (x-1) : -1;
        case RIGHT: return (x < width-1) ? y*width + (x+1) : -1; // Nouveau
        case BOTTOM_LEFT: return (x > 0 && y < height-1) ? (y+1)*width + (x-1) : -1; // Nouveau
        case BOTTOM: return (y < height-1) ? (y+1)*width + x : -1; // Nouveau
        case BOTTOM_RIGHT: return (x < width-1 && y < height-1) ? (y+1)*width + (x+1) : -1; // Nouveau
    }
    return -1;
}

void get_all_neighbours(int x, int y, int width, int height, int neighbours[8]) {
    for (int i = 0; i < 8; i++) {
        neighbours[i] = get_neighbour(x, y, width, height, i);
    }
}
/*
*/


int find(__global atomic_int* parent, int x) {
    int px = atomic_load(&parent[x]);
    while (px != x) {
        int next = atomic_load(&parent[px]);
        atomic_cmpxchg(&parent[x], px, next);
        x = px;
        px = atomic_load(&parent[x]);
    }

    return x;
}



void union_cells(__global atomic_int* parent, __global atomic_int* rank, int x, int y) {
    int rootX = find(parent, x);
    int rootY = find(parent, y);

    if (rootX != rootY) {
        if (atomic_load(&rank[rootX]) > atomic_load(&rank[rootY])) {
            atomic_store(&parent[rootY], rootX);
        } 
        else if (atomic_load(&rank[rootX]) < atomic_load(&rank[rootY])) {
            atomic_store(&parent[rootX], rootY);
        } 
        else {
            atomic_store(&parent[rootY], rootX);
            atomic_fetch_add(&rank[rootX], 1);
        }
    }
}

__kernel void merge_cells(
    __global const int* maskMatrix,
    __global atomic_int* parent,
    __global atomic_int* rank,
    int width,
    int height
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int index = y * width + x;
    
    if (maskMatrix[index] == -1) return;
    
    
    int neighbours[8];
    get_all_neighbours(x, y, width, height, neighbours);
    
    for (int i = 0; i < 8; i++) {
        int neighbourIndex = neighbours[i];
        if (neighbourIndex != -1 && maskMatrix[neighbourIndex] != -1) {
            union_cells(parent, rank, index, neighbourIndex);
        }
    }
}