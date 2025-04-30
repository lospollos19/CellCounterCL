#define THRESHOLD 200

/*
    >  An image to a mask matrix implementation. 
 */

float calculate_brightness(uchar4 pixel) {
    return (convert_float(pixel.x) + convert_float(pixel.y) + convert_float(pixel.z)) / 3.0f;
}

bool is_part_of_cell(uchar4 pixel) {
    return (calculate_brightness(pixel) < THRESHOLD);
}

__kernel void image_to_mask_matrix(
    __global const uchar4* inputImage,
    __global int* outputMask,
    int width,
    int height
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    uchar4 pixel = inputImage[y * width + x];
    float brightness = calculate_brightness(pixel);
    if (is_part_of_cell(pixel))
        outputMask[y * width + x] = y * width + x;
    else 
        outputMask[y * width + x] = -1;
}

/*
    > Union find implementation. 
 */

 enum Position {
    TOP_LEFT = 0,
    TOP_RIGHT = 1,
    TOP = 2,
    LEFT = 3,
};

int get_neighbour(int x, int y, int width, int height, enum Position pos) {
    switch (pos) {
        case TOP_LEFT:
            return (x > 0 && y > 0) ? (y - 1) * width + (x - 1) : -1;
        case TOP_RIGHT:
            return (x < width - 1 && y > 0) ? (y - 1) * width + (x + 1) : -1;
        case TOP:
            return (y > 0) ? (y - 1) * width + x : -1;
        case LEFT:
            return (x > 0) ? y * width + (x - 1) : -1;
    }
}

void get_all_neighbours(int x, int y, int width, int height, int neighbours[4]) {
    for (int i = 0; i < 4; i++) {
        neighbours[i] = get_neighbour(x, y, width, height, i);
    }
}


int find(int* parent, int x) {
    if (parent[x] != x) {
        parent[x] = find(parent, parent[x]);
    }
    return parent[x];
}


void union_cells(int* parent, int* rank, int x, int y) {
    int rootX = find(parent, x);
    int rootY = find(parent, y);

    if (rootX != rootY) {
        if (rank[rootX] > rank[rootY])
            parent[rootY] = rootX;
        else if (rank[rootX] < rank[rootY]) 
            parent[rootX] = rootY;
        else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }
}

__kernel void merge_cells(
    __global const int* maskMatrix,
    __global int* parent,
    __global int* rank,
    int width,
    int height
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int index = y * width + x;
    
    if (maskMatrix[index] == -1) return;
    
    parent[index] = index;
    rank[index] = 0;
    int neighbours[4];
    get_all_neighbours(x, y, width, height, neighbours);
    for (int i = 0; i < 4; i++) {
        int neighbourIndex = neighbours[i];
        if (neighbourIndex != -1 && maskMatrix[neighbourIndex] != -1) {
            union_cells(parent, rank, index, neighbourIndex);
        }
    }
}