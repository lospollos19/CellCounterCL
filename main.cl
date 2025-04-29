#define THRESHOLD 200

enum Position {
    TOP_LEFT = 0,
    TOP_RIGHT = 1,
    BOTTOM_LEFT = 2,
    BOTTOM_RIGHT = 3,
    TOP = 4,
    BOTTOM = 5,
    LEFT = 6,
    RIGHT = 7
};

int get_neighbour(int x, int y, int width, int height, enum Position pos) {
    switch (pos) {
        case TOP_LEFT: return (y - 1) * width + (x - 1);
        case TOP_RIGHT: return (y - 1) * width + (x + 1);
        case BOTTOM_LEFT: return (y + 1) * width + (x - 1);
        case BOTTOM_RIGHT: return (y + 1) * width + (x + 1);
        case TOP: return (y - 1) * width + x;
        case BOTTOM: return (y + 1) * width + x;
    }
    return -1;
}

int* get_all_neighbours(int x, int y, int width, int height) {
    int neighbours[8];
    for (int i = 0; i < 8; i++) {
        neighbours[i] = get_neighbour(x, y, width, height, i);
    }
    return neighbours;
}

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