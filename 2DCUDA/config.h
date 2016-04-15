#define WIDTH 512 // Window width
#define HEIGHT 512 // Window height
#define DIM 256 // Grid Dimension
#define DS (DIM * DIM)

#define THREADS 736
#define BLOCKS 90


#define TILE_SIZE_X (float) (WIDTH/DIM) / WIDTH
#define TILE_SIZE_Y (float) (HEIGHT/DIM) / HEIGHT

#define THREADS_PER_BLOCK 512