#define WIDTH 512 // Window width
#define HEIGHT 512 // Window height
#define DIM 512 // Grid Dimension
#define DS (DIM * DIM)

#define THREADS 256
#define BLOCKS (DS/THREADS)


#define TILE_SIZE_X (float) (WIDTH/DIM) / WIDTH
#define TILE_SIZE_Y (float) (HEIGHT/DIM) / HEIGHT