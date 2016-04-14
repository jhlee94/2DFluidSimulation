#define WIDTH 512 // Window width
#define HEIGHT 512 // Window height
#define DIM 512 // Grid Dimension
#define DS (DIM * DIM)

#define TILEX 64 // Tile Width
#define TILEY 64 // Tile Height
#define TIDSX 64 // Thread x
#define TIDSY 4 // Thread y


#define TILE_SIZE_X (float) (WIDTH/DIM) / WIDTH
#define TILE_SIZE_Y (float) (HEIGHT/DIM) / HEIGHT

#define THREADS_PER_BLOCK 512