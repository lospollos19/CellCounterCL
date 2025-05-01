import time
from PIL import Image
import numpy as np
import pyopencl as cl
import os


# A pixel is part of a cell if its brightness is lower than THRESHOLD.
THRESHOLD = 200

IMAGES = [
    "./images/Anisocytose_plaquettaire.jpg",
    "./images/Eosinophil_blood_smear.jpg",
    "./images/Neutrophils_monocyte_16694967012.jpg",
    "./images/Patology_of_platelet.jpg",
    "./images/Plaquettes_normales.jpg",
    "./images/Plaquette_geante.jpg",
    "./images/Platelets2.jpg",
    "./images/RBCs_Platelets_WBC_in_PBS_2.jpg",
    "./images/Thrombocytes.jpg",
    "./images/Thrombocytosis.jpg",
     "./images/tiny-example.bmp",  # Used in the assignment
]

def get_optimal_workgroup_size(device, kernel, width, height):
    """Calcule une taille de work-group optimale avec messages de debug."""
    if device.type == cl.device_type.CPU:
        print("[DEBUG] Device CPU - Laisser OpenCL choisir la taille du work-group")
        return None

    #print(f"[DEBUG] Calcul de la taille optimale pour device: {device.name}")

    # 1. Obtenir les limites matérielles
    max_wg_size = device.max_work_group_size
    preferred_multiple = kernel.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        device
    )

   #print(f"[DEBUG] Max work-group size: {max_wg_size}")
   #print(f"[DEBUG] Preferred multiple: {preferred_multiple}")

    # 2. Calculer une taille carrée maximale
    max_dim = int(np.sqrt(max_wg_size))
    wg_dim = min(max_dim, 16)  # Limité à 16x16 pour la plupart des GPUs

    # 3. Ajuster pour respecter le multiple préféré
    adjusted_dim = (wg_dim // preferred_multiple) * preferred_multiple
    wg_dim = max(adjusted_dim, preferred_multiple)
    
    #print(f"[DEBUG] Dimension calculée initiale: {wg_dim}x{wg_dim}")

    # 4. Vérifier la compatibilité avec le kernel
    try:
        kernel_wg_limit = kernel.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE,
            device
        )
        #print(f"[DEBUG] Kernel work-group limit: {kernel_wg_limit}")
        wg_dim = min(wg_dim, int(np.sqrt(kernel_wg_limit)))
    except Exception as e:
        print(f"[DEBUG] Impossible de récupérer la limite du kernel: {str(e)}")
        pass

    # 5. Forcer une taille valide
    wg_dim = max(wg_dim, 1)
    wg_size = (wg_dim, wg_dim)

    # 6. Validation finale
    if wg_size[0] * wg_size[1] > max_wg_size:
        print("[WARNING] Taille calculée trop grande - Fallback à (1,1)")
        wg_size = (1, 1)
    else:
        pass
        #print(f"[DEBUG] Taille optimale validée: {wg_size}")

    return wg_size



context = cl.create_some_context()
queue = cl.CommandQueue(context)
device = context.devices[0]
kernel_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "main.cl")
kernel_code = open(kernel_file).read()
program = cl.Program(context, kernel_code).build()


def load_image(path):
    """Load an image file into a PIL image."""
    return Image.open(path)


def pixel_brightness(pixel):
    """Calculate the brightness of a pixel."""
    # The brightness of a pixel is the average of its RGB values.
    # Note that pixels are of type np.uint8 (0-255), so if we just sum them, we
    # get overflows. Hence, we convert them to np.float32 first.
    # Note also that rounding in Python may be different from rounding in
    # OpenCL (also depending on if/where you convert to float or another int
    # size). You can adjust the Python code below to match your OpenCL code.
    return np.sum(pixel.astype(np.float32)) / 3


def is_part_of_cell(pixel):
    """Check if a pixel is part of a cell, i.e. if its brightness is below the
    threshold."""
    return pixel_brightness(pixel) < THRESHOLD

def print_matrix(matrix, cell_width=4, decimals=2, title=None, highlight_value=None, highlight_color='\033[91m'):
    """
    Affiche une matrice de manière formatée et lisible.
    
    Args:
        matrix: La matrice à afficher (liste 2D ou np.array)
        cell_width: Largeur de chaque cellule (default: 4)
        decimals: Nombre de décimales pour les floats (default: 2)
        title: Titre à afficher au-dessus (default: None)
        highlight_value: Valeur à mettre en évidence (default: None)
        highlight_color: Code couleur ANSI pour le surlignage (default: rouge)
    """
    # Reset color
    reset_color = '\033[0m'
    
    if title:
        print(f"\n{title.center(cell_width * len(matrix[0]))}\n")
    
    for row in matrix:
        formatted_row = []
        for val in row:
            # Formatage différent selon le type
            if isinstance(val, float):
                s = f"{val:.{decimals}f}"
            elif isinstance(val, (int, np.integer)):
                s = f"{val:d}"
            else:
                s = str(val)
            
            # Mise en évidence si demandée
            if highlight_value is not None and val == highlight_value:
                s = f"{highlight_color}{s}{reset_color}"
            
            formatted_row.append(s.rjust(cell_width))
        print("[" + " ".join(formatted_row) + "]")
    print()


def image_to_mask_matrix(image):
    height, width, channels = image.shape   
    if channels == 4:
        image = image[..., :3]
    
    image_flat = image.reshape(-1)
    
    mf = cl.mem_flags
    img_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_flat)
    mask_buf = cl.Buffer(context, mf.WRITE_ONLY, size=height * width * np.dtype('int32').itemsize)
    
    kernel = program.image_to_mask_matrix
    wg_size = get_optimal_workgroup_size(device, kernel, width, height)
    
    # Ajuster la taille globale pour être un multiple de wg_size
    if wg_size is not None:
        global_size = (
            ((width + wg_size[0] - 1) // wg_size[0]) * wg_size[0],
            ((height + wg_size[1] - 1) // wg_size[1]) * wg_size[1]
        )
    else:
        global_size = (width, height)
    
    kernel.set_args(img_buf, mask_buf, np.int32(width), np.int32(height))
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, wg_size).wait()
    
    mask_matrix = np.empty((height, width), dtype=np.int32)
    cl.enqueue_copy(queue, mask_matrix, mask_buf).wait()
    return mask_matrix

def count_cells(mask_matrix):
    height, width = mask_matrix.shape
    max_cells = height * width

    mf = cl.mem_flags
    mask_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask_matrix.flatten())
    parent_buf = cl.Buffer(context, mf.READ_WRITE, size=max_cells * np.dtype(np.int32).itemsize)
    rank_buf = cl.Buffer(context, mf.READ_WRITE, size=max_cells * np.dtype(np.int32).itemsize)

    # Initialisation des structures union-find
    parent_init = np.arange(max_cells, dtype=np.int32)
    rank_init = np.zeros(max_cells, dtype=np.int32)
    cl.enqueue_copy(queue, parent_buf, parent_init).wait()
    cl.enqueue_copy(queue, rank_buf, rank_init).wait()

    # Configuration de l'exécution
    kernel = program.merge_cells
    wg_size = get_optimal_workgroup_size(device, kernel, width, height)
    
    # Ajustement de la taille globale
    if wg_size is not None:
        global_size = (
            ((width + wg_size[0] - 1) // wg_size[0]) * wg_size[0],
            ((height + wg_size[1] - 1) // wg_size[1]) * wg_size[1]
        )
    else:
        global_size = (width, height)

    # Exécution du kernel
    program.merge_cells(
        queue, global_size, wg_size,
        mask_buf, parent_buf, rank_buf,
        np.int32(width), np.int32(height)
    )
    queue.finish()

    # Récupération des résultats
    parent = np.empty(max_cells, dtype=np.int32)
    cl.enqueue_copy(queue, parent, parent_buf).wait()

    # Post-traitement CPU (path compression)
    cell_numbers = np.zeros((height, width), dtype=np.int32)
    for row in range(height):
        for col in range(width):
            index = row * width + col
            if mask_matrix[row, col] == -1:
                cell_numbers[row, col] = -1
            else:
                root = index
                while parent[root] != root:
                    parent[root] = parent[parent[root]]  # Path compression
                    root = parent[root]
                cell_numbers[row, col] = root

    # Comptage des composantes connexes
    unique_cells = np.unique(cell_numbers)
    cell_count = len(unique_cells) - 1 if -1 in unique_cells else len(unique_cells)
    
    return (cell_count, cell_numbers)

def count_cells_flood_fill(mask_matrix):
    """Find and count the number of cells in the mask matrix using a flood
    fill algorithm.

    This is a simpler algorithm than the union find algorithm above, and will
    typcially be faster in a sequential execution. However, it is not as easy
    to parallelize on a GPU, because it requires either a queue or recursion.
    """
    height, width = mask_matrix.shape

    # The cell number for each pixel. -1 means the pixel is not part of a cell.
    # Initialize with all -1.
    cell_numbers = np.full((height, width), -1, dtype=np.int32)

    # The next cell number to use
    cell_index = 0

    def flood_fill(row, col, cell_number):
        """Fill a cell using a flood fill algorithm."""
        queue = [(row, col)]
        while len(queue) > 0:
            row, col = queue.pop(0)

            if row < 0 or row >= height:
                continue
            if col < 0 or col >= width:
                continue
            if mask_matrix[row, col] == -1:
                # Background pixel: skip
                continue
            if cell_numbers[row, col] != -1:
                # Already visited by a previous flood fill
                continue

            cell_numbers[row, col] = cell_number

            for dr, dc in [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                # skip self
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]:
                queue.append((row + dr, col + dc))

    # Iterate over all pixels, and fill each cell using a flood fill.
    for row in range(height):
        for col in range(width):
            if mask_matrix[row, col] == -1:
                continue

            if cell_numbers[row, col] != -1:
                # Already part of a cell: skip
                continue
            # Not part of a cell yet: fill this cell
            flood_fill(row, col, cell_index)
            cell_index += 1

    return (cell_index, cell_numbers)


def highlight_cells(cell_numbers):
    """Generate an image with the cells highlighted.

    This is not needed for the algorithm, but can be useful for debugging."""
    height, width = cell_numbers.shape
    cell_count = cell_numbers.max() + 2
    # +2 because the background is -1, and we want to include the last cell
    # number in the range.

    # Set random seed to get the same colors each time
    np.random.seed(0)

    # Generate a random color for each cell.
    # This is an matrix of shape (cell_count, 3), where each row is a color.
    # E.g. colors[123] is the RGB color for cell 123.
    # We use colors from 0 to 200 to avoid very light colors (close to white).
    colors = np.random.randint(0, 200, (cell_count, 3), dtype=np.uint8)
    # Cell number -1 is reserved for background pixels that are not part of a
    # cell: make these pixels white.
    colors[-1] = [255, 255, 255]

    # Create an image to store the result
    result = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign a color to each pixel based on the cell number
    for row in range(height):
        for col in range(width):
            result[row, col] = colors[cell_numbers[row, col]]

    return result


def matrix_to_svg(matrix, filename):
    """Convert a matrix of numbers to an SVG file.

    This is for the purposes of debugging. I used this to generate SVG files
    for the assignment. Note that this will only work for small images,
    otherwise the SVG file becomes too large.
    """
    height, width = matrix.shape
    with open(filename, "w", encoding="utf-8") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 {width} {height}">\n'
        )
        f.write(
            "<style>\nrect { fill: black; }\ntext { fill: white; font-size: 0.5px; text-anchor: middle; dominant-baseline: middle; }\n</style>\n"
        )
        for row in range(height):
            for col in range(width):
                if matrix[row, col] == -1:
                    continue
                f.write(f'<rect x="{col}" y="{row}" width="1" height="1" />\n')
                f.write(f'<text x="{col}.5" y="{row}.5">{matrix[row, col]}</text>\n')
        f.write("</svg>\n")


def main():
    print(f"Threshold: {THRESHOLD}")

    for image_path in IMAGES:
        image_name = image_path.rsplit("/", maxsplit=1)[-1].split(".")[0]
        print()
        print(f"Image: {image_name} ({image_path})")

        # Load image
        img = load_image(image_path)

        # Convert image to numpy array
        img_arr = np.asarray(img).astype(np.uint8)
        print(f"Image size: {img_arr.shape}, {img_arr.size} pixels")
        # Note: PIL uses (x, y) coordinates, while numpy uses (row, col)
        # coordinates. Hence, the width and height are swapped!
        # print(f"Image shape: {img.size}")
        # print(f"Image array size: {img_arr.shape}")

        # For OpenCL, you can flatten the image to a 1D array using
        # img_arr.flatten()

        start_time = time.perf_counter()

        # 1. Convert the image to a "mask matrix", where True indicates the
        # pixel is part of a cell.
        mask_matrix = image_to_mask_matrix(img_arr)
        # matrix_to_svg(mask_matrix, f"{image_name}.mask.svg")

        # 2. Count the number of cells in the image
        (cell_count, cell_numbers) = count_cells(mask_matrix)
        # matrix_to_svg(cell_numbers, f"{image_name}.result.svg")
        # (cell_count, cell_numbers) = count_cells_flood_fill(mask_matrix)

        end_time = time.perf_counter()

        # Generate image with the cells highlighted, using different colors.
        # This might be useful for debugging.
        cell_image = highlight_cells(cell_numbers)
        Image.fromarray(cell_image).save(f"results/{image_name}.result.png")
        # Note: even though the input is JPG, it is better to save the output
        # as PNG, to avoid compression artifacts.

        # Display results
        print(f"Cell count: {cell_count}")
        print(f"Execution time: {end_time - start_time:.4f}s")


if __name__ == "__main__":
    main()