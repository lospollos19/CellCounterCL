"""Count cells in microscope images with OpenCL.

A pixel is part of a cell if its brightness is below a threshold. Cells are the
8-connected components of those pixels, found with a parallel union-find.
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import pyopencl as cl
from PIL import Image

DEFAULT_THRESHOLD = 200
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

ROOT = os.path.dirname(os.path.realpath(__file__))
KERNEL_FILE = os.path.join(ROOT, "main.cl")


def get_optimal_workgroup_size(device, kernel):
    """Pick a square work-group size for a 2D kernel, or None to let the
    OpenCL driver choose (better on CPUs)."""
    if device.type & cl.device_type.CPU:
        return None

    max_wg_size = device.max_work_group_size
    preferred_multiple = kernel.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        device
    )

    # Largest square size, at most 16x16
    max_dim = int(np.sqrt(max_wg_size))
    wg_dim = min(max_dim, 16)

    # Round to the preferred multiple
    adjusted_dim = (wg_dim // preferred_multiple) * preferred_multiple
    wg_dim = max(adjusted_dim, preferred_multiple)

    # Respect the limit of this kernel
    kernel_wg_limit = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
    wg_dim = max(min(wg_dim, int(np.sqrt(kernel_wg_limit))), 1)

    if wg_dim * wg_dim > max_wg_size:
        return (1, 1)
    return (wg_dim, wg_dim)


class CellCounter:
    def __init__(self, context=None, threshold=DEFAULT_THRESHOLD, workgroup="auto"):
        # create_some_context picks the first device, or the one in PYOPENCL_CTX
        self.context = context or cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.context)
        self.device = self.context.devices[0]
        self.workgroup = workgroup

        with open(KERNEL_FILE) as f:
            kernel_code = f.read()
        self.program = cl.Program(self.context, kernel_code).build(options=[f"-DTHRESHOLD={threshold}"])

    def _sizes(self, kernel, width, height):
        """Global and local sizes for a 2D kernel over the image."""
        wg_size = get_optimal_workgroup_size(self.device, kernel) if self.workgroup == "auto" else None
        if wg_size is None:
            return (width, height), None
        # The global size has to be a multiple of the work-group size
        global_size = (
            ((width + wg_size[0] - 1) // wg_size[0]) * wg_size[0],
            ((height + wg_size[1] - 1) // wg_size[1]) * wg_size[1]
        )
        return global_size, wg_size

    def image_to_mask_matrix(self, image):
        """Mask matrix of an RGB image: the pixel index for cell pixels, -1 for
        the background."""
        height, width, _ = image.shape
        image_flat = np.ascontiguousarray(image[..., :3], dtype=np.uint8).reshape(-1)

        mf = cl.mem_flags
        img_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_flat)
        mask_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size=height * width * np.dtype(np.int32).itemsize)

        kernel = self.program.image_to_mask_matrix
        global_size, wg_size = self._sizes(kernel, width, height)
        kernel.set_args(img_buf, mask_buf, np.int32(width), np.int32(height))
        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, wg_size)

        mask_matrix = np.empty((height, width), dtype=np.int32)
        cl.enqueue_copy(self.queue, mask_matrix, mask_buf).wait()
        return mask_matrix

    def count_cells(self, mask_matrix):
        """Count the cells of a mask matrix with a parallel union-find.

        Returns the number of cells and a matrix with a cell number (0 to
        count - 1) for every pixel, -1 for the background."""
        height, width = mask_matrix.shape
        max_cells = height * width

        mf = cl.mem_flags
        mask_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=np.ascontiguousarray(mask_matrix).reshape(-1))
        parent_buf = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                               hostbuf=np.arange(max_cells, dtype=np.int32))

        kernel = self.program.merge_cells
        global_size, wg_size = self._sizes(kernel, width, height)
        kernel.set_args(mask_buf, parent_buf, np.int32(width), np.int32(height))
        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, wg_size)

        parent = np.empty(max_cells, dtype=np.int32)
        cl.enqueue_copy(self.queue, parent, parent_buf).wait()

        # Point every pixel straight to its root (pointer jumping, log(n) steps)
        while True:
            grandparent = parent[parent]
            if np.array_equal(grandparent, parent):
                break
            parent = grandparent

        # Number the cells 0..count-1
        is_cell = mask_matrix.reshape(-1) != -1
        roots, numbers = np.unique(parent[is_cell], return_inverse=True)
        cell_numbers = np.full(max_cells, -1, dtype=np.int32)
        cell_numbers[is_cell] = numbers
        return len(roots), cell_numbers.reshape(height, width)


def count_cells_flood_fill(mask_matrix):
    """Count the cells of a mask matrix with a sequential flood fill.

    Simpler than the union-find, but hard to parallelize. Used as the
    reference in the tests."""
    height, width = mask_matrix.shape

    # The cell number for each pixel, -1 for the background
    cell_numbers = np.full((height, width), -1, dtype=np.int32)
    cell_index = 0

    def flood_fill(row, col, cell_number):
        queue = deque([(row, col)])
        while queue:
            row, col = queue.popleft()

            if row < 0 or row >= height or col < 0 or col >= width:
                continue
            if mask_matrix[row, col] == -1 or cell_numbers[row, col] != -1:
                # Background, or already visited
                continue

            cell_numbers[row, col] = cell_number
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr != 0 or dc != 0:
                        queue.append((row + dr, col + dc))

    for row in range(height):
        for col in range(width):
            if mask_matrix[row, col] != -1 and cell_numbers[row, col] == -1:
                flood_fill(row, col, cell_index)
                cell_index += 1

    return (cell_index, cell_numbers)


def load_image(path):
    """Load an image file as an RGB numpy array (grayscale and RGBA included)."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def highlight_cells(cell_numbers):
    """Image with every cell in a random color and the background in white."""
    # Same colors on every run
    rng = np.random.default_rng(0)
    # +1 color for the background (-1), dark colors only so they stand out on white
    colors = rng.integers(0, 200, (cell_numbers.max() + 2, 3), dtype=np.uint8)
    colors[-1] = [255, 255, 255]
    return colors[cell_numbers]


def default_images():
    folder = os.path.join(ROOT, "images")
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Count cells in microscope images with OpenCL.")
    parser.add_argument("images", nargs="*", help="images to process (default: every image in images/)")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help="a pixel is part of a cell if its brightness (0-255) is below this (default: %(default)s)")
    parser.add_argument("--workgroup", choices=["auto", "driver"], default="auto",
                        help="auto: compute a square work-group size, driver: let OpenCL choose (default: %(default)s)")
    parser.add_argument("--output", default=os.path.join(ROOT, "results"),
                        help="folder for the highlighted images (default: results/)")
    args = parser.parse_args()
    missing = [path for path in args.images if not os.path.isfile(path)]
    if missing:
        parser.error(f"file not found: {', '.join(missing)}")
    return args


def main():
    args = parse_args()
    counter = CellCounter(threshold=args.threshold, workgroup=args.workgroup)
    os.makedirs(args.output, exist_ok=True)

    print(f"Device: {counter.device.name}")
    print(f"Threshold: {args.threshold}")

    for image_path in args.images or default_images():
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        img_arr = load_image(image_path)
        height, width, _ = img_arr.shape

        start_time = time.perf_counter()
        mask_matrix = counter.image_to_mask_matrix(img_arr)
        cell_count, cell_numbers = counter.count_cells(mask_matrix)
        end_time = time.perf_counter()

        # PNG to avoid compression artifacts
        output_path = os.path.join(args.output, f"{image_name}.result.png")
        Image.fromarray(highlight_cells(cell_numbers)).save(output_path)

        print()
        print(f"Image: {image_name} ({width}x{height})")
        print(f"Cell count: {cell_count}")
        print(f"Execution time: {end_time - start_time:.4f}s")


if __name__ == "__main__":
    main()
