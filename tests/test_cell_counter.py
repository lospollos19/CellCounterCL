import os
import sys

import numpy as np
import pyopencl as cl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import CellCounter, count_cells_flood_fill, default_images, highlight_cells, load_image  # noqa: E402

IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")
SMALL_IMAGES = ["tiny-example.bmp", "Plaquette_geante.jpg",
                "Plaquettes_normales.jpg", "Anisocytose_plaquettaire.jpg"]


def has_opencl():
    try:
        return any(p.get_devices() for p in cl.get_platforms())
    except cl.Error:
        return False


pytestmark = pytest.mark.skipif(not has_opencl(), reason="no OpenCL device")


@pytest.fixture(scope="module", params=["auto", "driver"])
def counter(request):
    return CellCounter(workgroup=request.param)


def picture(rows):
    """RGB image from strings: '#' is a dark (cell) pixel, '.' a bright one."""
    return np.array([[[0, 0, 0] if c == "#" else [255, 255, 255] for c in row] for row in rows], dtype=np.uint8)


def count(counter, image):
    return counter.count_cells(counter.image_to_mask_matrix(image))[0]


def test_no_cells(counter):
    assert count(counter, picture(["....", "...."])) == 0


def test_whole_image_is_one_cell(counter):
    assert count(counter, picture(["###", "###"])) == 1


def test_diagonal_pixels_are_connected(counter):
    assert count(counter, picture(["#.#.", ".#.#", "#.#."])) == 1


def test_separate_cells(counter):
    assert count(counter, picture([
        "##..#",
        "##..#",
        ".....",
        "#.###",
    ])) == 4


def test_threshold():
    gray = np.full((2, 2, 3), 150, dtype=np.uint8)
    assert count(CellCounter(threshold=200), gray) == 1
    assert count(CellCounter(threshold=100), gray) == 0


def test_tiny_example(counter):
    assert count(counter, load_image(os.path.join(IMAGES, "tiny-example.bmp"))) == 3


@pytest.mark.parametrize("name", SMALL_IMAGES)
def test_same_cells_as_flood_fill(counter, name):
    mask = counter.image_to_mask_matrix(load_image(os.path.join(IMAGES, name)))
    expected_count, expected = count_cells_flood_fill(mask)

    # run several times: a race in the union-find would show up as a different result
    for _ in range(3):
        cell_count, cell_numbers = counter.count_cells(mask)
        assert cell_count == expected_count
        # same partition: every cell maps to exactly one reference cell
        pairs = set(zip(cell_numbers[mask != -1].tolist(), expected[mask != -1].tolist()))
        assert len(pairs) == expected_count


def test_grayscale_image(tmp_path):
    from PIL import Image
    path = tmp_path / "gray.png"
    Image.fromarray(np.array([[0, 255, 0]], dtype=np.uint8), mode="L").save(path)
    assert load_image(path).shape == (1, 3, 3)


def test_highlight_cells():
    cell_numbers = np.array([[-1, 0], [1, 1]])
    image = highlight_cells(cell_numbers)
    assert image.shape == (2, 2, 3)
    assert image[0, 0].tolist() == [255, 255, 255]
    assert image[1, 0].tolist() == image[1, 1].tolist()


def test_default_images_skips_other_files():
    assert all(not p.endswith(".txt") for p in default_images())
