import cv2
import numpy as np
import os
import shutil


# matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches, is_affine):
    """
    :param matches: if affine (3 X 2 X 2) tuple, else (4 X 2 X 2)
    :param is_affine: flag that indicates whether the file is affine or not
    :return: the transformation from pieces[0] to pieces[i]

    procedure:
    1. get the source locations and their corresponding location in the target image
    2. convert them into float32 which is the proper type for the cv2 function we want to use
    3. use the proper cv2 wrapping function based on the type of file to get the proper transform matrix
    4. return the matrix

    """

    # step 1
    src_locations, target_locations = matches[:, 0], matches[:, 1]

    # step 2
    src_locations = src_locations.astype(np.float32)
    target_locations = target_locations.astype(np.float32)

    if is_affine:
        # step 3
        affine_transform_matrix = cv2.getAffineTransform(src_locations, target_locations)

        # step 4
        return affine_transform_matrix

    else:
        # step 3
        pers_transform_matrix = cv2.getPerspectiveTransform(src_locations, target_locations)

        # step 4
        return pers_transform_matrix

# code for the classic way
def stitch_classic(img1, img2):
    """
    :param img1: First image (background), size w x h x 3
    :param img2: Second image (overlay), size w x h x 3
    :return: Stitched image with img2 overlayed on img1
    """
    # Ensure the images are of the same size
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")

    # Clone img1 to avoid modifying it directly
    stitched_img = img1.copy()

    # Get dimensions of the image
    height, width, _ = img1.shape

    # Iterate over every pixel
    for y in range(height):
        for x in range(width):
            # Check if the pixel in img2 is not black
            if not all(img2[y, x] == [0, 0, 0]):  # Check RGB channels
                stitched_img[y, x] = img2[y, x]  # Replace pixel

    return stitched_img

# code for the mask way
def stitch(img1, img2):

    """
    :param img1: image of size w x h
    :param img2: image of size w x h
    :return: merge/overlay of img2 on img1

    purpose: replace pixels in img1 with the corresponding pixel in img2 where img2 is not black
    """

    mask = np.any(img2 != 0, axis=-1)

    if len(mask.shape) == 2:
        mask = np.stack([mask] * 3, axis=-1)

    stitched_img = np.where(mask, img2, img1)
    return stitched_img


# Output size is (w,h)
def inverse_transform_target_image(target_img, original_transform, output_size):
    if original_transform.shape == (2, 3):
        return cv2.warpAffine(target_img, original_transform, output_size, flags=cv2.WARP_INVERSE_MAP)
    else:
        return cv2.warpPerspective(target_img, original_transform, output_size, flags=cv2.WARP_INVERSE_MAP)


# returns list of pieces file names
def prepare_puzzle(puzzle_dir):
    edited = os.path.join(puzzle_dir, 'abs_pieces')
    if os.path.exists(edited):
        shutil.rmtree(edited)
    os.mkdir(edited)

    affine = 4 - int("affine" in puzzle_dir)

    matches_data = os.path.join(puzzle_dir, 'matches.txt')
    n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images - 1, affine, 2, 2)

    return matches, affine == 3, n_images


if __name__ == '__main__':

    """
    the procedure like the following:
    1. for each puzzle folder, create an 'abs_pieces' subdirectory
    2. let the first piece in there be the first piece in 'pieces' subdirectory
    3. store the output of 'prepare_puzzle' function in matches, is_affine, n_images
    4. for each piece in 'pieces' excluding the first:
    5.      get the proper transform matrix using the matched points in both pieces
    6.      transform the current piece to "look like" the first piece
    7.      store the piece after modification in 'abs_pieces'
    8.      stitch the first piece with the current
    9.      carry on the stitched image in the next iteration
    10. let piece1 be the final solution, and add it to the folder
    """

    lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']
    # lst = ['puzzle_homography_1']

    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')

        # step 1
        puzzle = os.path.join('puzzles', puzzle_dir)
        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')

        # step 2
        final_puzzle = cv2.imread(f'{pieces_pth}/piece_1.jpg')
        cv2.imwrite(os.path.join(edited, 'piece_1_absolute.jpg'), final_puzzle)

        # step 3
        matches, is_affine, n_images = prepare_puzzle(puzzle)

        # step 4
        for i in range(1, n_images):

            # step 5
            current_piece = cv2.imread(f'{pieces_pth}/piece_{i + 1}.jpg')
            transform_matrix = get_transform(matches[i - 1], is_affine)

            # step 6
            current_transformed = inverse_transform_target_image(current_piece, transform_matrix,
                                                          (np.array([final_puzzle.shape[1], final_puzzle.shape[0]])))

            # step 7
            cv2.imwrite(os.path.join(edited, f'piece_{i + 1}_absolute.jpg'), current_transformed)

            # step 8, 9
            final_puzzle = stitch_classic(final_puzzle, current_transformed)

        # step 10
        cv2.imwrite(os.path.join(puzzle, 'solution.jpg'), final_puzzle)
