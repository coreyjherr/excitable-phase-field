#!/home/cherr/miniconda3/envs/opencv/bin/python
# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from pathlib import Path
import cv2
import pandas as pd
from matplotlib.colors import hex2color, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy.ndimage import center_of_mass

def read_dynamics_file(filename: Path | str) -> pd.DataFrame:
    # open the file and read the first int to get the number of columns
    if isinstance(filename, str):
        filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File {filename} does not exist")

    if filename.suffix == ".dat":
        num_columns, real_width_bytes = np.fromfile(filename, dtype=np.float32, count=2)
        file_size_bytes = int(filename.stat().st_size)
        real_width_bytes = int(real_width_bytes)
        num_columns = int(num_columns)
        num_rows = int(
            (file_size_bytes - 2 * real_width_bytes) // (num_columns * real_width_bytes)
        )

        column_names = [
            "cell",
            "t",
            "grid_offset_x",
            "grid_offset_y",
            "phi_area",
            "A_area",
            "com_x",
            "com_y",
            "comA_x",
            "comA_y",
        ]
        dtype = np.float32 if real_width_bytes == 4 else np.float64
        data = np.fromfile(
            filename,
            dtype=dtype,
            count=int(num_columns * num_rows),
            offset=2 * real_width_bytes,
        )
        df = pd.DataFrame(
            data.reshape(int(num_rows), int(num_columns)), columns=column_names
        )
        df["com_x"] = df["com_x"] + df["grid_offset_x"]
        df["com_y"] = df["com_y"] + df["grid_offset_y"]
        df["comA_x"] = df["comA_x"] + df["grid_offset_x"]
        df["comA_y"] = df["comA_y"] + df["grid_offset_y"]

        com_x_avg_0 = df.query("t == 0")["com_x"].mean()
        com_y_avg_0 = df.query("t == 0")["com_y"].mean()
        df["com_x"] = df["com_x"] - com_x_avg_0
        df["com_y"] = df["com_y"] - com_y_avg_0
        df["comA_x"] = df["comA_x"] - com_x_avg_0
        df["comA_y"] = df["comA_y"] - com_y_avg_0
    elif filename.suffix == ".csv":
        df = pd.read_csv(filename)

    return df


class CompressedVideo:
    def __init__(self, filename: str | Path, dtype: np.dtype = np.float32):
        self.filename = filename if isinstance(filename, Path) else Path(filename)

        self.file_size_bytes = filename.stat().st_size

        self.frame_sizes = []
        self.frame_positions = []
        self.num_frames = 0

        self.video_indexed = False

        self.dtype = dtype

    def get_compressed_indices(self):
        current_position = 0
        current_size = (0, 0)
        while current_position < self.file_size_bytes:
            current_size = np.fromfile(
                self.filename, dtype=np.int32, count=2, offset=self.current_position
            )
            self.frame_sizes.append(current_size)
            self.frame_positions.append(current_position)
            current_position += current_size[0] * current_size[1] * self.dtype.itemsize
            self.num_frames += 1
        self.video_indexed = True

    def __len__(self):
        if not self.video_indexed:
            self.get_compressed_indices()
        return self.num_frames

    def __getitem__(self, idx):
        if idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of bounds")
        if not self.video_indexed:
            self.get_compressed_indices()
        return np.fromfile(
            self.filename,
            dtype=self.dtype,
            count=self.frame_sizes[idx][0] * self.frame_sizes[idx][1],
            offset=self.frame_positions[idx],
        ).reshape(self.frame_sizes[idx][1], self.frame_sizes[idx][0]).T

    def __iter__(self):
        current_size = (0, 0)
        current_position = 0
        while current_position < self.file_size_bytes:
            current_size = np.fromfile(
                self.filename, dtype=np.int32, count=2, offset=self.current_position
            )
            current_position += 8
            frame = np.fromfile(
                self.filename,
                dtype=self.dtype,
                count=current_size[0] * current_size[1],
                offset=self.current_position,
            ).reshape(current_size[1], current_size[0]).T
            current_position += current_size[0] * current_size[1] * self.dtype.itemsize
            yield frame



def create_angle_image(label):
    """Create an image where the angle of the vector from the center of mass to the center of each connected component is stored in the pixel value.

    Parameters:
    label: np.ndarray
        2D array of label image

    Returns:
    angle: np.ndarray
        2D array with the angle of the vector from the center of mass to the center of the cell
    """
    num_cells = label.max()
    centroids = center_of_mass(label > 0, label, range(1, num_cells + 1))
    centroids = np.array(centroids)
    x_centroids = centroids[:, 1]
    y_centroids = centroids[:, 0]
    X, Y = np.meshgrid(np.arange(label.shape[1]), np.arange(label.shape[0]))

    # replace the labels with the corresponding center of mass

    key = np.zeros((x_centroids.shape[0], 2))
    key[:, 0] = np.arange(1, x_centroids.shape[0] + 1)
    key[:, 1] = x_centroids

    idx = np.array(key[:, 0], dtype=float).astype(int)
    keyarray = np.empty(num_cells + 1, dtype=key[:, 1].dtype)
    keyarray[:] = 0
    keyarray[idx] = key[:, 1]

    com_image_x = keyarray[label]

    key[:, 1] = y_centroids
    keyarray[:] = 0
    keyarray[idx] = key[:, 1]
    com_image_y = keyarray[label]

    # calculate the angle of the vector from the center of mass to the center of the cell only where the label is not zero
    angle = np.arctan2(com_image_y - Y, com_image_x - X)
    angle = angle % (2 * np.pi)
    angle = angle / (2 * np.pi) * 255
    angle = angle.astype(np.uint8)
    angle[label == 0] = 0

    return angle


def load_label(filename: Path | str) -> np.ndarray:
    size = np.fromfile(filename, dtype=np.int32, count=2)
    label = (
        np.fromfile(filename, dtype=np.float32, count=size[0] * size[1], offset=8)
        .reshape(size[1], size[0])
        .T
    )
    return label


def test_create_angle_image():
    # test with bigger random array 100x100
    num_cells = 4
    dim = 700
    size = np.random.randint(10, 50, num_cells)
    size = size + 1
    label = np.zeros((dim, dim), dtype=np.int32)
    for i in range(1, num_cells):
        centroid = np.random.randint(0, dim, 2)
        label[
            centroid[0] - size[i] : centroid[0] + size[i],
            centroid[1] - size[i] : centroid[1] + size[i],
        ] = i

    angle = create_angle_image(label)
    # ignore the visual bug with imshow
    plt.imshow(angle)

    label = load_label(Path("../visualize_waves/label_20241230_134448_0.dat"))
    label = label.astype(np.int32)
    angle = create_angle_image(label)
    plt.figure()
    plt.imshow(angle)


# test_create_angle_image()
# %%


def find_min_max_avg_condensed(
    A_filename: Path, phi_filename: Path, start: int, end: int, quantile: float
):
    print(A_filename, phi_filename, start, end)
    current_position = 0
    min_A = np.inf
    max_A = 0
    avg_A = 0
    quant_A = 0
    for t in range(end - 1):
        current_size = np.fromfile(
            A_filename, dtype=np.int32, count=2, offset=current_position
        )
        current_position += 8
        if t < start:
            # skip the frames up until start
            current_position += current_size[0] * current_size[1] * 4
            continue
        try:
            A = np.fromfile(
                A_filename,
                dtype=np.float32,
                count=current_size[0] * current_size[1],
                offset=current_position,
            )
            phi = np.fromfile(
                phi_filename,
                dtype=np.float32,
                count=current_size[0] * current_size[1],
                offset=current_position,
            )
        except:
            break
        current_position += current_size[0] * current_size[1] * 4
        min_A = min(min_A, A.min())
        max_A = max(max_A, A.max())
        quant_A += np.quantile(A[phi > 0.5], quantile)
        avg_A += A.sum() / phi.sum()
    avg_A /= end - start
    quant_A /= end - start

    return min_A, max_A, avg_A, quant_A


def create_label_video(
    label_filename: Path,
    phi_filename: Path,
    A_filename: Path,
    dynamics_filename: Path,
    param_filename: Path,
    verbose: bool,
):
    # read the parameter file and flip the rows to columns
    param_df = pd.read_csv(param_filename, sep=r"\s*=?\s+", header=None, engine="python")
    param_df.set_index(0, inplace=True)
    param_df = param_df.T

    numCells = int(param_df["numCells"].iloc[0])
    # T = param_df['T'].iloc[0]/param_df['blind'].iloc[0]
    # step between frames is defined to 400
    T = param_df["T"].iloc[0] / 400
    T = int(T)
    skip_num = int(400 / param_df["blind"].iloc[0])
    num_rows = param_df["T"].iloc[0] / param_df["blind"].iloc[0] * numCells

    df = read_dynamics_file(dynamics_filename)
    R_cutoff = 500

    # get the grid locations reshaped to be numCells x T x 2
    grid_location_x = df.pivot_table(
        index="cell", columns="t", values="grid_offset_x"
    ).to_numpy()
    grid_location_y = df.pivot_table(
        index="cell", columns="t", values="grid_offset_y"
    ).to_numpy()
    location_filter = np.logical_and(
        np.abs(grid_location_x) < R_cutoff, np.abs(grid_location_y) < R_cutoff
    )
    directions = np.stack([grid_location_x, grid_location_y], axis=-1)
    directions = directions.astype(np.int32)[:, ::skip_num, :]
    print(directions.shape)

    max_y = np.max(directions[:, :, 1])
    max_x = np.max(directions[:, :, 0])
    min_y = np.min(directions[:, :, 1])
    min_x = np.min(directions[:, :, 0])

    # max_y = np.max(grid_location_y_filtered)
    # max_x = np.max(grid_location_x_filtered)
    # min_y = np.min(grid_location_y_filtered)
    # min_x = np.min(grid_location_x_filtered)

    N = param_df["N"].iloc[0]
    size_x = int(max_x - min_x + N)
    size_y = int(max_y - min_y + N)
    cut_size = int(N / 16)
    size_vid_x = size_x - 2 * cut_size
    size_vid_y = size_y - 2 * cut_size

    if verbose:
        print(size_vid_x, size_vid_y)

    if size_x * size_y > 5000 * 5000:
        print(f"Video is too large: {size_x*size_y}")
        return

    movie_path = label_filename.parent / "movies"
    if not movie_path.exists():
        movie_path.mkdir()
    label_movie_filename = str(movie_path / label_filename.with_suffix(".mp4").name)
    vidLabel = cv2.VideoWriter(
        label_movie_filename,
        cv2.VideoWriter_fourcc(*"avc1"),
        60,
        (size_vid_y, size_vid_x),
        isColor=True,
    )

    min_A, max_A, avg_A, quant_A = find_min_max_avg_condensed(
        A_filename, phi_filename, T // 20, T - 1, quantile=0.99
    )
    if verbose:
        print(min_A, max_A)
    max_A = max_A

    current_position = 0
    current_frame = 1
    make_angle_image = True

    background_color = np.array(hex2color("#FFFFFF"), dtype=np.uint8) * 255
    palette = sns.blend_palette(
        ["#fbfbfb", "#fbfbfb", "#fbfbfb", "#800000", "#ff0000"], as_cmap=True
    )

    if make_angle_image:
        background_color = np.array(hex2color("#000000"), dtype=np.uint8) * 255

    # norm = TwoSlopeNorm(vmin=min_A, vcenter=avg_A, vmax=max_A)
    # sm = ScalarMappable(norm=norm, cmap="bwr")

    # # Create a color palette centered on the average value of A
    # palette = sm.to_rgba(np.linspace(min_A, max_A, 256))[:, :3] * 255
    # palette = palette.astype(np.uint8)
    # palette = palette[:, np.newaxis, [2, 1, 0]]
    progress_bar_on = not verbose

    for i in trange(
        T - 1, desc=label_filename.stem, dynamic_ncols=True, disable=progress_bar_on
    ):
        current_size = np.fromfile(
            label_filename, dtype=np.int32, count=2, offset=current_position
        )
        current_position += 8

        current_min_x = np.min(directions[:, i, 0]) - min_x
        current_min_y = np.min(directions[:, i, 1]) - min_y

        try:
            label = (
                np.fromfile(
                    label_filename,
                    dtype=np.float32,
                    count=current_size[0] * current_size[1],
                    offset=current_position,
                )
                .reshape(current_size[1], current_size[0])
                .T
            )
            A = (
                np.fromfile(
                    A_filename,
                    dtype=np.float32,
                    count=current_size[0] * current_size[1],
                    offset=current_position,
                )
                .reshape(current_size[1], current_size[0])
                .T
            )
            phi = (
                np.fromfile(
                    phi_filename,
                    dtype=np.float32,
                    count=current_size[0] * current_size[1],
                    offset=current_position,
                )
                .reshape(current_size[1], current_size[0])
                .T
            )
        except Exception as e:
            print(e)
            break

        current_position += current_size[0] * current_size[1] * 4
        if i == 0:
            continue

        # label already has 32 bit integer values just stored as float32, change type to uint32
        label = label.astype(np.int32)

        # squash A to be between min_A and max_A
        if make_angle_image:
            min_A = 0
            max_A = quant_A

        A = (np.clip(A, min_A, max_A) - min_A) / (max_A - min_A)
        frame = np.ones((size_x, size_y, 3), dtype=np.uint8) * background_color
        phi = phi > 0.5
        phi = phi.astype(np.uint8)

        if make_angle_image:
            label = label.astype(np.uint32)
            angle_image = create_angle_image(label)
            angle_image = (
                cv2.applyColorMap(angle_image, cv2.COLORMAP_HSV) * A[:, :, np.newaxis]
            )
            angle_image = angle_image.astype(np.uint8)

            # make the intensity of the angle image dependent on the A value
            angle_image = cv2.bitwise_and(angle_image, angle_image, mask=phi)
            angle_image += cv2.bitwise_and(
                np.ones_like(angle_image) * background_color,
                np.ones_like(angle_image) * background_color,
                mask=1 - phi,
            )
            frame[
                current_min_x : current_min_x + current_size[0],
                current_min_y : current_min_y + current_size[1],
            ] = angle_image

        elif ~make_angle_image:
            A = (A * 255).astype(np.uint8)
            # A = cv2.applyColorMap(A, palette).astype(np.uint8)

            A_masked = cv2.bitwise_and(A, A, mask=phi)
            A_masked += cv2.bitwise_and(
                np.ones_like(A) * background_color,
                np.ones_like(A) * background_color,
                mask=1 - phi,
            )

            frame[
                current_min_x : current_min_x + current_size[0],
                current_min_y : current_min_y + current_size[1],
            ] = A_masked

        label_contour = np.zeros((size_x, size_y), dtype=np.int32)
        label_contour[
            current_min_x : current_min_x + current_size[0],
            current_min_y : current_min_y + current_size[1],
        ] = label

        contours, hierarchy = cv2.findContours(
            label_contour.astype(np.int32),
            mode=cv2.RETR_CCOMP,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        if make_angle_image:
            cv2.drawContours(frame, contours, -1, (255, 255, 255), 1)
        else:
            cv2.drawContours(frame, contours, -1, (0, 0, 0), 1)
        vidLabel.write(frame[cut_size:-cut_size, cut_size:-cut_size])

    vidLabel.release()


def get_min_max_avg_uncompressed(
    A_filename: Path, phi_filename: Path, T, numCells, N
) -> tuple[float, float, float]:
    min_val = np.inf
    max_val = 0
    avg = 0
    for t in range(T):
        phi = np.fromfile(
            phi_filename,
            dtype=np.float32,
            count=numCells * N * N,
            offset=t * numCells * N * N * 4,
        )
        A = np.fromfile(
            A_filename,
            dtype=np.float32,
            count=numCells * N * N,
            offset=t * numCells * N * N * 4,
        )
        min_val = min(min_val, (A * phi).min())
        max_val = max(max_val, (A * phi).max())
        avg += A.sum() / phi.sum()

    avg /= T

    return min_val, max_val, avg


def get_min_max_C(C_filename: Path, T, Nchem) -> tuple[float, float]:
    min_val = np.inf
    max_val = 0
    for t in range(T):
        C = np.fromfile(
            C_filename,
            dtype=np.float32,
            count=Nchem * Nchem,
            offset=t * Nchem * Nchem * 4,
        )
        min_val = min(min_val, C.min())
        max_val = max(max_val, C.max())

    return min_val, max_val


def create_chemotaxis_video(C_filename: Path, param_filename: Path):
    param_df = pd.read_csv(param_filename, sep="\s+", header=None)
    param_df.set_index(0, inplace=True)
    param_df = param_df.T

    Nchem = int(param_df["Nchem"].iloc[0])
    T = int(param_df["T"].iloc[0] / 400)
    movie_path = C_filename.parent / "movies"
    movie_filename = str(movie_path / C_filename.with_suffix(".mp4").name)

    min_val, max_val = get_min_max_C(C_filename, T, Nchem)
    print(min_val, max_val)

    # create a video
    chem_video = cv2.VideoWriter(
        movie_filename,
        cv2.VideoWriter_fourcc(*"avc1"),
        60,
        (Nchem, Nchem),
        isColor=False,
    )

    for t in trange(T, desc=C_filename.stem, dynamic_ncols=True):
        C = np.fromfile(
            C_filename,
            dtype=np.float32,
            count=Nchem * Nchem,
            offset=t * Nchem * Nchem * 4,
        ).reshape(Nchem, Nchem)
        C = ((C - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        chem_video.write(C)

    chem_video.release()


def create_A_video(A_filename: Path, param_filename: Path):
    param_df = pd.read_csv(param_filename, sep="\s+", header=None)
    param_df.set_index(0, inplace=True)
    param_df = param_df.T

    N = int(param_df["N"].iloc[0])
    T = int(param_df["T"].iloc[0] / 400)
    movie_path = A_filename.parent / "movies"
    movie_filename = str(movie_path / A_filename.with_suffix(".mp4").name)
    if not movie_path.exists():
        movie_path.mkdir()

    C_tmp = np.fromfile(A_filename, dtype=np.float32)
    min_val = C_tmp.min()
    max_val = C_tmp.max()
    print(min_val, max_val)
    # delete the temporary array
    del C_tmp

    # create a video
    A_video = cv2.VideoWriter(
        movie_filename, cv2.VideoWriter_fourcc(*"avc1"), 60, (N, N), isColor=False
    )

    for t in trange(T, desc=A_filename.stem, dynamic_ncols=True):
        C = np.fromfile(
            A_filename, dtype=np.float32, count=N * N, offset=t * N * N * 4
        ).reshape(N, N)
        C = ((C - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        A_video.write(C)

    A_video.release()


def create_label_video_from_uncompressed(
    phi_filename: Path, A_filename: Path, dynamics_filename: Path, param_filename: Path
):
    # read the parameter file and flip the rows to columns
    param_df = pd.read_csv(param_filename, sep="\s+", header=None)
    param_df.set_index(0, inplace=True)
    param_df = param_df.T

    numCells = int(param_df["numCells"].iloc[0])
    # T = param_df['T'].iloc[0]/param_df['blind'].iloc[0]
    # step between frames is defined to 400
    T = param_df["T"].iloc[0] / 400
    T = int(T)
    skip_num = int(400 / param_df["blind"].iloc[0])
    print(T, skip_num)
    num_rows = param_df["T"].iloc[0] / param_df["blind"].iloc[0] * numCells

    df = read_dynamics_file(dynamics_filename)
    R_cutoff = 500 / 0.2

    # get the grid locations reshaped to be numCells x T x 2
    grid_location_x = df.pivot_table(
        index="cell", columns="t", values="grid_offset_x"
    ).to_numpy()
    grid_location_y = df.pivot_table(
        index="cell", columns="t", values="grid_offset_y"
    ).to_numpy()
    directions = np.stack([grid_location_x, grid_location_y], axis=-1)
    directions = directions.astype(np.int32)[:, ::skip_num, :]
    directions_delta = directions - directions[:, 0, :][:, np.newaxis, :]
    directions_filtered_x = directions[:, :, 0][
        directions_delta[:, :, 0] ** 2 + directions_delta[:, :, 1] ** 2 < R_cutoff**2
    ]
    directions_filtered_y = directions[:, :, 1][
        directions_delta[:, :, 0] ** 2 + directions_delta[:, :, 1] ** 2 < R_cutoff**2
    ]

    min_x = np.min(directions_filtered_x)
    min_y = np.min(directions_filtered_y)
    max_x = np.max(directions_filtered_x)
    max_y = np.max(directions_filtered_y)
    N = int(param_df["N"].iloc[0])
    size_x = int(max_x - min_x + N)
    size_y = int(max_y - min_y + N)
    cut_size = int(N // 8)
    size_vid_x = size_x - 2 * cut_size
    size_vid_y = size_y - 2 * cut_size

    if size_x * size_y > 5000 * 5000:
        print(f"Video is too large: {size_x*size_y}")
        return

    movie_path = phi_filename.parent / "movies"
    if not movie_path.exists():
        movie_path.mkdir()
    label_movie_filename = str(
        movie_path / phi_filename.with_suffix(".mp4").name
    ).replace("phi", "label")
    vidLabel = cv2.VideoWriter(
        label_movie_filename,
        cv2.VideoWriter_fourcc(*"avc1"),
        60,
        (size_vid_y, size_vid_x),
        isColor=True,
    )

    min_A, max_A, avg_A = get_min_max_avg_uncompressed(
        A_filename, phi_filename, T - 1, numCells, N
    )
    print(min_A, max_A)
    max_A = 0.9 * max_A

    background_color = np.array(hex2color("#FFFFFF"), dtype=np.uint8) * 255
    # norm = TwoSlopeNorm(vmin=min_A, vcenter=avg_A, vmax=max_A)
    # sm = ScalarMappable(norm=norm, cmap="bwr")

    # # Create a color palette centered on the average value of A
    # palette = sm.to_rgba(np.linspace(min_A, max_A, 256))[:, :3] * 255
    # palette = palette.astype(np.uint8)
    # palette = palette[:, np.newaxis, [2, 1, 0]]

    for t in trange(T - 1, desc=phi_filename.stem, dynamic_ncols=True):
        label = np.zeros((size_x, size_y), dtype=np.int32)
        A = np.zeros((size_x, size_y), dtype=np.float32)
        phi = np.zeros((size_x, size_y), dtype=np.float32)
        for j in range(numCells):
            current_r2 = (directions[j, t, 0] - directions[j, 0, 0]) ** 2 + (
                directions[j, t, 1] - directions[j, 0, 1]
            ) ** 2
            if current_r2 > R_cutoff**2:
                continue
            phi_cell = np.fromfile(
                phi_filename,
                dtype=np.float32,
                count=N * N,
                offset=t * numCells * N * N * 4 + j * N * N * 4,
            ).reshape(N, N)
            label_cell = (
                cv2.threshold(phi_cell, 0.5, 1, cv2.THRESH_BINARY)[1].astype(np.uint32)
                * j
            )
            A_cell = np.fromfile(
                A_filename,
                dtype=np.float32,
                count=N * N,
                offset=t * numCells * N * N * 4 + j * N * N * 4,
            ).reshape(N, N)

            current_min_x = directions[j, t, 0] - min_x
            current_min_y = directions[j, t, 1] - min_y

            label[
                current_min_x : current_min_x + N, current_min_y : current_min_y + N
            ] += label_cell
            A[current_min_x : current_min_x + N, current_min_y : current_min_y + N] += (
                A_cell * phi_cell
            )
            phi[
                current_min_x : current_min_x + N, current_min_y : current_min_y + N
            ] += phi_cell

        # squash A to be between min_A and max_A
        A = ((np.clip(A, min_A, max_A) - min_A) / (max_A - min_A) * 255).astype(
            np.uint8
        )
        # A = cv2.applyColorMap(A, palette).astype(np.uint8)
        phi = phi > 0.5
        phi = phi.astype(np.uint8)

        # # turn phi into a 3 channel image
        A_masked = cv2.bitwise_and(A, A, mask=phi)
        A_masked += cv2.bitwise_and(
            np.ones_like(A) * background_color,
            np.ones_like(A) * background_color,
            mask=1 - phi,
        )

        contours, hierarchy = cv2.findContours(
            label.astype(np.int32), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(A, contours, -1, (0, 0, 0), 1)
        vidLabel.write(A[cut_size:-cut_size, cut_size:-cut_size])

    vidLabel.release()


def get_wall_matrix(shape, radius):
    wall = np.zeros(shape)
    x = (np.arange(wall.shape[0]) - wall.shape[0] // 2) * 0.2
    X, Y = np.meshgrid(x, x)
    # draw the wall between r>20 r<21
    wall[np.logical_and(X**2 + Y**2 > radius**2, X**2 + Y**2 < (radius + 1) ** 2)] = 1
    return wall


import os, sys


def video_wrapper(args) -> None:
    phi_filename, video_mode, delete_files, label, chemotaxis, verbose = args
    phi_filename = Path(phi_filename)

    # replace only the last occurrence of phi with A
    A_filename = phi_filename.with_name(phi_filename.name.replace("phi", "A"))
    dynamics_filename = phi_filename.with_name(
        phi_filename.name.replace("phi", "dynamics")
    )
    if not dynamics_filename.exists():
        dynamics_filename = dynamics_filename.with_suffix(".csv")
    param_filename = phi_filename.with_name(
        phi_filename.name.replace("phi", "param")
    ).with_suffix(".txt")
    if not param_filename.exists():
        param_filename = param_filename.with_suffix(".toml")
    label_filename = phi_filename.with_name(phi_filename.name.replace("phi", "label"))

    if label:
        try:
            if video_mode == "condensed":
                create_label_video(
                    label_filename,
                    phi_filename,
                    A_filename,
                    dynamics_filename,
                    param_filename,
                    verbose,
                )
            if video_mode == "binary":
                # create_label_video_from_uncompressed(phi_filename, A_filename, dynamics_filename, param_filename)
                create_A_video(A_filename, param_filename)

        except Exception as e:
            # get line number of error
            import traceback

            print(traceback.format_exc())
            print(e)
            return

    C_filename = Path(str(phi_filename).replace("phi", "C"))
    if chemotaxis and C_filename.exists():
        try:
            create_chemotaxis_video(C_filename, param_filename)
        except:
            print(f"Error creating chemotaxis video for {C_filename}")

    if delete_files:
        A_filename.unlink()
        phi_filename.unlink()
        label_filename.unlink()
        if chemotaxis:
            C_filename.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Create videos from simulation results."
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="+",
        help="Directory containing the results files.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["condensed", "binary"],
        default="condensed",
        help="Mode of the simulation output.",
    )
    parser.add_argument(
        "-d", "--delete", action="store_true", help="Delete files after processing."
    )
    parser.add_argument(
        "-c", "--chemotaxis", action="store_true", help="Create chemotaxis video."
    )
    parser.add_argument(
        "-l", "--label", action="store_false", help="Create label video."
    )
    parser.add_argument(
        "-g", "--glob", type=str, default="", help="Glob pattern to match the files."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    for directory in args.directory:
        if directory.is_file():
            video_wrapper(
                (directory, args.mode, args.delete, args.label, args.chemotaxis)
            )
        elif directory.is_dir():
            if not directory.exists():
                raise FileNotFoundError(f"Directory {directory} does not exist")

            if args.glob == "":
                glob_string = "phi*.dat"
            else:
                glob_string = "phi*" + args.glob + "*.dat"

            phi_filenames = list(directory.glob(glob_string))
            for phi_filename in phi_filenames:
                video_wrapper(
                    (
                        phi_filename,
                        args.mode,
                        args.delete,
                        args.label,
                        args.chemotaxis,
                        args.verbose,
                    )
                )


import argparse

if __name__ == "__main__":
    main()
