import cv2 as cv
import numpy as np
from multiprocessing import Pool
import click
from pathlib import Path
import logging


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Calculate the rotation matrix
    M = cv.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the bounding box of the rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # New bounding dimensions
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account the translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation and return the image
    rotated_image = cv.warpAffine(image, M, (new_w, new_h))
    return rotated_image


def worker(task):
    img_gray, template_original, angle, scales, confidence = task
    print("Processing task with angle {} and scales {}".format(angle, scales))
    detections = []
    for scale in scales:
        template_copy = template_original.copy()
        # Rotate and scale
        rotated_template = rotate_image(template_copy, angle)
        scaled_template = cv.resize(rotated_template, None, fx=scale, fy=scale)
        w, h = scaled_template.shape[::-1]

        # Template matching
        res = cv.matchTemplate(img_gray, scaled_template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= confidence)

        # Store detections
        for pt in zip(*loc[::-1]):
            detections.append((pt, (pt[0] + w, pt[1] + h), res[pt[::-1]]))

    return detections


def multi_scale_template_matching_parallel(
    img_rgb, template, scales, angles, confidence, num_processes
):
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template_original = template.copy()

    # Prepare tasks for multiprocessing
    tasks = []

    for angle in angles:
        # Split scales among processes
        split_scales = np.array_split(scales, num_processes)
        for split_scale in split_scales:
            tasks.append((img_gray, template_original, angle, split_scale, confidence))

    # Process tasks in parallel
    with Pool(num_processes) as p:
        results = p.map(worker, tasks)

    # Combine results from all processes
    detections = [item for sublist in results for item in sublist]
    return detections


def cluster_boxes(detections, center_threshold=10, size_threshold=0.2):
    # Convert the bounding boxes to the format (x1, y1, x2, y2)
    boxes = np.array([[x1, y1, x2, y2] for ((x1, y1), (x2, y2), _) in detections])
    # Calculate the center and size of each box
    centers = np.array(
        [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes]
    )
    sizes = np.array([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes])

    # Initialize clusters
    clusters = []
    for i, box in enumerate(boxes):
        added_to_cluster = False
        for cluster in clusters:
            for c_idx in cluster:
                # Calculate the center distance and size ratio
                center_distance = np.linalg.norm(centers[i] - centers[c_idx])
                size_ratio = min(sizes[i], sizes[c_idx]) / max(sizes[i], sizes[c_idx])

                # Check if the current box is close enough to be in the same cluster
                if center_distance < center_threshold and size_ratio > (
                    1 - size_threshold
                ):
                    cluster.append(i)
                    added_to_cluster = True
                    break

            if added_to_cluster:
                break

        if not added_to_cluster:
            clusters.append([i])

    # Merge boxes within each cluster
    merged_boxes = []
    for cluster in clusters:
        # Calculate the mean coordinates of the boxes in the cluster
        cluster_boxes = boxes[cluster]
        x1, y1 = np.mean(cluster_boxes[:, :2], axis=0)
        x2, y2 = np.mean(cluster_boxes[:, 2:], axis=0)
        merged_boxes.append(((int(x1), int(y1)), (int(x2), int(y2))))

    return merged_boxes


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--scale-start", default=0.5, help="Start of scale range.")
@click.option("--scale-end", default=5.1, help="End of scale range.")
@click.option("--scale-step", default=0.1, help="Step for scale range.")
@click.option("--angle-start", default=0, help="Start of angle range.")
@click.option("--angle-end", default=360, help="End of angle range.")
@click.option("--angle-step", default=90, help="Step for angle range.")
@click.option(
    "--confidence", default=0.8, help="Confidence threshold for template matching."
)
@click.option(
    "--num-processes", default=6, help="Number of processes for parallel execution."
)
@click.option(
    "--template",
    default="template.png",
    type=click.Path(exists=True),
    help="Path to template image.",
)
@click.option(
    "--output-dir",
    default=".",
    type=click.Path(exists=True),
    help="Path to output directory.",
)
def main(
    path,
    scale_start,
    scale_end,
    scale_step,
    angle_start,
    angle_end,
    angle_step,
    confidence,
    num_processes,
    template,
    output_dir,
):
    path = Path(path)
    output_dir = Path(output_dir)

    template = cv.imread(
        template, cv.IMREAD_GRAYSCALE
    )  # Replace 'template.png' with your template image path
    assert (
        template is not None
    ), "Template file could not be read, check with os.path.exists()"

    if path.is_dir():
        # ignore hidden files
        image_paths = [
            f for f in path.iterdir() if f.is_file() and not f.name.startswith(".")
        ]
    else:
        image_paths = [path]

    for image_path in image_paths:
        try:
            img_rgb = cv.imread(str(image_path))
            assert (
                img_rgb is not None
            ), f"File {image_path} could not be read, check with os.path.exists()"

            scales = np.arange(scale_start, scale_end, scale_step)
            angles = np.arange(angle_start, angle_end, angle_step)

            detections = multi_scale_template_matching_parallel(
                img_rgb, template, scales, angles, confidence, num_processes
            )

            clustered_detections = cluster_boxes(detections)
            print(f"detected {len(clustered_detections)} objects in {image_path}")

            # Draw rectangles for detections
            for detection in clustered_detections:
                cv.rectangle(img_rgb, detection[0], detection[1], (0, 0, 255), 2)

            output_path = output_dir / f"result_{image_path.name}"
            cv.imwrite(str(output_path), img_rgb)
            # create file with every line being a tuple of (x1, y1, x2, y2)
            with open(str(output_dir / f"result_{image_path.name}.txt"), "w") as f:
                for detection in clustered_detections:
                    f.write(
                        f"{detection[0][0]},{detection[0][1]},{detection[1][0]},{detection[1][1]}\n"
                    )
            print(
                f"Processed image saved as {output_path} and bounding boxes saved as {output_path}.txt"
            )
        except Exception as e:
            print(f"Error while processing {image_path}: {e}")
            logging.exception(f"Error while processing {image_path}: {e}")


if __name__ == "__main__":
    main()
