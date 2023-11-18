# imgmatch: Image Template Matching Tool

`imgmatch` is a command-line interface (CLI) tool designed for efficient template matching in images, making it a great asset for computer vision, digital forensics, and automated image analysis. It offers a streamlined approach to detect and analyze templates in a variety of image sizes and formats.

## Key Features

- **Multi-Scale Template Matching**: Detects templates at multiple scales, providing thorough matching across diverse image dimensions.
- **Rotation and Flip Detection**: Capable of recognizing templates that are rotated or flipped, enhancing its effectiveness in complex imaging scenarios.
- **Parallel Processing**: Leverages multi-processing for speedy template matching, especially useful for processing large image datasets.
- **Customizable Search Parameters**: Allows adjustments of scale range, rotation angles, and confidence thresholds to meet specific requirements.
- **User-Friendly Interface**: Offers a simple and intuitive CLI, making it accessible for both beginners and experienced users. Clear documentation ensures ease of use.
- **Python Integration**: Built using Python and popular libraries like OpenCV and NumPy, ensuring robust and reliable performance.

## Ideal Use Cases

- Suitable for computer vision professionals and enthusiasts.
- Useful for researchers and students in digital image processing and related fields.
- A tool for practitioners in digital forensics and content authentication.
- Applicable for automated quality inspection in manufacturing and industrial environments.

## Getting Started

### Installation

To install `imgmatch`, use the following command:

```bash
pipx install imgmatch

```

### Usage
Run imgmatch on your desired image or directory of images:

```bash
imgmatch /path/to/image/or/directory --scale-start 0.5 --scale-end 2.1 --confidence 0.8 --num-processes 4
```

### Customizing Parameters
You can customize the search parameters to fit your specific needs. Here are some of the options you can adjust:

- --scale-start: Starting scale (default is 0.5).
- --scale-end: Ending scale (default is 5.1).
- --confidence: Confidence threshold for template matching (default is 0.8).
- --num-processes: Number of processes for parallel execution (default is 6).
- --angle-start: Starting angle for rotation (default is 0).
- --angle-end: Ending angle for rotation (default is 360).
- --angle-step: Angle step for rotation (default is 90).
- --template: Path to template image (default is None).
- --output-dir: Path to output directory (default is current directory).
### Contributing
Contributions to imgmatch are welcome! Whether it involves fixing bugs, improving documentation, or suggesting new features, we value your input.

### License
imgmatch is released under the MIT License.
