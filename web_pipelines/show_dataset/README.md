# Dataset Visualization Tool

A web-based visualization tool for exploring images with segmentation masks and bounding boxes from various datasets. This tool provides an interactive interface to view and analyze dataset samples with their associated metadata.

## Features

- **Interactive Web Interface**: Clean, responsive web interface for browsing dataset samples
- **Multi-View Visualization**: Display original images, segmentation masks, bounding boxes, and applied masks
- **Metadata Display**: Show classification labels, predictions, and other relevant metadata
- **Random Sampling**: Generate new random samples from the dataset with a single click
- **Multiple Dataset Support**: Compatible with various datasets including Urban Cars, Waterbirds, ImageNet-D, ImageNet-9, and Counter Animal datasets
- **Flexible Configuration**: Support for custom dataset configurations and parameters

## Prerequisites

Before using this tool, ensure you have:

1. **Python 3.10+** installed
2. **Required dependencies** installed (see Installation section)
3. **Dataset files** in the expected format (parquet files with segmentation data)
4. **Label converter files** (`.pt` files) for proper class name mapping

## Installation

1. Install the main project dependencies:
```bash
pip install -r requirements.txt
```

2. The tool requires the following key dependencies:
   - Flask (for web interface)
   - PyTorch (for data loading)
   - Matplotlib (for image visualization)
   - NumPy (for array operations)

## Usage

### Basic Usage

1. **Start the web application**:
```bash
cd web_pipelines/show_dataset
python app.py --config_path ../../configs/show_dataset/data_config.yaml --path_within_config data/dataset_configs/bboxed_dataset_urban_cars
```

2. **Open your web browser** and navigate to `http://localhost:5000`

3. **View the dataset samples** - The interface will display random samples from your dataset

4. **Generate new samples** - Click the "Sample Other Images" button to get new random samples

### Advanced Usage

#### Command Line Options

```bash
python app.py --help
```

**Required Arguments:**
- `--config_path`: Path to the YAML configuration file containing dataset descriptions. Example config is here: `./configs/show_dataset/data_config.yaml`.
- `--path_within_config`: Path to the specific dataset configuration within the config file (e.g., `data/dataset_configs/bboxed_dataset_urban_cars`)

**Optional Arguments:**
- `--images_list`: Comma-separated list of specific image names to show (instead of the full name you can pass substrings of full names)
- `--split`: Data split to show (default: "train")
- `--n_images`: Number of images to display (default: 10)
- `--data_kwargs`: Additional keyword arguments for data configuration (format: `key1=value1;key2=value2`)
- `--debug`: Run in debug mode

#### Example Commands

**View Urban Cars dataset:**
```bash
python app.py --config_path ../../configs/show_dataset/data_config.yaml --path_within_config data/dataset_configs/bboxed_dataset_urban_cars
```

**View specific number of images:**
```bash
python app.py --config_path ../../configs/show_dataset/data_config.yaml --n_images 20
```

**View specific images:**
```bash
python app.py --config_path ../../configs/show_dataset/data_config.yaml --images_list "image1.jpg,image2.jpg,image3.jpg"
```

**View with custom parameters:**
```bash
python app.py --config_path ../../configs/show_dataset/data_config.yaml --data_kwargs "visualization_mode=unicorn;extended_output=true"
```

## Configuration

### Dataset Configuration

The tool uses YAML configuration files to define datasets. Each dataset configuration should include:

```yaml
dataset_name:
  train_transform: null
  eval_transform:
  # Transform configuration
  train_val_split: 0.0
  csv_path: "./path/to/dataset.parquet"
  dataset_task: "classification"
  label_converter: "./path/to/label_converter.pt"
  foreground_keyword: "optional_keyword"
```

### Supported Datasets

The tool supports several pre-configured datasets:

1. **Urban Cars** (`bboxed_dataset_urban_cars`)
2. **Waterbirds Group 2** (`bboxed_dataset_wb_group_2`) - Waterbirds on land background
3. **Waterbirds Group 3** (`bboxed_dataset_wb_group_3`) - Waterbirds on water background
4. **ImageNet-D** (`bboxed_dataset_in_d`)
5. **ImageNet-9** (`bboxed_dataset_in_9`)
6. **Counter Animal** (`bboxed_dataset_counter`)

## Visualization Modes

### Standard Mode (Default)
Displays:
- Original image with class information
- Segmentation mask
- Applied mask with metadata
- Bounding box
- All masks (if available)
- Overlay of bounding box on mask

### Unicorn Mode
A simplified view showing:
- Original image
- Segmentation mask
- Applied mask with metadata
- All masks (if available)

To use unicorn mode, add `visualization_mode: unicorn` to your dataset configuration.

## Output Format

Each displayed image shows multiple views in a grid layout:

1. **Original Image**: The source image with class label and file path
2. **Segmentation Mask**: The segmentation mask for the main object
3. **Applied Mask**: The mask applied to the image with metadata (predictions, scores, etc.)
4. **Bounding Box**: The bounding box annotation
5. **All Masks**: Combined view of all available masks (if multiple masks exist)
6. **Overlay**: Bounding box overlaid on the segmentation mask

## File Structure

```
web_pipelines/show_dataset/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface template
├── static/
│   └── images/           # Generated visualization images (auto-created)
└── README.md            # This file
```

## Troubleshooting

### Common Issues

1. **"Dataset is empty" error**: Ensure your dataset files exist and contain valid data
2. **Label converter not found**: Verify the path to your `.pt` label converter files
3. **Images not displaying**: Check that the `static/images/` directory is writable
4. **Port already in use**: The default port is 5000. If busy, Flask will automatically use the next available port

### Debug Mode

Run with `--debug` flag to enable Flask's debug mode:
```bash
python app.py --config_path ../../configs/show_dataset/data_config.yaml --debug
```

## Technical Details

- **Framework**: Flask web application
- **Image Processing**: Matplotlib for visualization, PyTorch for data loading
- **Data Format**: Supports parquet files with extended bboxed dataset format
- **Image Storage**: Generated visualizations are temporarily stored in `static/images/`
- **Sampling**: Uses uniform random sampling to select dataset items

