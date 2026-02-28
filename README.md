# [DFLER: Drone Flight Log Entity Recognizer](https://www.sciencedirect.com/science/article/pii/S2665963822001415)

DFLER is an open-source tool developed to perform named entity recognition on drone flight log data to support forensic investigations. It uses a fine-tuned BERT model to identify entities (like Actions, Components, Issues) in log messages and constructs a forensic timeline.

## Features
- **Forensic Timeline Construction**: Merges and sorts logs from Android and iOS devices.
- **Entity Recognition**: Uses a BERT model to highlight key entities in log messages.
- **Forensic Report Generation**: Generates a PDF report with statistics and the highlighted timeline.

## Installation

### Prerequisites
1.  **Python 3.7+**
2.  **wkhtmltopdf**: Required for report generation.
    *   **Windows**: Download and install from [wkhtmltopdf.org](https://wkhtmltopdf.org/downloads.html/). Default path: `C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe`.
    *   **Linux**: `sudo apt-get install wkhtmltopdf`. Default path: `/usr/bin/wkhtmltopdf`.

### Install DFLER
You can install DFLER directly from PyPI (coming soon) or from source.

**From Source:**
```bash
git clone https://github.com/DroneNLP/dfler.git
cd dfler
pip install .
```

### Model
DFLER uses a fine-tuned BERT model (`swardiantara/droner`). By default, the tool will **automatically download and cache** the model from Hugging Face on its first run.

If you prefer to use a locally downloaded model or an offline environment, you can point to your local directory containing `pytorch_model.bin` using the `--model` argument.

## Usage

DFLER provides a Command Line Interface (CLI).

**Structure:**
```bash
dfler [arguments]
```

### Arguments

| Argument | Description |
| :--- | :--- |
| `--evidence <path>` | Path to the directory containing input flight logs (must have `android` and/or `ios` subfolders). |
| `--output <path>` | Path to the directory where results will be saved. |
| `--model <path>` | Path to the directory containing the BERT model (`pytorch_model.bin`). |
| `--config <path>` | Path to a custom `config.json` file. |

### Examples

**1. Run Pipeline (Recommended):**
```bash
dfler --evidence "./flight_logs" --output "./results/case_001"
```

**2. Run Pipeline with Custom Model:**
```bash
dfler --evidence "./flight_logs" --output "./results/case_001" --model "./local_model_dir"
```

## Output Structure

Running `dfler` will create the following structure in your output directory:

```text
output_dir/
├── raw_list.json             # List of detected log files
├── forensic_timeline.csv     # Merged and sorted timeline of all logs
├── ner_result.json           # Timeline with detected entities (JSON format)
├── statistics.json           # Counts of detected entity types
├── forensic_report_.html     # Intermediate HTML report
├── forensic_report_.pdf      # Final PDF Forensic Report
└── parsed/                   # Intermediate parsed CSVs
    ├── android/
    └── ios/
```

## Configuration

You can optionally use a `config.json` file instead of passing long arguments.

**Example `config.json`:**
```json
{
    "source_evidence": "./flight_logs",
    "output_dir": "./results/test_run",
    "model_dir": "./model",
    "wkhtml_path": {
        "windows": "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe",
        "linux": "/usr/bin/wkhtmltopdf"
    },
    "use_cuda": true
}
```

Run with config:
```bash
dfler --config my_config.json
```

## License
[MIT](LICENSE)
