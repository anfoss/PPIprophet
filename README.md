# PPIprophet: Protein-Protein Interaction Prediction Suite

PPIprophet is a Python tool for predicting protein-protein interactions (PPIs) from proteomics data. It supports a variety of quantitation schemes and provides results compatible with Cytoscape for network analysis. This README covers installation, input formats, parameter configuration, running the tool, and interpreting results.

---

## Table of Contents
- [PPIprophet: Protein-Protein Interaction Prediction Suite](#ppiprophet-protein-protein-interaction-prediction-suite)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Input File Format](#input-file-format)
  - [Experimental Information File](#experimental-information-file)
  - [Parameter Configuration](#parameter-configuration)
  - [Running PPIprophet](#running-ppiprophet)
  - [Output Files](#output-files)
  - [Importing Results into Cytoscape](#importing-results-into-cytoscape)
  - [Troubleshooting](#troubleshooting)
  - [Contact](#contact)

---

## Overview
PPIprophet predicts protein-protein interactions from proteomics data using deep learning and network analysis. It is highly configurable and supports various input formats and quantitation schemes (MS1/MS2 XIC, spectral counts, TMT, SILAC, etc.).

PPIprophet is designed for co-fractionation MS experiments and large-scale PPI prediction.

---

## Installation
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

---

## Input File Format
Input files must be wide-format matrices with these requirements:
- **First column:** `GN` (Gene name or protein ID, unique for each row)
- **Remaining columns:** Quantitation values, ordered by fractionation scheme (column names are flexible)

**Note:** Duplicate rows in the `GN` column will trigger an error.

Example:

| GN | Fraction1 | Fraction2 | ... |
|----|-----------|-----------|-----|
| A  | 20400     | ...       |     |
| D  | 1230120   | ...       |     |
| C  | 1230120   | ...       |     |

All common proteomics quantitation schemes are supported. Example files are provided in `test/test_fract.txt`.

---

## Experimental Information File
The sample IDs file (e.g., `sample_ids.txt`) must have the following columns:

| Sample              | cond   | group | short_id   | repl | fr |
|---------------------|--------|-------|------------|------|----|
| ./Input/c1r1.txt    | Ctrl   | 1     | ipsc_2i_1  | 1    | 65 |
| ./Input/c1r2.txt    | Ctrl   | 1     | ipsc_2i_2  | 2    | 64 |
| ./Input/c1r3.txt    | Ctrl   | 1     | ipsc_2i_3  | 3    | 65 |
| ./Input/c2r1.txt    | Treat1 | 2     | ipsc_ra_1  | 1    | 65 |
| ./Input/c2r2.txt    | Treat1 | 2     | ipsc_ra_2  | 2    | 65 |
| ./Input/c2r3.txt    | Treat1 | 2     | ipsc_ra_3  | 3    | 65 |

- **Sample:** Full path to the file to process (must match the actual filename)
- **cond:** Condition name (`Ctrl`, `Treat1`, `Treat2`, etc.)
- **group:** Group number (1 for control)
- **short_id:** Alternative ID
- **repl:** Replicate number
- **fr:** Number of fractions per file

**Note:**
- The `Sample` column must match the input file names exactly (including extension).
- For multiple conditions, use `Ctrl`, `Treat1`, `Treat2`, etc. as labels.

---

## Parameter Configuration
Parameters can be set via the `ProphetConfig.conf` file or directly on the command line. Command-line parameters override the config file and are saved to it.

| Argument   | Description                                       | Default           |
| ---------- | ------------------------------------------------- | ----------------- |
| `-db`      | Path to PPI network file (STRING format)           | `None`            |
| `-fdr`     | Global FDR threshold                              | `0.3`             |
| `-sid`     | Path to sample IDs file                           | `sample_ids.txt`  |
| `-out`     | Output folder name                                | `Output`          |
| `-crapome` | Crapome contaminant reference file (optional)     | `crapome.org.txt` |
| `-thresh`  | Frequency threshold for Crapome filtering         | `0.5`             |
| `-skip`    | Skip preprocessing/feature generation (`True`/`False`) | `False`       |

To see all available parameters:

```sh
python3 main.py --help
```

---

## Running PPIprophet
After installing dependencies, you can test PPIprophet with the example dataset:

```sh
python3 main.py -sid test/test_ids.txt
```

To run with all default settings:

```sh
python3 main.py
```

To specify parameters, use command-line arguments. For example, to use a custom interaction network and a 10% FDR:

```sh
python3 main.py -db myppi.txt -fdr 0.1
```

---

## Output Files
PPIprophet generates two main folders:
- `tmp/`: Intermediate files for debugging/validation (can be deleted after completion)
- `Output/`: Final results, including:
  - `adj_list.txt`: PPI list (proteinA/proteinB/Probability) and Crapome frequencies
  - `communities.txt`: Modules detected after clustering
  - `d_scores.txt`: Interaction probabilities as modified WD scores
  - `probtot.txt`: Adjacency list with filtered interactions (FDR applied)
  - `prot_centr.txt`: Protein-centric interactors list

File names may vary slightly depending on the run (see output folder for details).

---

## Importing Results into Cytoscape
- `probtot.txt` and `d_scores.txt` can be imported directly into Cytoscape.
- `adj_list.txt` can be used to visualize interaction frequencies.

---

## Troubleshooting
Common errors and solutions:
- **NaRowError / NaInMatrixError:** 'NA' values in input; replace with 0.
- **MissingColumnError:** Required columns (`GN` or `ID`) missing.
- **DuplicateRowError / DuplicateIdentifierError:** Duplicate entries in `GN` column; ensure uniqueness.
- **EmptyColumnError:** A column contains only 'NA'.

*Tip:* If files have different numbers of fractions, add zero-filled columns as needed.

---

## Contact
For questions or support, see the [project README on GitHub](https://github.com/fossatiA/PPIprophet/blob/master/README.md).