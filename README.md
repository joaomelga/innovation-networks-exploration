# Nested Syndication Networks

This repository contains the complete data pipeline, analysis code, and documentation for a research article on **Nested Syndication Networks: Community Structure and Hierarchical Organization in Venture Capital Ecosystems**. The project analyzes the structural properties of VC syndication networks, particularly nestedness patterns and community structures, using data from France and the United States.

## 🎯 Project Overview

This research investigates the structure and evolution of venture capital syndication networks by:

- Analyzing bipartite networks between early-stage and late-stage investors
- Detecting community structures within VC syndication networks
- Measuring nestedness patterns and their statistical significance
- Conducting temporal analysis of network evolution
- Comparing patterns between French and US ecosystems
- Identifying nested hierarchical organization in Silicon Valley investor communities

## 📋 Table of Contents

- [Installation & Setup](#-installation--setup)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Documentation](#-documentation)
- [Technical Details](#️-technical-details)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

## 🚀 Installation & Setup

### Prerequisites

Before running this project, you need to have Python installed on your system.

#### Installing Python (Windows)

1. **Download Python**: Visit [python.org](https://www.python.org/downloads/) and download Python 3.8 or higher
2. **Run the installer**: Make sure to check "Add Python to PATH" during installation
3. **Verify installation**: Open Command Prompt and run:

   ```bash
   python --version
   ```

### Setting Up the Project

1. **Clone the repository**:

   ```bash
   git clone https://github.com/joaomelga/memoire.git
   cd memoire
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**:

   **On Windows (Git Bash/WSL):**

   ```bash
   source .venv/Scripts/activate
   ```

   **On Windows (Command Prompt):**

   ```cmd
   .venv\Scripts\activate.bat
   ```

   **On Windows (PowerShell):**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

### Verifying Installation

To verify everything is working correctly, open the main analysis notebook:

```bash
jupyter notebook src/main.ipynb
```

## 📁 Repository Structure

``` md
root/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── NOTEBOOKS.md        # Notebook organization guide
│
├── src/                # 🔧 Main analysis pipeline
│   ├── main.ipynb      # Primary analysis notebook
│   └── libs/           # Core analysis modules
│
├── data/               # 📊 Dataset storage
│   ├── raw/            # Original datasets
│   │   ├── france/     # French VC data
│   │   └── us/         # US VC data
│   └── processed/      # Clean, analysis-ready data
│       ├── france/
│       └── us/
│
├── experiments/        # 🔬 Research notebooks
│
├── reports/            # 📈 Output and documentation
│   ├── figures/        # Generated visualizations
│   └── texts/          # Written analysis
│
└── docs/               # 📚 Reference materials
    ├── literature/     # Academic papers
    └── presentations/  # Project presentations
```

## 🎬 Getting Started

### Quick Start - Running the Main Analysis

1. **Navigate to the main analysis notebook**:

   ```bash
   jupyter notebook src/main.ipynb
   ```

2. **Configure analysis parameters** (in the notebook):

    Adjust main analysis parameters in `src/main.ipynb` as needed:
  
    ```python
    # Data generation
    GENERATE_CLEAN_DATA = False    # Set True for first run
    CALCULATE_COMMUNITIES = False  # Recalculate communities  
    CALCULATE_COMMUNITIES_NESTEDNESS = False  # Recalculate nestedness

    # Visualization  
    PLOT_KKL = False              # Generate network layout plots (expensive)
    ```

3. **Run the complete analysis**:
   - Execute all cells in order
   - The notebook will automatically generate processed data, perform network analysis, and create visualizations

### Understanding the Main Workflow

The primary analysis follows this workflow:

1. **Data Cleaning** → Clean raw investment data
2. **Network Construction** → Build bipartite investor networks  
3. **Community Detection** → Identify investor communities
4. **Nestedness Analysis** → Calculate nestedness with null models
5. **Temporal Analysis** → Study network evolution over time
6. **Visualization** → Generate publication-ready figures

## 📚 Documentation

### Academic References

Key literature in `docs/literature/`:

- **Borgatti & Halgin (2011)** - Network theory foundations
- **Granovetter (2012)** - Economic action and social structure  
- **Mariani (2019)** - Nestedness in complex networks
- **Dalle et al. (2025)** - Accelerator-mediated access to investors

### Code Documentation

All analysis modules include comprehensive docstrings:

- `src/libs/data_cleaning.py` - Data preprocessing functions
- `src/libs/network_analysis.py` - Network construction and metrics
- `src/libs/nestedsness_calculator.py` - Nestedness computation algorithms
- `src/libs/curveball.py` - Null model generation for statistical testing

## 🛠️ Technical Details

### Dependencies

Key Python packages (see `requirements.txt` for complete list):

- **Data Analysis**: `pandas`, `numpy`, `scipy`
- **Network Analysis**: `networkx`
- **Statistical Testing**: `scipy.stats`
- **Visualization**: `matplotlib`, `seaborn`
- **Jupyter**: `jupyter`, `ipywidgets`

### Performance Considerations

- **Memory Usage**: Large networks require significant RAM
- **Computation Time**: Nestedness null models are computationally intensive
- **Parallel Processing**: Some analyses support parallel computation

## 🤝 Contributing

### Research Extensions

Potential areas for extension:

- Additional geographic regions
- Different network construction methods  
- Alternative community detection algorithms
- Extended temporal analysis
- Sector-specific studies

### Code Contributions

1. Fork the repository
2. Create a feature branch
3. Add comprehensive documentation
4. Include unit tests where applicable
5. Submit a pull request

### Data Contributions

- Additional regional datasets
- Updated time series data
- Alternative data sources
- Data quality improvements

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{melga2025vcnetworks,
  title={Nested Syndication Networks: Community Structure and Hierarchical Organization in Venture Capital Ecosystems},
  author={Melga, João},
  year={2025},
  journal={[not yet published...]},
  note={Working Paper}
}
```

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This code is provided for academic and research purposes. If you use this work in your research, please cite the article and respect the data usage guidelines outlined in the license.

---

*This documentation is actively maintained. For the most current information, please refer to the repository's latest commit.*
