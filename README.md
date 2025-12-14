# Nested Syndication Networks

This repository was initially created to store and share the complete data pipeline, analysis code, and documentation for my master's thesis: **Nested Syndication Networks: Community Structure and Hierarchical Organization in Venture Capital Ecosystems**.

📄 **[Read the full paper](Melga%20-%202025%20-%20Nested%20Investor%20Syndication%20Networks.pdf)**

In this research, I explore the hypothesis that **nestedness** - a network property typically observed in biological and ecological systems - might also emerge in venture capital syndication networks.

I was genuinely surprised to discover that real-world investment networks exhibit organizational patterns strikingly similar to those found in nature, such as mutualistic ecosystems where species interact in hierarchical, nested structures.

As a result, this repository has evolved beyond the original thesis work to include additional experiments and analyses that explore related network phenomena in investment ecosystems. Some of these ongoing investigations may lead to future publications.

The project analyzes the structural properties of VC syndication networks, particularly nestedness patterns and community structures, using comprehensive data from France and the United States.

## 🎯 Project Overview

This research investigates the structure and evolution of venture capital syndication networks by:

- Analyzing bipartite networks between early-stage and late-stage investors
- Detecting community structures within VC syndication networks
- Measuring nestedness patterns and their statistical significance
- Conducting temporal analysis of network evolution
- Comparing patterns between communities
- Special attention for Silicon Valley investor communities

## 📋 Table of Contents

- [Installation & Setup](#-installation--setup)
- [Repository Structure](#-repository-structure)
- [Data Sources](#-data-sources)
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
│   ├── thesis_fr.ipynb # Final French data analysis
│   ├── thesis_us.ipynb # Final US data analysis
│   ├── experiments/    # 🔬 Research notebooks and exploratory analysis
│   └── libs/           # Core analysis modules
│
├── data/               # 📊 Dataset storage
│   ├── raw/            # Original datasets (not included - see Data Availability below)
│   └── processed/      # Clean, analysis-ready data
│
├── reports/            # 📈 Output and documentation
│   ├── article/        # LaTeX article files
│   ├── figures/        # Generated visualizations
│   └── drafts/         # Draft documents and comparisons
│
└── references/         # 📚 Reference materials
    └── *.pdf           # Academic papers and literature
```

## 📊 Data Sources

**Raw Data**: The original datasets got from CrunchBase are not included in this repository due to their large size. The raw data includes comprehensive venture capital investment records from France and the United States. I can make it available upon request.

**Processed Data**: Clean, analysis-ready datasets are included in the `data/processed/` directory and are sufficient to reproduce all analyses and results.

## 🎬 Getting Started

### Running the Thesis Analysis

1. **Navigate to the thesis analysis notebook**:

   For US data analysis:
   ```bash
   jupyter notebook src/thesis_us.ipynb
   ```

   For French data analysis:
   ```bash
   jupyter notebook src/thesis_fr.ipynb
   ```

2. **Configure analysis parameters** (in the notebook):

    Adjust main analysis parameters in `src/thesis_us.ipynb` (or `src/thesis_fr.ipynb`) as needed:
  
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
   - All plots and figures are automatically saved to `reports/figures/` and used in the LaTeX article files
   - Running/updating the thesis notebooks automatically updates all figures used in the research article

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
