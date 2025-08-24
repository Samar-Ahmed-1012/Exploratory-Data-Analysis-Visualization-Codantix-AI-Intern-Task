Exploratory Data Analysis & Visualization: Global Healthcare System Performance
https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/Pandas-1.3%252B-orange
https://img.shields.io/badge/Matplotlib-3.4%252B-blueviolet
https://img.shields.io/badge/Seaborn-0.11%252B-red
https://img.shields.io/badge/License-MIT-green

A comprehensive data analysis project examining healthcare system performance across 127 countries worldwide. This project includes data cleaning, exploratory analysis, visualization, and strategic insights for public health policy improvement.

📊 Project Overview
This project analyzes key healthcare metrics including:

Healthcare expenditure (% of GDP)
Doctor density per 1,000 people
Universal healthcare coverage rates
Life expectancy and infant mortality rates
Vaccination rates and healthcare quality indices

📁 Dataset Information
Source: Kaggle (Global Healthcare System Performance Dataset)
Original Size: 627 records, 18 features
Cleaned Size: 127 unique countries

Key Features:
Regional classification
Income levels (High, Upper-Middle, Lower-Middle, Low)
Healthcare system types (Single-payer, Multi-payer, Mixed)
Performance metrics across accessibility, quality, and outcomes

🛠️ Technical Implementation
Data Cleaning Process
Removed 500 duplicate entries
Handled missing values using region-wise mean imputation
Corrected data types for all columns
Prepared categorical data for encoding

Technologies Used
Python 3.8+
Pandas for data manipulation
NumPy for numerical operations
Matplotlib & Seaborn for visualizations
Scikit-learn for simple imputation

Key Analysis Steps
Loaded and inspected the dataset
Performed missing value analysis and imputation
Conducted statistical analysis of healthcare metrics
Created regional comparisons
Generated correlation analysis
Developed strategic insights and recommendations

📈 Key Findings
Regional Disparities
Western Europe: Highest doctor density (4.76/1,000) and coverage (95.5%)
Africa: Lowest doctor density (1.48/1,000) and coverage (50.6%)
45-point gap in universal healthcare coverage between Western Europe and Africa

Expenditure vs Outcomes
High-income countries show diminishing returns on healthcare spending
Several nations achieve top-tier outcomes with moderate spending (efficiency exemplars)
North America spends more than Western Europe but has lower life expectancy

Critical Correlations
+0.75 correlation between healthcare expenditure and quality index
-0.70 correlation between infant mortality and life expectancy
+0.65 correlation between doctor density and vaccination rates

📸 Visualizations
The project includes 8 comprehensive visualizations:
Bar Chart: Average Life Expectancy by Region
Line Graph: Healthcare Expenditure vs Life Expectancy
Histogram: Distribution of Doctors per 1,000 People
Heatmap: Correlation Matrix of Healthcare Indicators
Pie Chart: Region-wise Distribution of Countries
Boxplot: Infant Mortality Rate by Region
Scatter Plot: Doctors vs Healthcare Coverage (colored by Life Expectancy)
Pairplot: Healthcare Indicators Relationship Analysis by Region

🎯 Strategic Insights
1. Investment in Medical Workforce Yields Highest Return
Each additional doctor per 1,000 people is associated with increased life expectancy and better health outcomes.

2. Significant Inefficiency in Healthcare Spending
How money is spent (prevention, primary care, workforce) is more important than how much is spent.

3. The 70% Universal Coverage Threshold
Countries below ~70% universal health coverage see exponentially worse outcomes for infant mortality and life expectancy.

💡 Actionable Recommendations
Prioritize Medical Workforce Development: Target regions with doctor density below 2.5 per 1,000 people
Bridge the Coverage Gap: Implement policies for countries below the 70% universal coverage threshold
Adopt Best Practices: Facilitate knowledge transfer from efficient regions (e.g., Western Europe) to high-spending, lower-performing regions

Expected Impact
Short-Term (1-3 yrs): Reduce infant mortality rates by up to 20% in target regions
Mid-Term (3-5 yrs): Increase average life expectancy by 3-5 years in lowest-performing regions
Long-Term (5-10 yrs): Create more resilient health systems, saving billions in avoidable costs

🚀 How to Use This Project
Prerequisites
Python 3.8 or higher
Jupyter Notebook (optional)
Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

Installation
Clone the repository:

bash
git clone https://github.com/Samar-Ahmed-1012/Exploratory-Data-Analysis-Visualization-Codantix-AI-Intern-Task.git
Install required packages:

bash
pip install pandas numpy matplotlib seaborn scikit-learn
Run the main analysis script:

bash
python main.py
File Structure
text
├── main.py                     # Main analysis script
├── Healthcare_System_Performance_Dataset_CLEANED.csv      # Original cleaned dataset
├── Healthcare_System_Performance_Dataset_FINAL.csv        # Final dataset after imputation
├── 1_Life_Expectancy_by_Region.png                       # Visualization outputs
├── 2_Expenditure_vs_LifeExpectancy.png
├── ... (7 more visualization files)
└── README.md                   # Project documentation
📋 Future Enhancements
Interactive dashboard using Plotly or Tableau
Machine learning models to predict healthcare outcomes
Time-series analysis of healthcare metrics
Integration with additional data sources (economic, demographic)

👨‍💻 Developer
Samar Ahmed
GitHub: Samar-Ahmed-1012
Project completed as part of the Codantix AI Internship Program


🙏 Acknowledgments
Kaggle for providing the dataset
Codantix AI for the internship opportunity
Python community for excellent data science libraries

Note: This project was created for educational and analytical purposes. The insights and recommendations are based on statistical analysis of the provided dataset and should be validated with domain experts before implementation in real-world policy decisions.
