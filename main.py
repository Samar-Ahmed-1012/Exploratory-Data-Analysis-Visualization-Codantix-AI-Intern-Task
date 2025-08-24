import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer

# Dataset load karna
dataset_path = r"D:\Public Health & Countries\dataset"
file_name = "Healthcare_System_Performance_Dataset_CLEANED.csv"
file_path = os.path.join(dataset_path, file_name)
df = pd.read_csv(file_path)

print("Missing values BEFORE imputation:")
print(df.isnull().sum())

# Missing values ke statistics
print("\nMissing values statistics:")
missing_cols = ['Doctors per 1,000 People', 'Universal Healthcare Coverage (%)']
for col in missing_cols:
    if df[col].isnull().sum() > 0:
        print(f"\n{col}:")
        print(f"Missing values: {df[col].isnull().sum()}")
        print(f"Mean: {df[col].mean():.2f}")
        print(f"Median: {df[col].median():.2f}")
        print(f"Min: {df[col].min():.2f}")
        print(f"Max: {df[col].max():.2f}")

# Missing values ko fill karna
print("\n=== MISSING VALUES IMPUTATION ===")

# Strategy 3: Region-wise mean se fill karna (most accurate)
df_final = df.copy()
for col in missing_cols:
    for region in df_final['Region'].unique():
        region_mean = df_final[df_final['Region'] == region][col].mean()
        df_final.loc[(df_final[col].isnull()) & (df_final['Region'] == region), col] = region_mean

# Final check
print("\nMissing values AFTER imputation:")
print(df_final.isnull().sum())

# Final statistics
print("\nFinal dataset statistics:")
print(f"Total countries: {len(df_final)}")
print(f"Total columns: {len(df_final.columns)}")

# Final dataset save karna
final_file_path = os.path.join(dataset_path, "Healthcare_System_Performance_Dataset_FINAL.csv")
df_final.to_csv(final_file_path, index=False)
print(f"\nFinal dataset saved as: {final_file_path}")

# Kuch important statistics
print("\n=== IMPORTANT STATISTICS ===")
print("Average Healthcare Expenditure: {:.2f}% of GDP".format(df_final['Healthcare Expenditure (% GDP)'].mean()))
print("Average Life Expectancy: {:.2f} years".format(df_final['Average Life Expectancy'].mean()))
print("Average Infant Mortality: {:.2f} per 1000".format(df_final['Infant Mortality Rate (per 1,000)'].mean()))
print("Average Vaccination Rate: {:.2f}%".format(df_final['Vaccination Rate (%)'].mean()))

# Region-wise analysis
print("\n=== REGION-WISE ANALYSIS ===")
for region in df_final['Region'].unique():
    region_data = df_final[df_final['Region'] == region]
    print(f"\n{region} ({len(region_data)} countries):")
    print("  Avg. Doctors per 1000: {:.2f}".format(region_data['Doctors per 1,000 People'].mean()))
    print("  Avg. Healthcare Coverage: {:.2f}%".format(region_data['Universal Healthcare Coverage (%)'].mean()))

# =============================================================================
# VISUALIZATION SECTION
# =============================================================================
print("\n" + "="*60)
print("CREATING VISUALIZATIONS...")
print("="*60)

# Style set karein
plt.style.use('default')
sns.set_palette("husl")

# 1. Bar Chart - Region-wise Average Life Expectancy
plt.figure(figsize=(14, 7))
region_life_exp = df_final.groupby('Region')['Average Life Expectancy'].mean().sort_values(ascending=False)
bars = plt.bar(region_life_exp.index, region_life_exp.values, color=plt.cm.Set3(np.arange(len(region_life_exp))))
plt.title('Average Life Expectancy by Region', fontsize=16, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Average Life Expectancy (Years)', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Bars par values add karein
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(dataset_path, '1_Life_Expectancy_by_Region.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Line Graph - Healthcare Expenditure vs Life Expectancy
plt.figure(figsize=(12, 7))
sorted_df = df_final.sort_values('Healthcare Expenditure (% GDP)')
plt.plot(sorted_df['Healthcare Expenditure (% GDP)'], sorted_df['Average Life Expectancy'], 
         marker='o', linestyle='-', alpha=0.7, color='steelblue', markersize=4)
plt.title('Healthcare Expenditure vs Life Expectancy', fontsize=16, fontweight='bold')
plt.xlabel('Healthcare Expenditure (% of GDP)', fontsize=12)
plt.ylabel('Average Life Expectancy (Years)', fontsize=12)
plt.grid(True, alpha=0.3)

# Trend line add karein
z = np.polyfit(sorted_df['Healthcare Expenditure (% GDP)'], sorted_df['Average Life Expectancy'], 1)
p = np.poly1d(z)
plt.plot(sorted_df['Healthcare Expenditure (% GDP)'], p(sorted_df['Healthcare Expenditure (% GDP)']), 
         "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.savefig(os.path.join(dataset_path, '2_Expenditure_vs_LifeExpectancy.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. Histogram - Doctors per 1,000 People Distribution
plt.figure(figsize=(12, 7))
plt.hist(df_final['Doctors per 1,000 People'], bins=15, edgecolor='black', alpha=0.7, color='lightseagreen')
plt.title('Distribution of Doctors per 1,000 People', fontsize=16, fontweight='bold')
plt.xlabel('Doctors per 1,000 People', fontsize=12)
plt.ylabel('Number of Countries', fontsize=12)
plt.axvline(df_final['Doctors per 1,000 People'].mean(), color='red', linestyle='dashed', linewidth=2, 
            label=f'Mean: {df_final["Doctors per 1,000 People"].mean():.2f}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(dataset_path, '3_Doctors_Distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. Heatmap - Correlation Matrix
plt.figure(figsize=(14, 10))
numeric_cols = df_final.select_dtypes(include=[np.number]).columns
correlation_matrix = df_final[numeric_cols].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8}, annot_kws={"size": 10})
plt.title('Correlation Matrix of Healthcare Indicators', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(dataset_path, '4_Correlation_Heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. Pie Chart - Region-wise Distribution of Countries
plt.figure(figsize=(12, 10))
region_counts = df_final['Region'].value_counts()
colors = plt.cm.Paired(np.linspace(0, 1, len(region_counts)))
plt.pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=colors, textprops={'fontsize': 12})
plt.title('Distribution of Countries by Region', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(dataset_path, '5_Region_Distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 6. Boxplot - Infant Mortality Rate by Region
plt.figure(figsize=(14, 8))
region_order = df_final.groupby('Region')['Infant Mortality Rate (per 1,000)'].median().sort_values().index
sns.boxplot(x='Region', y='Infant Mortality Rate (per 1,000)', data=df_final, order=region_order)
plt.title('Infant Mortality Rate Distribution by Region', fontsize=16, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Infant Mortality Rate (per 1,000 live births)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(dataset_path, '6_Infant_Mortality_by_Region.png'), dpi=300, bbox_inches='tight')
plt.show()

# 7. Scatter Plot - Doctors vs Healthcare Coverage with Life Expectancy as color
plt.figure(figsize=(14, 10))
scatter = plt.scatter(df_final['Doctors per 1,000 People'], 
                     df_final['Universal Healthcare Coverage (%)'],
                     c=df_final['Average Life Expectancy'], 
                     cmap='viridis', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)

cbar = plt.colorbar(scatter)
cbar.set_label('Average Life Expectancy (Years)', fontsize=12)
plt.title('Doctors Density vs Healthcare Coverage\n(Colored by Life Expectancy)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Doctors per 1,000 People', fontsize=12)
plt.ylabel('Universal Healthcare Coverage (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(dataset_path, '7_Doctors_vs_Coverage.png'), dpi=300, bbox_inches='tight')
plt.show()

# 8. Pairplot for multiple relationships
print("Creating pairplot (this may take a moment)...")
g = sns.pairplot(df_final[['Healthcare Expenditure (% GDP)', 'Doctors per 1,000 People', 
                          'Universal Healthcare Coverage (%)', 'Average Life Expectancy',
                          'Infant Mortality Rate (per 1,000)', 'Region']], 
                 hue='Region', diag_kind='hist', corner=True, height=2.2, plot_kws={'alpha': 0.6})

# Pairplot par title add karne ka sahi tareeka
g.fig.suptitle('Healthcare Indicators Relationship Analysis by Region', 
               fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(os.path.join(dataset_path, '8_Pairplot_Analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("ALL VISUALIZATIONS COMPLETED AND SAVED!")
print("="*60)