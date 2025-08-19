import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

def load_data(data_dir: str = 'data/france') -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the France data directory.
    
    Args:
        data_dir: Path to the directory containing CSV files
        
    Returns:
        Dictionary containing all loaded DataFrames
    """
    data_files = {
        'companies': 'companies.csv',
        'funding_rounds': 'funding_rounds.csv',
        'investments': 'investments.csv',
        'investors': 'investors.csv'
    }
    
    data = {}
    
    for key, filename in data_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
                data[key] = df
                # print(f"✓ Loaded {key}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        else:
            print(f"✗ File not found: {filepath}")
    
    return data

def load_clean_data(data_dir: str = 'data/france') -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the France data directory.
    
    Args:
        data_dir: Path to the directory containing CSV files
        
    Returns:
        Dictionary containing all loaded DataFrames
    """
    data_files = {
        'companies': 'companies_clean.csv',
        'funding_rounds': 'funding_rounds_clean.csv',
        'investments': 'investments_clean.csv',
        'investors': 'investors_clean.csv'
    }
    
    data = {}
    
    for key, filename in data_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
                data[key] = df
                # print(f"✓ Loaded {key}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        else:
            print(f"✗ File not found: {filepath}")
    
    return data

def extract_accelerator_investments(investments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract investments from accelerators/incubators.
    
    Args:
        investments_df: DataFrame containing investment data
        
    Returns:
        DataFrame containing only accelerator investments
    """
    # Filter for accelerator/incubator investments (following clustering analysis approach)
    accelerator_mask = investments_df['investor_types'].str.contains(
        'accelerator|incubator', case=False, na=False
    )
    
    accelerated_investments = investments_df[accelerator_mask].copy()
    
    print(f"Found {accelerated_investments.shape[0]} accelerator investments")
    print(f"Unique companies in accelerators: {accelerated_investments['org_uuid'].nunique()}")
    
    return accelerated_investments

def extract_investments_by_type(investments: pd.DataFrame, funding_rounds: pd.DataFrame, investment_type: str) -> pd.DataFrame:
    """
    Extract specific investments based on funding round investment type.
    
    Args:
        investments: DataFrame containing investment data
        funding_rounds: DataFrame containing funding rounds data
        investment_type: The investment type to filter for
        
    Returns:
        DataFrame containing only investments from funding rounds with the specified investment type
    """
    # First, filter funding rounds by investment type
    filtered_funding_rounds = funding_rounds[
        funding_rounds['investment_type'].str.contains(investment_type, case=False, na=False)
    ]
    
    # Get the UUIDs of funding rounds with the specified investment type
    funding_round_uuids = set(filtered_funding_rounds['uuid'].dropna())
    
    # Filter investments that have a funding_round_uuid in the filtered set
    filtered_investments = investments[
        investments['funding_round_uuid'].isin(funding_round_uuids)
    ].copy()
    
    print(f"Found {len(filtered_funding_rounds)} funding rounds with '{investment_type}' type")
    print(f"Found {filtered_investments.shape[0]} investments from these funding rounds")
    print(f"Unique companies with {investment_type} investments: {filtered_investments['org_uuid'].nunique()}")
    
    return filtered_investments

def extract_vc_investments(investments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract venture capital investments (excluding accelerators/incubators).
    
    Args:
        investments_df: DataFrame containing investment data
        
    Returns:
        DataFrame containing only VC investments
    """
    # Filter for VC investments, excluding accelerators/incubators
    vc_mask = (investments_df['investor_types'].str.contains('venture', case=False, na=False))
    
    vc_investments = investments_df[vc_mask].copy()
    
    print(f"Found {vc_investments.shape[0]} VC investments")
    print(f"Unique companies with VC funding: {vc_investments['org_uuid'].nunique()}")
    
    return vc_investments

# Maybe not so useful, because we are actually looking to coinvestment
def identify_two_stage_companies(accelerated_df: pd.DataFrame, vc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify companies that received both accelerator and VC funding (two-stage approach).
    
    Args:
        accelerated_df: DataFrame of accelerator investments
        vc_df: DataFrame of VC investments
        
    Returns:
        DataFrame of companies with both types of funding
    """
    # Get unique company IDs from each type
    accelerated_companies = set(accelerated_df['org_uuid'].dropna())
    vc_companies = set(vc_df['org_uuid'].dropna())
    
    # Find intersection - companies with both accelerator and VC funding
    two_stage_companies = accelerated_companies.intersection(vc_companies)
    
    print(f"Companies with both accelerator and VC funding: {len(two_stage_companies)}")
    
    # Create summary DataFrame
    two_stage_df = pd.DataFrame({
        'org_uuid': list(two_stage_companies)
    })
    
    return two_stage_df

def extract_company_funding_summary(companies_df: pd.DataFrame, 
                                  investments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of funding information for each company.
    
    Args:
        companies_df: DataFrame containing company information
        investments_df: DataFrame containing investment data
        
    Returns:
        DataFrame with funding summary per company
    """
    # Group investments by company
    funding_summary = investments_df.groupby('org_uuid').agg({
        'total_funding_usd': ['sum', 'count', 'mean'],
        'announced_year': ['min', 'max'],
        'investment_type': lambda x: ', '.join(x.dropna().unique()),
        'investor_types': lambda x: ', '.join(x.dropna().unique())
    }).reset_index()
    
    # Flatten column names
    funding_summary.columns = [
        'org_uuid', 'total_funding_sum', 'funding_rounds_count', 'avg_funding_per_round',
        'first_funding_year', 'last_funding_year', 'investment_types', 'investor_types'
    ]
    
    # Merge with company information
    company_funding = companies_df.merge(
        funding_summary, 
        left_on='uuid', 
        right_on='org_uuid', 
        how='left'
    )
    
    print(f"Company funding summary created for {company_funding.shape[0]} companies")
    
    return company_funding

def categorize_investment_types(investments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize and clean investment type information.
    
    Args:
        investments_df: DataFrame containing investment data
        
    Returns:
        DataFrame with categorized investment types
    """
    df = investments_df.copy()
    
    # Clean and standardize investment types
    df['investment_type_clean'] = df['investment_type'].str.lower().str.strip()
    
    # Create broader categories
    def categorize_investment(inv_type):
        if pd.isna(inv_type):
            return 'unknown'
        
        inv_type = str(inv_type).lower()
        
        if 'seed' in inv_type:
            return 'seed'
        elif 'series_a' in inv_type or 'series a' in inv_type:
            return 'series_a'
        elif 'series_b' in inv_type or 'series b' in inv_type:
            return 'series_b'
        elif 'series_c' in inv_type or 'series c' in inv_type:
            return 'series_c'
        elif 'angel' in inv_type:
            return 'angel'
        elif 'grant' in inv_type:
            return 'grant'
        else:
            return 'other'
    
    df['investment_category'] = df['investment_type_clean'].apply(categorize_investment)
    
    print("Investment type categorization:")
    print(df['investment_category'].value_counts())
    
    return df

def extract_sector_information(companies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and clean sector/category information from companies.
    
    Args:
        companies_df: DataFrame containing company information
        
    Returns:
        DataFrame with cleaned sector information
    """
    df = companies_df.copy()
    
    # Extract main categories (following clustering analysis approach)
    if 'category_groups_list' in df.columns:
        df['main_category'] = df['category_groups_list'].str.split(',').str[0].str.strip()
    elif 'category_list' in df.columns:
        df['main_category'] = df['category_list'].str.split(',').str[0].str.strip()
    else:
        df['main_category'] = 'Unknown'
    
    # Count category distribution
    print("Top 10 sectors:")
    if 'main_category' in df.columns:
        print(df['main_category'].value_counts().head(10))
    
    return df

def get_data_summary(data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Generate summary statistics for all loaded datasets.
    
    Args:
        data: Dictionary containing all DataFrames
        
    Returns:
        Dictionary containing summary information
    """
    summary = {}
    
    for name, df in data.items():
        summary[name] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_data': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
    
    return summary

def create_investment_pairs(left_side_investments: pd.DataFrame, right_side_investments: pd.DataFrame, suffix_left: str = 'left', suffix_right: str = 'right') -> pd.DataFrame:
    """
    Create accelerator-VC pairs following the clustering analysis approach.
    
    Args:
        left_side_investments: DataFrame of accelerator investments
        right_side_investments: DataFrame of VC investments
        
    Returns:
        DataFrame with accelerator-VC investment pairs
    """
    # Merge on company UUID to find two-stage investments
    merged = left_side_investments.merge( 
        right_side_investments, 
        on='org_uuid', 
        suffixes=('_' + suffix_left, '_' + suffix_right)
    )
    
    # Filter out self-pairs (same investor)
    merged = merged[merged[f'investor_name_{suffix_left}'] != merged[f'investor_name_{suffix_right}']]
    
    # Sort the investor names to avoid duplicate pairs in different order
    # Drop duplicates (each pair should appear only once)
    merged['investor_pair'] = merged.apply(
        lambda row: tuple(sorted([row[f'investor_name_{suffix_left}'], row[f'investor_name_{suffix_right}']])), axis=1
    )
    
    merged['pairs'] = merged.apply(
        lambda row: (row['org_uuid'], row['investor_pair']), axis=1
    )
    merged = merged.drop_duplicates(subset='pairs')

    # (Optional) Drop helper column
    two_stage_investments = merged.drop(columns='pairs')
    
    print(f"Created {two_stage_investments.shape[0]} investment pairs")
    print(f"Covering {two_stage_investments['org_uuid'].nunique()} unique statups")
    
    return two_stage_investments