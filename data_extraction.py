import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

def load_france_data(data_dir: str = 'data/france') -> Dict[str, pd.DataFrame]:
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
                print(f"✓ Loaded {key}: {df.shape[0]} rows, {df.shape[1]} columns")
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

def extract_vc_investments(investments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract venture capital investments (excluding accelerators/incubators).
    
    Args:
        investments_df: DataFrame containing investment data
        
    Returns:
        DataFrame containing only VC investments
    """
    # Filter for VC investments, excluding accelerators/incubators
    vc_mask = (
        investments_df['investor_types'].str.contains('venture', case=False, na=False) &
        ~investments_df['investor_types'].str.contains('accelerator|incubator', case=False, na=False)
    )
    
    vc_investments = investments_df[vc_mask].copy()
    
    print(f"Found {vc_investments.shape[0]} VC investments")
    print(f"Unique companies with VC funding: {vc_investments['org_uuid'].nunique()}")
    
    return vc_investments

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

def create_accelerator_vc_pairs(accelerated_investments: pd.DataFrame, 
                               vc_investments: pd.DataFrame) -> pd.DataFrame:
    """
    Create accelerator-VC pairs following the clustering analysis approach.
    
    Args:
        accelerated_investments: DataFrame of accelerator investments
        vc_investments: DataFrame of VC investments
        
    Returns:
        DataFrame with accelerator-VC investment pairs
    """
    # Merge on company UUID to find two-stage investments
    two_stage_investments = pd.merge(
        accelerated_investments[['uuid', 'investor_uuid', 'investor_name', 'investor_types', 'org_uuid']], 
        vc_investments[['uuid', 'investor_uuid', 'investor_name', 'investor_types', 'org_uuid', 
                       'company_name', 'category_groups_list', 'investment_type', 'total_funding_usd']], 
        on='org_uuid', 
        how='inner',
        suffixes=('_accelerator', '_vc')
    )
    
    print(f"Created {two_stage_investments.shape[0]} accelerator-VC investment pairs")
    print(f"Covering {two_stage_investments['org_uuid'].nunique()} unique companies")
    
    return two_stage_investments