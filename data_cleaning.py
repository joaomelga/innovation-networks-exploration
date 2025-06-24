import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

def clean_companies_data(companies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean companies dataset according to the paper methodology.
    Excludes companies with incomplete information, founded after 2014, or with exit status.
    
    Args:
        companies_df: Raw companies DataFrame
        
    Returns:
        Cleaned companies DataFrame
    """
    df = companies_df.copy()
    initial_count = len(df)
    
    print(f"Initial companies count: {initial_count}")
    
    # 1. Exclude companies with incomplete information
    essential_columns = ['uuid', 'name', 'founded_year']
    
    for col in essential_columns:
        if col in df.columns:
            before = len(df)
            df = df.dropna(subset=[col])
            removed = before - len(df)
            if removed > 0:
                print(f"Removed {removed} companies missing {col}")
    
    # 2. Exclude companies founded after 2014 (as per paper methodology)
    if 'founded_year' in df.columns:
        before = len(df)
        df = df[df['founded_year'] <= 2014]
        removed = before - len(df)
        print(f"Removed {removed} companies founded after 2014")
    
    # 3. Exclude companies that had an "exit" (bankruptcy or takeover)
    if 'status' in df.columns:
        before = len(df)
        # Keep only operating companies or those with unknown status
        df = df[~df['status'].isin(['closed', 'acquired', 'ipo'])]
        removed = before - len(df)
        print(f"Removed {removed} companies with exit status")
    
    print(f"Final companies count after cleaning: {len(df)}")
    print(f"Total removed: {initial_count - len(df)}")
    
    return df

def clean_funding_data(investments_df: pd.DataFrame, 
                      funding_rounds_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Clean funding data and ensure consistency as per paper methodology.
    
    Args:
        investments_df: Raw investments DataFrame
        funding_rounds_df: Optional funding rounds DataFrame for validation
        
    Returns:
        Cleaned investments DataFrame
    """
    df = investments_df.copy()
    initial_count = len(df)
    
    print(f"Initial investments count: {initial_count}")
    
    # 1. Remove investments with missing essential information
    essential_columns = ['org_uuid', 'investor_uuid']
    
    for col in essential_columns:
        if col in df.columns:
            before = len(df)
            df = df.dropna(subset=[col])
            removed = before - len(df)
            if removed > 0:
                print(f"Removed {removed} investments missing {col}")
    
    # 2. Clean funding amounts
    if 'total_funding_usd' in df.columns:
        before = len(df)
        # Remove negative or zero funding amounts
        df = df[df['total_funding_usd'] > 0]
        removed = before - len(df)
        if removed > 0:
            print(f"Removed {removed} investments with invalid funding amounts")
    
    # 3. Validate funding consistency if funding_rounds data is available
    if funding_rounds_df is not None:
        df = validate_funding_consistency(df, funding_rounds_df)
    
    print(f"Final investments count after cleaning: {len(df)}")
    print(f"Total removed: {initial_count - len(df)}")
    
    return df

def validate_funding_consistency(investments_df: pd.DataFrame, 
                                funding_rounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that sum of individual investments matches total funding rounds.
    As mentioned in the paper: "excluding funding data that were inconsistent 
    (notably when the sum of funds listed per funding round did not match 
    the total amount of funds indicated in the database)"
    
    Args:
        investments_df: Investments DataFrame
        funding_rounds_df: Funding rounds DataFrame
        
    Returns:
        Validated investments DataFrame
    """
    # Group investments by funding round
    if 'funding_round_uuid' in investments_df.columns:
        inv_totals = investments_df.groupby('funding_round_uuid')['total_funding_usd'].sum()
        
        # Compare with funding rounds totals
        inconsistent_rounds = []
        
        for round_uuid, inv_total in inv_totals.items():
            round_data = funding_rounds_df[funding_rounds_df['uuid'] == round_uuid]
            if not round_data.empty and 'money_raised_usd' in funding_rounds_df.columns:
                round_total = round_data['money_raised_usd'].iloc[0]
                if pd.notna(round_total) and abs(inv_total - round_total) > 0.01 * round_total:  # 1% tolerance
                    inconsistent_rounds.append(round_uuid)
        
        if inconsistent_rounds:
            print(f"Found {len(inconsistent_rounds)} funding rounds with inconsistent totals")
            # Remove investments from inconsistent rounds
            before = len(investments_df)
            investments_df = investments_df[~investments_df['funding_round_uuid'].isin(inconsistent_rounds)]
            removed = before - len(investments_df)
            print(f"Removed {removed} investments from inconsistent rounds")
    
    return investments_df

def apply_funding_threshold(companies_df: pd.DataFrame, 
                          investments_df: pd.DataFrame, 
                          threshold: float = 150000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply funding threshold as per paper methodology.
    "We further restricted our study to start-ups that had raised more than $150,000"
    
    Args:
        companies_df: Companies DataFrame
        investments_df: Investments DataFrame
        threshold: Minimum funding threshold in USD (default: $150,000)
        
    Returns:
        Tuple of filtered (companies_df, investments_df)
    """
    print(f"Applying funding threshold of ${threshold:,.0f}")
    
    # Calculate total funding per company
    company_funding = investments_df.groupby('org_uuid')['total_funding_usd'].sum().reset_index()
    company_funding.columns = ['uuid', 'total_funding']
    
    # Identify companies that meet the threshold
    qualified_companies = company_funding[company_funding['total_funding'] >= threshold]['uuid']
    
    print(f"Companies meeting funding threshold: {len(qualified_companies)}")
    
    # Filter companies and investments
    filtered_companies = companies_df[companies_df['uuid'].isin(qualified_companies)]
    filtered_investments = investments_df[investments_df['org_uuid'].isin(qualified_companies)]
    
    print(f"Filtered companies: {len(filtered_companies)}")
    print(f"Filtered investments: {len(filtered_investments)}")
    
    return filtered_companies, filtered_investments

def exclude_accelerator_only_funding(companies_df: pd.DataFrame, 
                                   investments_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exclude companies that only received funding from accelerators.
    As per paper: "we exclude start-ups that did not manage to attract at least 
    one investor other than their accelerator" to prevent endogeneity bias.
    
    Args:
        companies_df: Companies DataFrame
        investments_df: Investments DataFrame
        
    Returns:
        Tuple of filtered (companies_df, investments_df)
    """
    print("Excluding companies with accelerator-only funding...")
    
    # For each company, check if they have non-accelerator investors
    company_investor_types = investments_df.groupby('org_uuid').apply(
        lambda x: x['investor_types'].str.contains('accelerator|incubator', case=False, na=False).all()
    )
    
    # Companies with ONLY accelerator funding
    accelerator_only_companies = company_investor_types[company_investor_types].index
    
    print(f"Companies with accelerator-only funding: {len(accelerator_only_companies)}")
    
    # Filter out accelerator-only companies
    filtered_companies = companies_df[~companies_df['uuid'].isin(accelerator_only_companies)]
    filtered_investments = investments_df[~investments_df['org_uuid'].isin(accelerator_only_companies)]
    
    print(f"Companies after excluding accelerator-only: {len(filtered_companies)}")
    print(f"Investments after excluding accelerator-only: {len(filtered_investments)}")
    
    return filtered_companies, filtered_investments

def create_final_sample(data: Dict[str, pd.DataFrame], 
                       funding_threshold: float = 150000) -> Dict[str, pd.DataFrame]:
    """
    Create the final cleaned sample following the paper methodology exactly.
    
    This implements the complete cleaning process described in the paper:
    1. Clean companies data (remove incomplete info, post-2014 founding, exits)
    2. Clean funding data (ensure consistency)
    3. Apply $150,000 funding threshold
    4. Exclude accelerator-only companies (prevent endogeneity bias)
    
    Args:
        data: Dictionary containing raw DataFrames
        funding_threshold: Minimum funding threshold in USD
        
    Returns:
        Dictionary containing cleaned DataFrames
    """
    print("=" * 50)
    print("CREATING FINAL SAMPLE FOLLOWING PAPER METHODOLOGY")
    print("=" * 50)
    
    # Step 1: Clean companies data
    print("\n1. Cleaning companies data...")
    companies_clean = clean_companies_data(data['companies'])
    
    # Step 2: Clean funding data
    print("\n2. Cleaning funding data...")
    funding_rounds = data.get('funding_rounds')
    investments_clean = clean_funding_data(data['investments'], funding_rounds)
    
    # Step 3: Apply funding threshold ($150,000 as per paper)
    print("\n3. Applying funding threshold...")
    companies_thresh, investments_thresh = apply_funding_threshold(
        companies_clean, investments_clean, funding_threshold
    )
    
    # Step 4: Exclude accelerator-only companies (prevent endogeneity bias)
    print("\n4. Excluding accelerator-only companies...")
    companies_final, investments_final = exclude_accelerator_only_funding(
        companies_thresh, investments_thresh
    )
    
    # Step 5: Final consistency check
    print("\n5. Final consistency check...")
    
    # Ensure all investments reference existing companies
    valid_companies = set(companies_final['uuid'])
    investments_final = investments_final[investments_final['org_uuid'].isin(valid_companies)]
    
    # Ensure all companies have at least one investment
    companies_with_investments = set(investments_final['org_uuid'])
    companies_final = companies_final[companies_final['uuid'].isin(companies_with_investments)]
    
    print(f"\nFINAL SAMPLE:")
    print(f"Companies: {len(companies_final)}")
    print(f"Investments: {len(investments_final)}")
    print(f"Unique investors: {investments_final['investor_uuid'].nunique()}")
    
    # Create final cleaned dataset
    cleaned_data = {
        'companies': companies_final,
        'investments': investments_final,
        'funding_rounds': data.get('funding_rounds'),
        'investors': data.get('investors')
    }
    
    return cleaned_data

def get_sample_statistics(cleaned_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Generate statistics about the final cleaned sample to compare with paper results.
    The paper reports: "Our final sample was composed of 6000 start-ups, 
    with a participation in an accelerator program for 688 among them"
    
    Args:
        cleaned_data: Dictionary containing cleaned DataFrames
        
    Returns:
        Dictionary containing sample statistics
    """
    companies = cleaned_data['companies']
    investments = cleaned_data['investments']
    
    stats = {}
    
    # Basic counts
    stats['total_companies'] = len(companies)
    stats['total_investments'] = len(investments)
    stats['unique_investors'] = investments['investor_uuid'].nunique()
    
    # Accelerator participation
    accelerator_investments = investments[
        investments['investor_types'].str.contains('accelerator|incubator', case=False, na=False)
    ]
    stats['accelerated_companies'] = accelerator_investments['org_uuid'].nunique()
    stats['acceleration_rate'] = stats['accelerated_companies'] / stats['total_companies']
    
    # Funding statistics
    company_funding = investments.groupby('org_uuid')['total_funding_usd'].sum()
    stats['avg_funding_per_company'] = company_funding.mean()
    stats['median_funding_per_company'] = company_funding.median()
    stats['total_funding_volume'] = company_funding.sum()
    
    # Time period
    if 'founded_year' in companies.columns:
        stats['founding_years'] = {
            'min': companies['founded_year'].min(),
            'max': companies['founded_year'].max()
        }
    
    if 'announced_year' in investments.columns:
        stats['investment_years'] = {
            'min': investments['announced_year'].min(),
            'max': investments['announced_year'].max()
        }
    
    return stats

def identify_accelerated_companies(investments_df: pd.DataFrame) -> List[str]:
    """
    Identify companies that participated in accelerator programs.
    This follows the paper's approach of identifying accelerated companies.
    
    Args:
        investments_df: Cleaned investments DataFrame
        
    Returns:
        List of company UUIDs that participated in accelerator programs
    """
    accelerator_investments = investments_df[
        investments_df['investor_types'].str.contains('accelerator|incubator', case=False, na=False)
    ]
    
    accelerated_companies = accelerator_investments['org_uuid'].unique().tolist()
    
    print(f"Identified {len(accelerated_companies)} companies that participated in accelerator programs")
    
    return accelerated_companies