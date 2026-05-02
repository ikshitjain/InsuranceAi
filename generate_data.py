import pandas as pd
import numpy as np
import os

def generate_fraud_dataset(n_samples=2000):
    np.random.seed(42)
    
    # Features
    vehicle_age = np.random.randint(0, 20, n_samples)
    # Insured Value (IDV) usually decreases with age
    base_value = np.random.randint(300000, 5000000, n_samples)
    insured_value = base_value * (1 - 0.1 * vehicle_age)
    insured_value = np.maximum(insured_value, 50000) # Min 50k
    
    # Premium is usually 2-4% of IDV
    premium = insured_value * np.random.uniform(0.02, 0.05, n_samples)
    
    # Claim amount relative to insured value
    claim_ratio = np.random.uniform(0.05, 1.2, n_samples)
    claim_amount = insured_value * claim_ratio
    
    accident_type = np.random.choice(['Rear-end', 'Side-swipe', 'Front-end', 'Parked Car'], n_samples)
    police_report = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    witness_present = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    previous_claims = np.random.randint(0, 5, n_samples)
    
    # Improved Fraud Score logic:
    # 1. High claim ratio (claiming more than value)
    # 2. No police report on a high claim
    # 3. Old vehicle with a "total loss" claim
    # 4. Multiple previous claims
    
    fraud_score = (
        0.4 * (claim_ratio > 0.9).astype(float) +
        0.3 * ((claim_ratio > 0.5) & (police_report == 0)).astype(float) +
        0.2 * ((vehicle_age > 10) & (claim_ratio > 0.7)).astype(float) +
        0.1 * (previous_claims / 5) +
        0.1 * (1 - witness_present)
    )
    
    # Add some noise
    fraud_score += np.random.normal(0, 0.1, n_samples)
    
    fraud_reported = (fraud_score > 0.65).astype(int)
    
    df = pd.DataFrame({
        'claim_amount': claim_amount.astype(int),
        'vehicle_age': vehicle_age,
        'accident_type': accident_type,
        'police_report': police_report,
        'witness_present': witness_present,
        'previous_claims': previous_claims,
        'premium': premium.astype(int),
        'insured_value': insured_value.astype(int),
        'fraud_reported': fraud_reported
    })
    
    output_path = "fraud_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {output_path} ({n_samples} rows)")

if __name__ == "__main__":
    generate_fraud_dataset()
