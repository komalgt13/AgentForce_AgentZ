#!/usr/bin/env python3
"""
Simple Demo Script for AI Persona Generator
Demonstrates the core agentic AI system with sample data
"""

import pandas as pd
import numpy as np
from agentic_persona_generator import AgenticPersonaGenerator
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Generate sample customer survey data for demonstration"""
    np.random.seed(42)
    
    # Demographics
    ages = np.random.normal(35, 12, 200).astype(int)
    ages = np.clip(ages, 18, 70)
    
    genders = np.random.choice(['Male', 'Female', 'Other'], 200, p=[0.45, 0.50, 0.05])
    
    locations = np.random.choice([
        'Urban', 'Suburban', 'Rural'
    ], 200, p=[0.4, 0.5, 0.1])
    
    incomes = np.random.choice([
        'Under $30K', '$30K-$50K', '$50K-$75K', '$75K-$100K', 'Over $100K'
    ], 200, p=[0.15, 0.25, 0.30, 0.20, 0.10])
    
    # Behavioral data
    purchase_frequency = np.random.choice([
        'Weekly', 'Monthly', 'Quarterly', 'Annually', 'Rarely'
    ], 200, p=[0.1, 0.3, 0.4, 0.15, 0.05])
    
    preferred_channels = np.random.choice([
        'Online', 'In-store', 'Mobile App', 'Phone', 'Social Media'
    ], 200, p=[0.35, 0.25, 0.20, 0.10, 0.10])
    
    # Satisfaction and engagement
    satisfaction = np.random.randint(1, 6, 200)  # 1-5 scale
    engagement_score = np.random.randint(1, 11, 200)  # 1-10 scale
    
    # Product preferences
    product_categories = np.random.choice([
        'Electronics', 'Clothing', 'Home & Garden', 'Health & Beauty', 
        'Sports & Outdoors', 'Books & Media'
    ], 200)
    
    # Pain points
    pain_points = np.random.choice([
        'High prices', 'Poor customer service', 'Limited selection',
        'Slow delivery', 'Complicated returns', 'Website issues'
    ], 200)
    
    return pd.DataFrame({
        'customer_id': range(1, 201),
        'age': ages,
        'gender': genders,
        'location': locations,
        'income_bracket': incomes,
        'purchase_frequency': purchase_frequency,
        'preferred_channel': preferred_channels,
        'satisfaction_score': satisfaction,
        'engagement_score': engagement_score,
        'favorite_category': product_categories,
        'main_pain_point': pain_points,
        'annual_spend': np.random.exponential(500, 200).astype(int),
        'loyalty_years': np.random.exponential(2, 200).round(1)
    })

async def run_demo():
    """Run the AI persona generator demo"""
    print("AI Persona Generator Demo")
    print("=" * 50)
    
    try:
        # Create sample data
        print("Generating sample customer data...")
        customer_data = create_sample_data()
        print(f"Created dataset with {len(customer_data)} customers")
        print(f"Columns: {list(customer_data.columns)}")
        
        # Initialize the generator
        print("\nInitializing AI Persona Generator...")
        generator = AgenticPersonaGenerator()
        
        # Generate personas
        print("\nStarting persona generation process...")
        print("This may take a few minutes as agents collaborate...")
        
        personas = await generator.generate_personas(customer_data)
        
        # Display results
        print("\n" + "=" * 50)
        print("PERSONA GENERATION COMPLETE!")
        print("=" * 50)
        
        for i, persona in enumerate(personas, 1):
            print(f"\nPERSONA {i}:")
            print("-" * 30)
            
            # Core info
            print(f"Name: {persona.get('name', 'Persona ' + str(i))}")
            print(f"Size: {persona.get('size', 'N/A')} customers")
            
            # Demographics
            if 'demographics' in persona:
                print(f"\nDemographics:")
                for key, value in persona['demographics'].items():
                    print(f"  • {key.title()}: {value}")
            
            # Behaviors
            if 'behaviors' in persona:
                print(f"\nKey Behaviors:")
                behaviors = persona['behaviors']
                if isinstance(behaviors, list):
                    for behavior in behaviors[:3]:  # Show top 3
                        print(f"  • {behavior}")
                else:
                    print(f"  • {behaviors}")
            
            # Pain points
            if 'pain_points' in persona:
                print(f"\nMain Pain Points:")
                pain_points = persona['pain_points']
                if isinstance(pain_points, list):
                    for pain in pain_points[:2]:  # Show top 2
                        print(f"  • {pain}")
                else:
                    print(f"  • {pain_points}")
            
            # Marketing strategies
            if 'marketing_strategies' in persona:
                print(f"\nMarketing Strategies:")
                strategies = persona['marketing_strategies']
                if isinstance(strategies, list):
                    for strategy in strategies[:2]:  # Show top 2
                        print(f"  • {strategy}")
                else:
                    print(f"  • {strategies}")
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("Try running the Streamlit app for an interactive experience:")
        print("   streamlit run agentic_app.py")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\nDemo failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check that all dependencies are installed: pip install -r requirements.txt")
        print("2. Verify your .env file is configured correctly")
        print("3. Ensure you have internet connectivity for model downloads")

if __name__ == "__main__":
    asyncio.run(run_demo())
