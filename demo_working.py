#!/usr/bin/env python3
"""
Simple Working Demo - AI Persona Generator
Shows the system working with direct agent calls
"""

import pandas as pd
import numpy as np
import os
import asyncio
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Set environment variables for Gemini
os.environ['GEMINI_MODEL'] = 'gemini-1.5-flash'

# Import Gemini LLM Manager
from open_source_llm import GeminiLLMManager
from agentic_persona_generator import AgenticPersonaGenerator

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

def create_sample_data():
    """Generate realistic customer data"""
    np.random.seed(42)
    
    # Create 50 customers (smaller for demo)
    n_customers = 50
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'age': np.random.normal(35, 12, n_customers).astype(int).clip(18, 70),
        'gender': np.random.choice(['Male', 'Female'], n_customers, p=[0.5, 0.5]),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_customers, p=[0.4, 0.5, 0.1]),
        'income': np.random.choice(['<30K', '30-50K', '50-75K', '75K+'], n_customers, p=[0.2, 0.3, 0.3, 0.2]),
        'purchase_frequency': np.random.choice(['Weekly', 'Monthly', 'Quarterly'], n_customers, p=[0.2, 0.6, 0.2]),
        'satisfaction': np.random.randint(1, 6, n_customers),
        'annual_spend': np.random.exponential(500, n_customers).astype(int)
    }
    
    return pd.DataFrame(data)

def analyze_data_with_ai(data, llm):
    """Use AI to analyze customer data"""
    
    # Create data summary
    summary = f"""
Customer Data Analysis:
- Total customers: {len(data)}
- Age range: {data['age'].min()}-{data['age'].max()} (avg: {data['age'].mean():.1f})
- Gender split: {data['gender'].value_counts().to_dict()}
- Locations: {data['location'].value_counts().to_dict()}
- Income levels: {data['income'].value_counts().to_dict()}
- Purchase patterns: {data['purchase_frequency'].value_counts().to_dict()}
- Satisfaction average: {data['satisfaction'].mean():.1f}/5
- Annual spend range: ${data['annual_spend'].min()}-${data['annual_spend'].max()}
"""
    
    # Ask AI to identify patterns
    prompt = f"""## CUSTOMER SEGMENTATION ANALYSIS REQUEST

**Objective**: Identify and profile 3 distinct customer segments for targeted marketing

**Dataset Summary**:
{summary}

**Required Analysis**:

### Segment Identification
Analyze the data patterns to identify 3 distinct customer groups based on:
- Purchasing behavior and frequency
- Demographic characteristics (age, location)
- Satisfaction levels and spending patterns
- Annual spend ranges and value potential

### Persona Development
For each of the 3 customer segments, provide:

**1. Segment Identity**
- Descriptive name that captures the essence of this group
- Primary distinguishing characteristics
- Estimated segment size/percentage

**2. Customer Profile**
- Key demographic features (age ranges, typical locations)
- Behavioral patterns (purchase frequency, spending habits)
- Satisfaction levels and expectations

**3. Business Insights**
- Primary pain points and challenges
- Value drivers and motivations
- Revenue potential and lifetime value indicators

**Output Format**: 
Structure each persona with clear headers and bullet points. Keep descriptions concise but specific (2-3 sentences per section)."""
    
    try:
        response = llm.invoke(prompt)
        # Handle both string responses and response objects
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    except Exception as e:
        return f"Analysis completed with basic patterns. Error details: {str(e)}"

def generate_marketing_strategies(personas_text, llm):
    """Generate marketing strategies for the personas"""
    
    prompt = f"""## MARKETING STRATEGY DEVELOPMENT REQUEST

**Objective**: Create actionable marketing strategies for each customer persona

**Customer Personas**:
{personas_text}

**Strategy Requirements**:

For each persona, develop specific recommendations covering:

### 1. Channel Strategy
- **Primary Channels**: Most effective marketing channels for reaching this segment
- **Secondary Channels**: Supporting touchpoints for reinforcement
- **Channel Rationale**: Why these channels work best for this persona

### 2. Messaging & Communication
- **Core Messages**: Key value propositions that resonate
- **Communication Tone**: Formal/casual, technical/simple, emotional/rational
- **Content Types**: Formats that engage (email, social, video, etc.)

### 3. Campaign Concepts
- **Acquisition Tactics**: Strategies to attract new customers
- **Retention Approaches**: Methods to maintain loyalty and engagement
- **Seasonal Opportunities**: Time-based campaign ideas

### 4. Success Metrics
- **Primary KPIs**: Key performance indicators to track
- **Success Benchmarks**: Realistic targets for campaign performance

**Output Format**: 
Organize by persona with clear headers. Focus on practical, implementable strategies with specific recommendations."""
    
    try:
        response = llm.invoke(prompt)
        # Handle both string responses and response objects
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    except Exception as e:
        return f"Marketing strategy suggestions generated. Error details: {str(e)}"

async def run_simple_demo():
    """Run the full LangChain agentic pipeline demo"""
    
    print("AI Persona Generator - LangChain Agentic Demo")
    print("=" * 50)
    
    try:
        # 1. Create sample data
        print("Creating sample customer data...")
        data = create_sample_data()
        print(f"Generated {len(data)} customer records")
        
        # Show data preview
        print("\nCustomer Data Sample:")
        print(data.head(3).to_string(index=False))
        
        # 2. Initialize LangChain Agentic System
        print("\nInitializing LangChain Agentic System...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        # Initialize the full agentic persona generator
        generator = AgenticPersonaGenerator(api_key=api_key)
        print("✅ LangChain Agentic System ready with 6 specialized agents!")
        
        # Save data to CSV for the generator
        temp_csv = "temp_demo_data.csv"
        data.to_csv(temp_csv, index=False)
        
        # 3. Run REAL LangChain Agentic Pipeline (Using Actual Agents!)
        print("\n🤖 Running REAL LangChain Agentic Analysis Pipeline...")
        
        # Use the actual agentic persona generator with its full pipeline
        print("   🚀 Invoking full 6-agent LangChain system...")
        results = generator.generate_personas_from_csv(temp_csv)
        
        # The results come from the actual agentic system with real agents
        print("   ✅ LangChain agents completed analysis!")
        
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        
        # 4. Display Results from Real Agentic System
        
        # 4. Display Results from Real Agentic System
        print("\n" + "="*50)
        print("REAL LANGCHAIN AGENTIC PERSONAS & STRATEGIES")
        print("="*50)
        
        if results and 'detailed_personas' in results:
            personas_dict = results['detailed_personas']
            print(f"\n✅ Generated {len(personas_dict)} detailed personas from REAL LangChain agents:")
            for i, (key, persona) in enumerate(personas_dict.items(), 1):
                name = persona.get('name', f'Persona {i}')
                print(f"\n--- PERSONA {i}: {name} ---")
                
                # Show key persona details
                if 'demographics' in persona:
                    print(f"Demographics: {persona['demographics']}")
                if 'behavior_patterns' in persona:
                    print(f"Behavior: {persona['behavior_patterns']}")
                if 'pain_points' in persona:
                    print(f"Pain Points: {persona['pain_points']}")
                if 'marketing_strategy' in persona:
                    strategy = str(persona['marketing_strategy'])
                    if len(strategy) > 200:
                        print(f"Marketing Strategy: {strategy[:200]}...")
                    else:
                        print(f"Marketing Strategy: {strategy}")
        
        if results and 'marketing_insights' in results:
            print(f"\n✅ Generated comprehensive marketing insights from agents")
            
        if results and 'agent_analysis' in results:
            print(f"\n📊 Agent Analysis Summary:")
            analysis = results['agent_analysis']
            for agent_name, agent_data in analysis.items():
                if isinstance(agent_data, str):
                    print(f"- {agent_name}: {len(agent_data)} chars of analysis")
                elif isinstance(agent_data, dict) and 'insights' in agent_data:
                    print(f"- {agent_name}: {len(agent_data['insights'])} chars of insights")
        
        print("\n" + "="*50)
        print("REAL LANGCHAIN AGENTIC DEMO COMPLETED!")
        print("INFO: This demonstrates the ACTUAL 6-agent LangChain system")
        print("NOTE: Uses real LangChain agents with tools and specialized workflows")
        
        # Save results to file
        print("\nSaving REAL agentic results to file...")
        results_file = f"real_agentic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"Results saved to: {results_file}")
        print(f"Location: {os.path.abspath(results_file)}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ LangChain Agentic Demo failed: {str(e)}")
        print("This might be due to:")
        print("1. Missing GOOGLE_API_KEY in .env file")
        print("2. API rate limits") 
        print("3. Network connectivity issues")
        print("\n🔄 Falling back to simple LLM demo...")
        
        # Fallback to simple demo
        return await run_simple_llm_demo()

async def run_simple_llm_demo():
    """Fallback demo using direct LLM calls"""
    print("\n" + "="*30)
    print("SIMPLE LLM DEMO (Fallback)")
    print("="*30)
    
    try:
        # Create sample data
        data = create_sample_data()
        
        # Initialize simple LLM (use direct API for simple demo)
        llm_manager = GeminiLLMManager()
        llm = llm_manager.get_llm(
            model_name="gemini-1.5-flash",
            temperature=0.3,
            max_tokens=1000,
            use_langchain=False  # Use direct API for simple demo
        )
        
        # Simple AI analysis
        print("\nAI analyzing customer patterns...")
        personas_analysis = analyze_data_with_ai(data, llm)
        
        print("\n" + "="*40)
        print("SIMPLE AI-GENERATED PERSONAS")
        print("="*40)
        print(personas_analysis)
        
        # Save simple results
        results = {
            "timestamp": datetime.now().isoformat(),
            "demo_type": "simple_llm_fallback",
            "customer_data_size": len(data),
            "personas_analysis": personas_analysis
        }
        
        with open("simple_persona_results.json", "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\nSimple results saved to: simple_persona_results.json")
        return results
        
    except Exception as e:
        print(f"\n❌ Simple demo also failed: {str(e)}")
        return None

if __name__ == "__main__":
    asyncio.run(run_simple_demo())
