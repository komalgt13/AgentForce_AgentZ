# AI Customer Persona Generator

An enterprise-grade, multi-agent AI system that processes customer data to generate actionable customer personas using Google's Gemini 2.5-flash AI model.

## Overview

This system leverages advanced AI and machine learning to transform raw customer data into comprehensive personas with targeted marketing strategies. Built with a modular architecture using 6 specialized AI agents powered by Gemini 2.5-flash, it provides businesses with high-quality customer insights and marketing strategies.

## Core Capabilities

- **Automated Data Analysis**: Processes customer surveys, transaction data, and behavioral metrics
- **Multi-Agent Intelligence**: 6 specialized AI agents collaborate using Gemini 2.5-flash for deep analysis
- **Customer Segmentation**: Uses ML clustering to identify distinct customer groups
- **Persona Generation**: Creates detailed customer profiles with demographics, behaviors, and pain points
- **Marketing Strategy Development**: Generates targeted campaigns and messaging strategies
- **Quality Validation**: Ensures consistency and accuracy of generated personas

### AI Agents
1. **Data Analyst Agent**: Statistical analysis and pattern recognition
2. **Psychology Expert Agent**: Behavioral analysis and customer motivation insights
3. **Clustering Agent**: Machine learning segmentation and group identification
4. **Persona Creator Agent**: Comprehensive persona profile generation
5. **Marketing Strategy Agent**: Campaign development and messaging optimization
6. **Validation Agent**: Quality assurance and result validation

### Technology Stack
- **AI Model**: Google Gemini 2.5-flash for superior reasoning and analysis
- **AI Framework**: LangChain + LangGraph for agent orchestration
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Web Interface**: Streamlit with real-time monitoring
- **Visualization**: Plotly for analytics and insights

## Installation

### Prerequisites
- Python 3.8 or higher
- Google API key for Gemini access
- Internet connection for API calls

### Setup Instructions

1. **Clone Repository**
```bash
git clone <repository-url>
cd AgentForce_AgentZ
```

2. **Install Dependencies**
```bash
```bash
pip install -r requirements.txt
```

3. **API Key Setup**
Get your Google API key and configure:
```bash
# Copy the environment template
cp .env.example .env

# Edit .env file and add your Google API key
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

4. **Get Google API Key**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key to your `.env` file

## Usage

### Quick Start Demo
```bash
# Run enhanced demonstration with detailed output
python demo_enhanced.py
```

### Web Interface
```bash
# Launch interactive Streamlit application
streamlit run agentic_app.py
```

### AI-Powered Demo
```bash
# AI-powered persona generation (requires API key)
python demo_working.py
```

### Advanced Usage
```bash
# Full multi-agent system (requires API key)
python demo_simple.py
```

## Data Requirements

### Input Data Format
The system accepts CSV files with customer information including:

**Required Fields:**
- Customer ID or identifier
- Demographic data (age, location, income level)
- Behavioral metrics (purchase frequency, satisfaction scores)

**Optional Fields:**
- Product preferences
- Communication channel preferences  
- Feedback and survey responses
- Transaction history

**Sample Data Structure:**
```csv
customer_id,age,gender,location,income_bracket,purchase_frequency,satisfaction_score
1,35,Female,Urban,50-75K,Monthly,4
2,42,Male,Suburban,30-50K,Quarterly,3
```

See `Customer-survey-data.csv` for a complete example.

## Project Structure

```
AgentForce_AgentZ/
├── Core System
│   ├── agentic_persona_generator.py    # Multi-agent orchestration system
│   ├── open_source_llm.py             # LLM integration and management
│   └── requirements.txt               # Python dependencies
├── Applications
│   ├── agentic_app.py                 # Streamlit web interface
│   ├── demo_enhanced.py               # Enhanced demo with guaranteed output
│   ├── demo_working.py                # Simple AI demonstration script
│   └── demo_simple.py                 # Full agentic system demo
├── Configuration
│   ├── .env.example                   # Configuration template
│   └── .gitignore                     # Git ignore rules
├── Data & Output
│   ├── Customer-survey-data.csv       # Sample dataset
│   ├── enhanced_personas.json         # Demo output
│   └── persona_results.json           # AI-generated results
└── Documentation
    ├── README.md                      # This comprehensive guide
    └── OUTPUT_GUIDE.md                # Output storage locations
```

## System Workflow

## System Workflow

1. **Data Ingestion**: Load and validate customer data from CSV files
2. **Multi-Agent Processing**: 
   - Data Analyst performs statistical analysis
   - Psychology Expert identifies behavioral patterns
   - Clustering Agent creates customer segments
   - Persona Creator builds detailed profiles
   - Marketing Strategist develops campaigns
   - Validation Agent ensures quality
3. **Output Generation**: Comprehensive personas with marketing strategies
4. **Validation & Refinement**: Quality checks and consistency validation

## Configuration Options

### Language Model Providers

**HuggingFace (Recommended)**
- Easy setup and configuration
- No additional software required
- Models download automatically
- CPU and GPU support

**Ollama (Advanced)**
- Requires separate Ollama installation
- Better performance with larger models
- Local model management
- Advanced customization options

### Model Selection

### Model Selection

**Lightweight Models (Recommended for testing)**
- `microsoft/DialoGPT-small` - 350MB, fast processing
- `distilgpt2` - 240MB, basic capabilities

**High-Performance Models (Requires more resources)**  
- `microsoft/DialoGPT-medium` - 774MB, better quality
- `llama3.2:3b` - 2GB, professional-grade results (Ollama)
- `mistral:7b` - 4GB, enterprise-level analysis (Ollama)

## Performance Optimization

### System Requirements
- **Minimum**: 4GB RAM, 2GB storage
- **Recommended**: 8GB RAM, 5GB storage
- **Optimal**: 16GB RAM, 10GB storage, GPU acceleration

### Best Practices
1. **Start with lightweight models** for initial testing
2. **Clean your data** before processing for better results
3. **Use GPU acceleration** when available for larger models
4. **Monitor memory usage** during processing

## Expected Output

### Persona Structure
Each generated persona includes:

**Demographics Profile**
- Age range and gender distribution
- Geographic location patterns
- Income bracket analysis
- Education and occupation insights

**Behavioral Analysis**
- Purchase patterns and frequency
- Channel preferences (online, in-store, mobile)
- Product category preferences
- Engagement metrics

**Psychological Profile**
- Pain points and frustrations
- Motivations and drivers
- Decision-making patterns
- Communication preferences

**Marketing Strategy**
- Targeted messaging themes
- Optimal communication channels
- Campaign recommendations
- Content strategy suggestions

### Sample Output Format
```json
{
  "persona_name": "Tech-Savvy Millennials",
  "demographics": {
    "age_range": "25-35",
    "income": "$50K-$75K",
    "location": "Urban"
  },
  "behaviors": [
    "Frequent online shopping",
    "Mobile-first engagement",
    "Social media active"
  ],
  "pain_points": [
    "Limited time for research",
    "Information overload"
  ],
  "marketing_strategy": {
    "channels": ["Social Media", "Email", "Mobile App"],
    "messaging": "Convenience and efficiency focused"
  }
}
```

## Troubleshooting

## Troubleshooting

### Common Issues

**Installation Errors**
```bash
# Solution: Update pip and reinstall
python -m pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**Model Download Failures**
- Check internet connection
- Verify sufficient storage space
- Try switching to a smaller model

**Memory Issues**
```bash
# Switch to lightweight model in .env
LLM_MODEL=microsoft/DialoGPT-small
```

**Performance Problems**
- Close other applications to free RAM
- Use smaller batch sizes for data processing
- Consider GPU acceleration for large models

### Support

For technical issues:
1. Check the troubleshooting section above
2. Verify your environment configuration
3. Test with the simple demo first
4. Review system requirements

## Development and Contribution

### Development Setup
```bash
# Development mode installation
pip install -r requirements.txt
python -m pytest  # Run tests (if available)
```

### Architecture Notes
- **Modular Design**: Each agent is independently configurable
- **Extensible Framework**: Easy to add new agents or modify existing ones
- **Provider Agnostic**: Supports multiple LLM providers
- **Scalable Processing**: Handles datasets from hundreds to thousands of customers

## License and Usage

This project is designed for educational and business use. Please ensure compliance with your organization's data privacy and AI usage policies when processing customer data.

## Getting Started Checklist

- [ ] Install Python 3.8+
- [ ] Clone repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Configure `.env` file
- [ ] Test with demo (`python demo_working.py`)
- [ ] Upload your customer data
- [ ] Generate personas with web interface (`streamlit run agentic_app.py`)
- [ ] Review and export results

Ready to transform your customer data into actionable business insights!
