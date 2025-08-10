# AI Persona Generator - Output Storage Guide

## Output Storage Locations

### 1. Enhanced Demo Output (RECOMMENDED) 
**File**: `enhanced_personas.json`
**Created by**: `python demo_enhanced.py`
**Location**: Same directory as the script
**Content**: 
- 3 detailed customer personas with comprehensive profiles
- Demographics, behaviors, pain points, motivations
- Marketing strategies and campaign ideas
- Complete segment analysis and statistics

### 2. Simple Demo Output
**File**: `persona_results.json`
**Created by**: `python demo_working.py`
**Location**: Same directory as the script
**Content**: 
- Basic AI responses (may be minimal with small models)
- Customer data summary
- Timestamp and model information

### 3. Full Multi-Agent System Output  
**File**: `open_source_personas.json`
**Created by**: `python demo_simple.py` (full system)
**Location**: Same directory as the script
**Content**: 
- Complete persona profiles from 6-agent collaboration
- Advanced behavioral analysis
- Detailed marketing campaigns
- AI validation results

### 3. Streamlit Web App Output
**File**: `customer_personas.json` 
**Created by**: `streamlit run agentic_app.py`
**Location**: User's Downloads folder (via browser download)
**Content**: 
- Interactive web interface results
- Downloadable JSON format
- Real-time generated personas

## File Formats

### JSON Structure Example
```json
{
  "timestamp": "2025-08-10T07:59:30.176200",
  "model_used": "microsoft/DialoGPT-small",
  "customer_data_summary": {
    "total_customers": 50,
    "columns": ["customer_id", "age", "gender", ...]
  },
  "ai_generated_personas": "Detailed persona descriptions...",
  "marketing_strategies": "Campaign recommendations..."
}
```

## Output File Locations Summary

| Demo Type | Output File | Location | Content |
|-----------|-------------|----------|---------|
| **Simple Demo** | `persona_results.json` | `./persona_results.json` | Basic AI analysis |
| **Full System** | `open_source_personas.json` | `./open_source_personas.json` | Complete personas |
| **Web Interface** | `customer_personas.json` | Downloads folder | Interactive results |

## Accessing Your Results

1. **After running demo**: Check current directory for JSON files
2. **File paths**: Full absolute paths are printed during execution  
3. **Content**: Open JSON files in any text editor or JSON viewer
4. **Import**: Results can be imported into other applications

## Next Steps

- View results in JSON format
- Import into business intelligence tools
- Use for marketing campaign development
- Share with stakeholders
