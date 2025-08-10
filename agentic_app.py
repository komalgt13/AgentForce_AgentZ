import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agentic_persona_generator import AgenticPersonaGenerator
from open_source_llm import list_available_models
import json
import os
from datetime import datetime
import asyncio
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Agentic AI Persona Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page configuration
st.set_page_config(
    page_title="🤖 Agentic AI Persona Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 2rem;
}
.agent-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}
.persona-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 6px solid #ff6b6b;
    margin-bottom: 1rem;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}
.insight-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
}
.workflow-step {
    background-color: #f8f9fa;
    padding: 1rem;
    border-left: 4px solid #28a745;
    margin-bottom: 1rem;
    border-radius: 5px;
}
.agent-status {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: bold;
}
.status-working {
    background-color: #ffc107;
    color: #856404;
}
.status-complete {
    background-color: #28a745;
    color: white;
}
.status-error {
    background-color: #dc3545;
    color: white;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'agentic_generator' not in st.session_state:
        st.session_state.agentic_generator = None
    if 'personas_result' not in st.session_state:
        st.session_state.personas_result = None
    if 'workflow_status' not in st.session_state:
        st.session_state.workflow_status = {}
    if 'llm_ready' not in st.session_state:
        st.session_state.llm_ready = False

def check_llm_availability(provider: str, model: str) -> bool:
    """Check if the specified LLM is available"""
    try:
        from open_source_llm import OpenSourceLLMManager
        manager = OpenSourceLLMManager()
        
        if provider == 'ollama':
            # Check if Ollama is running
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                return model in models
        elif provider == 'huggingface':
            # HuggingFace models are generally available if transformers is installed
            try:
                import transformers
                return True
            except ImportError:
                return False
        return False
    except Exception:
        return False

def display_agent_workflow():
    """Display the AI agent workflow"""
    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Agent Workflow")
    st.markdown("Our multi-agent AI system uses specialized agents working together:")
    st.markdown('</div>', unsafe_allow_html=True)
    
    agents = [
        ("📊 Data Analyst Agent", "Analyzes patterns and extracts insights from customer data", "data_analysis"),
        ("🧠 Psychology Agent", "Understands customer motivations and behavioral patterns", "psychology"),
        ("🎯 Clustering Agent", "Groups customers into distinct segments using ML", "clustering"),
        ("👥 Persona Creator", "Synthesizes insights into comprehensive personas", "personas"),
        ("📈 Marketing Strategist", "Develops targeted campaigns and messaging", "marketing"),
        ("✅ Validation Agent", "Ensures quality and consistency of results", "validation")
    ]
    
    cols = st.columns(3)
    for i, (name, description, key) in enumerate(agents):
        with cols[i % 3]:
            status = st.session_state.workflow_status.get(key, "pending")
            if status == "working":
                status_class = "status-working"
                status_text = "Working..."
            elif status == "complete":
                status_class = "status-complete"
                status_text = "Complete"
            elif status == "error":
                status_class = "status-error"
                status_text = "Error"
            else:
                status_class = "status-working"
                status_text = "Pending"
            
            st.markdown(f"""
            <div class="workflow-step">
                <h4>{name}</h4>
                <p>{description}</p>
                <span class="agent-status {status_class}">{status_text}</span>
            </div>
            """, unsafe_allow_html=True)

def display_real_time_insights(insights: Dict[str, Any]):
    """Display real-time insights from AI agents"""
    st.subheader("🔍 Real-time AI Insights")
    
    if insights:
        for insight_type, insight_data in insights.items():
            st.markdown(f'<div class="insight-card">', unsafe_allow_html=True)
            st.markdown(f"**{insight_type.title()}:** {insight_data}")
            st.markdown('</div>', unsafe_allow_html=True)

def display_agentic_personas(results: Dict[str, Any]):
    """Display AI-generated personas with enhanced formatting"""
    personas = results.get('personas', {})
    marketing_strategies = results.get('marketing_strategies', {})
    insights = results.get('insights', {})
    
    # Display overview
    st.header("🎭 AI-Generated Customer Personas")
    
    # Insights summary
    if insights:
        st.subheader("🧠 AI Analysis Summary")
        with st.expander("View Detailed AI Analysis"):
            if 'analysis' in insights:
                st.write("**Data Analysis:**")
                st.write(insights['analysis'])
            if 'psychology' in insights:
                st.write("**Psychology Analysis:**")
                st.write(insights['psychology'])
    
    # Display personas
    if personas:
        st.subheader(f"👥 Generated Personas ({len(personas)} total)")
        
        # Create tabs for each persona
        persona_names = [f"Persona {i+1}" for i in range(len(personas))]
        tabs = st.tabs(persona_names)
        
        for i, (persona_id, persona) in enumerate(personas.items()):
            with tabs[i]:
                display_enhanced_persona(persona, marketing_strategies)
    
    # Display validation results
    if 'metadata' in results and 'validation' in results['metadata']:
        st.subheader("✅ AI Validation Results")
        st.write(results['metadata']['validation'])

def display_enhanced_persona(persona: Dict[str, Any], marketing_strategies: Dict[str, Any]):
    """Display an enhanced persona with AI insights"""
    
    st.markdown(f'<div class="persona-card">', unsafe_allow_html=True)
    st.subheader(f"🎭 {persona.get('name', 'Unknown Persona')}")
    st.write(f"**Description:** {persona.get('description', 'AI-generated customer persona')}")
    
    if 'size_percentage' in persona:
        st.write(f"**Market Share:** {persona['size_percentage']:.1f}% of customers")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster characteristics
        if 'cluster_characteristics' in persona:
            st.subheader("📊 Data-Driven Characteristics")
            characteristics = persona['cluster_characteristics']
            
            for metric, values in characteristics.items():
                if isinstance(values, dict):
                    st.metric(
                        label=metric.replace('_', ' ').title(),
                        value=f"{values.get('mean', 0):.2f}",
                        delta=f"±{values.get('std', 0):.2f}"
                    )
        
        # AI Analysis
        if 'ai_analysis' in persona:
            st.subheader("🤖 AI Analysis")
            with st.expander("View AI-Generated Insights"):
                st.write(persona['ai_analysis'])
    
    with col2:
        # Marketing strategies
        st.subheader("🎯 AI-Generated Marketing Strategy")
        if marketing_strategies and 'ai_generated_strategies' in marketing_strategies:
            with st.expander("View Marketing Recommendations"):
                st.write(marketing_strategies['ai_generated_strategies'])
        
        # Recommended actions
        st.subheader("💡 Recommended Actions")
        st.write("• Develop targeted messaging campaigns")
        st.write("• Create personalized content strategies")
        st.write("• Optimize communication channels")
        st.write("• Design specific product offerings")

def main():
    st.markdown('<h1 class="main-header">AI Customer Persona Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Next-Generation AI-Powered Persona Creation** 
    
    This advanced system uses multiple specialized AI agents working together through LangChain and LangGraph 
    to analyze your customer data and create comprehensive, actionable personas with targeted marketing strategies.
    """)
    
    # Initialize session state
    initialize_session_state()
    
    # LLM Configuration
    st.sidebar.header("Open Source LLM Configuration")
    
    # Provider selection
    provider = st.sidebar.selectbox(
        "LLM Provider",
        ["ollama", "huggingface"],
        help="Choose your open source LLM provider"
    )
    
    # Get available models
    try:
        available_models = list_available_models()
        provider_models = available_models.get(provider, [])
    except:
        provider_models = []
    
    # Model selection
    if provider == "ollama":
        default_models = ["llama3.2:3b", "llama3.2:7b", "mistral:7b", "codellama:7b"]
        model_options = provider_models if provider_models else default_models
        default_model = "llama3.2:3b"
    else:  # huggingface
        default_models = ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-small", "distilgpt2", "gpt2"]
        model_options = provider_models if provider_models else default_models
        default_model = "microsoft/DialoGPT-small"
    
    selected_model = st.sidebar.selectbox(
        "Model",
        model_options,
        index=0 if default_model not in model_options else model_options.index(default_model),
        help=f"Choose your {provider} model"
    )
    
    # Check model availability
    model_available = check_llm_availability(provider, selected_model)
    
    if model_available:
        st.sidebar.success(f"✅ {provider}/{selected_model} available")
        st.session_state.llm_ready = True
        
        # Initialize agentic generator
        if st.session_state.agentic_generator is None:
            try:
                with st.sidebar.spinner("Initializing AI agents..."):
                    st.session_state.agentic_generator = AgenticPersonaGenerator(
                        model=selected_model,
                        provider=provider
                    )
                st.sidebar.success("🤖 AI Agents Initialized")
            except Exception as e:
                st.sidebar.error(f"Error initializing AI: {str(e)}")
                st.session_state.llm_ready = False
    else:
        st.sidebar.error(f"❌ {provider}/{selected_model} not available")
        st.session_state.llm_ready = False
        
        # Show setup instructions
        if provider == "ollama":
            st.sidebar.info("""
            **Setup Ollama:**
            1. Install Ollama: https://ollama.com
            2. Run: `ollama pull llama3.2:3b`
            3. Start Ollama service
            """)
        else:
            st.sidebar.info("""
            **Setup HuggingFace:**
            1. Install: `pip install transformers torch`
            2. Models download automatically
            """)
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.number_input("Max Tokens", 100, 4096, 2048)
        
        if provider == "ollama":
            base_url = st.text_input("Ollama Base URL", "http://localhost:11434")
        
        use_gpu = st.checkbox("Use GPU (if available)", value=True)
    
    # Data Upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Customer Data (CSV)",
        type=['csv'],
        help="Upload your customer data CSV file"
    )
    
    use_sample = st.sidebar.checkbox("Use Sample Restaurant Data", value=True if not uploaded_file else False)
    
    # Configuration
    n_personas = st.sidebar.slider(
        "Number of Personas",
        min_value=2,
        max_value=6,
        value=4,
        help="Number of customer personas to generate"
    )
    
    # Display agent workflow
    display_agent_workflow()
    
    # Main content area
    if not st.session_state.llm_ready:
        st.info("👆 Please configure and test your open source LLM in the sidebar.")
        
        # Show model status
        st.subheader("🤖 Open Source LLM Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ollama Status:**")
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    st.success(f"✅ Running ({len(models)} models available)")
                    for model in models[:3]:  # Show first 3 models
                        st.write(f"• {model['name']}")
                else:
                    st.error("❌ Not responding")
            except:
                st.error("❌ Not running")
                st.info("Install Ollama: https://ollama.com")
        
        with col2:
            st.markdown("**HuggingFace Status:**")
            try:
                import transformers
                st.success("✅ Transformers library available")
                st.info("Models will download automatically")
            except ImportError:
                st.error("❌ Transformers not installed")
                st.info("Install: `pip install transformers torch`")
        
        return
    
    # Load data
    data = None
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(data)} records from uploaded file")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    elif use_sample:
        try:
            data = pd.read_csv("Customer-survey-data.csv")
            st.success(f"✅ Loaded {len(data)} records from sample data")
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            return
    else:
        st.info("👆 Please upload a CSV file or use the sample data.")
        return
    
    # Data preview
    with st.expander("📊 Data Preview"):
        st.write(f"**Shape:** {data.shape}")
        st.write(f"**Columns:** {', '.join(data.columns)}")
        st.dataframe(data.head())
    
    # Generate personas
    if st.button("🚀 Generate AI Personas", type="primary", disabled=not st.session_state.llm_ready):
        
        # Reset workflow status
        st.session_state.workflow_status = {
            "data_analysis": "working",
            "psychology": "pending",
            "clustering": "pending", 
            "personas": "pending",
            "marketing": "pending",
            "validation": "pending"
        }
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with st.spinner("🤖 AI Agents are analyzing your data..."):
                # Update status
                status_text.text("🔍 Data Analysis Agent is processing...")
                st.session_state.workflow_status["data_analysis"] = "working"
                progress_bar.progress(0.16)
                
                # Generate personas using agentic system
                results = st.session_state.agentic_generator.generate_personas(
                    data, 
                    n_personas=n_personas
                )
                
                # Update progress through workflow
                workflow_steps = [
                    ("psychology", "🧠 Psychology Agent analyzing behavior..."),
                    ("clustering", "🎯 Clustering Agent segmenting customers..."),
                    ("personas", "👥 Persona Creator synthesizing profiles..."),
                    ("marketing", "📈 Marketing Strategist developing campaigns..."),
                    ("validation", "✅ Validation Agent ensuring quality...")
                ]
                
                for i, (step, message) in enumerate(workflow_steps, 1):
                    status_text.text(message)
                    st.session_state.workflow_status[step] = "working"
                    progress_bar.progress((i + 1) * 0.16)
                    
                    # Simulate processing time for demo
                    import time
                    time.sleep(1)
                    
                    st.session_state.workflow_status[step] = "complete"
                
                st.session_state.workflow_status["data_analysis"] = "complete"
                st.session_state.personas_result = results
                
                progress_bar.progress(1.0)
                status_text.text("🎉 AI Persona Generation Complete!")
                
        except Exception as e:
            st.error(f"Error generating personas: {str(e)}")
            st.session_state.workflow_status = {k: "error" for k in st.session_state.workflow_status.keys()}
    
    # Display results
    if st.session_state.personas_result:
        display_agentic_personas(st.session_state.personas_result)
        
        # Export options
        st.header("📥 Export AI Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save to JSON"):
                filename = f"agentic_personas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, "w") as f:
                    json.dump(st.session_state.personas_result, f, indent=2, default=str)
                st.success(f"Saved to {filename}")
        
        with col2:
            # Create downloadable JSON
            personas_json = json.dumps(st.session_state.personas_result, indent=2, default=str)
            st.download_button(
                label="⬇️ Download Results",
                data=personas_json,
                file_name=f"agentic_personas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by Open Source LLMs • LangChain & LangGraph • Built with Streamlit</p>
        <p>Advanced Multi-Agent AI System for Customer Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
