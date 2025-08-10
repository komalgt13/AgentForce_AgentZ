"""
Agentic AI Customer Persona Generator using LangChain and LangGraph
This system uses multiple AI agents working together to analyze customer data
and generate comprehensive personas with marketing strategies.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Open Source LLM integration
from open_source_llm import OpenSourceLLMManager, get_open_source_llm, get_open_source_embeddings

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
try:
    from langgraph.prebuilt import ToolExecutor, ToolInvocation
except ImportError:
    # Fallback for newer versions
    from langgraph.prebuilt.tool_node import ToolNode as ToolExecutor
    ToolInvocation = dict  # Use dict as fallback
from langgraph.checkpoint.memory import MemorySaver

# Pydantic for data models
from pydantic import BaseModel, Field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class PersonaState(TypedDict):
    """State object for the persona generation workflow"""
    raw_data: pd.DataFrame
    processed_data: Dict[str, Any]
    data_insights: Dict[str, Any]
    clustering_results: Dict[str, Any]
    personas: Dict[str, Any]
    marketing_strategies: Dict[str, Any]
    messages: Annotated[List[BaseMessage], "Messages in the conversation"]
    next_step: str
    errors: List[str]
    metadata: Dict[str, Any]

class CustomerInsight(BaseModel):
    """Data model for customer insights"""
    insight_type: str = Field(description="Type of insight (demographic, behavioral, satisfaction)")
    description: str = Field(description="Detailed description of the insight")
    supporting_data: Dict[str, Any] = Field(description="Supporting statistical data")
    confidence_score: float = Field(description="Confidence in this insight (0-1)")

class PersonaProfile(BaseModel):
    """Data model for a customer persona"""
    name: str = Field(description="Persona name")
    description: str = Field(description="Brief description")
    demographics: Dict[str, str] = Field(description="Demographic information")
    psychographics: Dict[str, Any] = Field(description="Psychological characteristics")
    behavior_patterns: List[str] = Field(description="Behavioral patterns")
    pain_points: List[str] = Field(description="Main frustrations and problems")
    goals_motivations: List[str] = Field(description="Goals and motivations")
    preferred_channels: List[str] = Field(description="Preferred communication channels")
    messaging_strategy: Dict[str, Any] = Field(description="Messaging recommendations")
    campaign_ideas: List[Dict[str, Any]] = Field(description="Marketing campaign suggestions")
    size_percentage: float = Field(description="Percentage of total customer base")

class AgenticPersonaGenerator:
    """
    Main class for the agentic AI persona generation system using open source LLMs
    """
    
    def __init__(self, model: str = None, provider: str = None, **kwargs):
        """Initialize the agentic system with open source LLM"""
        
        # Initialize LLM manager
        self.llm_manager = OpenSourceLLMManager()
        
        # Get model configuration from environment or parameters
        self.model = model or os.getenv("LLM_MODEL", "llama3.2:3b")
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        
        # Initialize LLM and embeddings
        try:
            self.llm = self.llm_manager.get_llm(
                model_name=self.model,
                provider=self.provider,
                temperature=kwargs.get('temperature', 0.3),
                max_tokens=kwargs.get('max_tokens', 2048)
            )
            
            self.embeddings = self.llm_manager.get_embeddings(
                model_name=kwargs.get('embedding_model'),
                provider=kwargs.get('embedding_provider')
            )
            
            print(f"SUCCESS: Initialized with {self.provider}/{self.model}")
            
        except Exception as e:
            print(f"ERROR: LLM initialization failed: {str(e)}")
            print("INFO: Make sure Ollama is running or HuggingFace models are available")
            raise
        
        self.memory = MemorySaver()
        
        # Initialize agents and tools
        self._setup_agents()
        self._setup_workflow()
    
    def _setup_agents(self):
        """Setup specialized AI agents"""
        
        # Data Analysis Agent
        self.data_analyst = self._create_agent(
            role="Senior Data Analyst",
            goal="Analyze customer data to extract meaningful insights and patterns",
            backstory="""You are an expert data analyst with 10+ years of experience in customer analytics. 
            You excel at finding hidden patterns in data and translating complex statistics into business insights.""",
            tools=self._get_data_analysis_tools()
        )
        
        # Customer Psychology Agent
        self.psychologist = self._create_agent(
            role="Customer Psychology Expert",
            goal="Understand customer motivations, behaviors, and psychological profiles",
            backstory="""You are a customer psychology expert who understands what drives customer behavior. 
            You can identify personality types, motivations, and emotional triggers from data patterns.""",
            tools=self._get_psychology_tools()
        )
        
        # Marketing Strategy Agent
        self.marketer = self._create_agent(
            role="Marketing Strategy Director",
            goal="Create targeted marketing strategies and campaigns for different customer segments",
            backstory="""You are a marketing strategy director with expertise in persona-based marketing. 
            You create compelling campaigns that resonate with specific customer segments.""",
            tools=self._get_marketing_tools()
        )
        
        # Persona Creation Agent
        self.persona_creator = self._create_agent(
            role="Customer Persona Specialist",
            goal="Synthesize insights into comprehensive customer personas",
            backstory="""You are a persona creation specialist who transforms data insights into vivid, 
            actionable customer personas that teams can easily understand and use.""",
            tools=self._get_persona_tools()
        )
    
    def _create_agent(self, role: str, goal: str, backstory: str, tools: List[Tool]):
        """Create a specialized agent with specific role and tools"""
        system_prompt = f"""
        Role: {role}
        Goal: {goal}
        Backstory: {backstory}
        
        You are part of a collaborative AI system generating customer personas. Work with other agents to:
        1. Provide your specialized expertise
        2. Build on insights from other agents
        3. Ensure the final personas are comprehensive and actionable
        
        Always provide specific, data-driven recommendations based on your analysis.
        """
        
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            system_message=system_prompt
        )
    
    def _get_data_analysis_tools(self) -> List[Tool]:
        """Tools for the data analysis agent"""
        return [
            Tool(
                name="analyze_satisfaction_patterns",
                description="Analyze customer satisfaction patterns and trends",
                func=self._analyze_satisfaction_patterns
            ),
            Tool(
                name="identify_customer_segments",
                description="Identify distinct customer segments using clustering",
                func=self._identify_customer_segments
            ),
            Tool(
                name="calculate_statistical_insights",
                description="Calculate key statistical insights from customer data",
                func=self._calculate_statistical_insights
            ),
            Tool(
                name="find_correlations",
                description="Find correlations between different customer attributes",
                func=self._find_correlations
            )
        ]
    
    def _get_psychology_tools(self) -> List[Tool]:
        """Tools for the customer psychology agent"""
        return [
            Tool(
                name="analyze_behavior_patterns",
                description="Analyze customer behavior patterns and motivations",
                func=self._analyze_behavior_patterns
            ),
            Tool(
                name="identify_pain_points",
                description="Identify customer pain points from satisfaction data",
                func=self._identify_pain_points
            ),
            Tool(
                name="determine_personality_traits",
                description="Determine personality traits from customer behavior",
                func=self._determine_personality_traits
            ),
            Tool(
                name="assess_emotional_triggers",
                description="Assess emotional triggers and motivations",
                func=self._assess_emotional_triggers
            )
        ]
    
    def _get_marketing_tools(self) -> List[Tool]:
        """Tools for the marketing strategy agent"""
        return [
            Tool(
                name="create_messaging_strategy",
                description="Create targeted messaging strategies for personas",
                func=self._create_messaging_strategy
            ),
            Tool(
                name="design_campaign_concepts",
                description="Design marketing campaign concepts for specific personas",
                func=self._design_campaign_concepts
            ),
            Tool(
                name="recommend_channels",
                description="Recommend optimal marketing channels for each persona",
                func=self._recommend_channels
            ),
            Tool(
                name="create_content_strategy",
                description="Create content strategy recommendations",
                func=self._create_content_strategy
            )
        ]
    
    def _get_persona_tools(self) -> List[Tool]:
        """Tools for the persona creation agent"""
        return [
            Tool(
                name="synthesize_persona_profile",
                description="Synthesize all insights into a comprehensive persona profile",
                func=self._synthesize_persona_profile
            ),
            Tool(
                name="validate_persona_consistency",
                description="Validate that persona characteristics are consistent and realistic",
                func=self._validate_persona_consistency
            ),
            Tool(
                name="generate_persona_narrative",
                description="Generate compelling narrative descriptions for personas",
                func=self._generate_persona_narrative
            )
        ]
    
    def _setup_workflow(self):
        """Setup the LangGraph workflow"""
        
        # Create the state graph
        workflow = StateGraph(PersonaState)
        
        # Add nodes for each step
        workflow.add_node("data_analysis", self._data_analysis_node)
        workflow.add_node("psychology_analysis", self._psychology_analysis_node)
        workflow.add_node("clustering", self._clustering_node)
        workflow.add_node("persona_creation", self._persona_creation_node)
        workflow.add_node("marketing_strategy", self._marketing_strategy_node)
        workflow.add_node("validation", self._validation_node)
        
        # Define the workflow edges
        workflow.add_edge("data_analysis", "psychology_analysis")
        workflow.add_edge("psychology_analysis", "clustering")
        workflow.add_edge("clustering", "persona_creation")
        workflow.add_edge("persona_creation", "marketing_strategy")
        workflow.add_edge("marketing_strategy", "validation")
        workflow.add_edge("validation", END)
        
        # Set entry point
        workflow.set_entry_point("data_analysis")
        
        # Compile the workflow
        self.workflow = workflow.compile(checkpointer=self.memory)
    
    def generate_personas(self, data: pd.DataFrame, n_personas: int = 4) -> Dict[str, Any]:
        """
        Main method to generate personas using the agentic workflow
        """
        
        # Initialize state
        initial_state = PersonaState(
            raw_data=data,
            processed_data={},
            data_insights={},
            clustering_results={},
            personas={},
            marketing_strategies={},
            messages=[HumanMessage(content=f"Generate {n_personas} customer personas from the provided data.")],
            next_step="data_analysis",
            errors=[],
            metadata={
                "n_personas": n_personas,
                "timestamp": datetime.now().isoformat(),
                "data_shape": data.shape
            }
        )
        
        # Run the workflow
        config = {"configurable": {"thread_id": f"persona_gen_{datetime.now().timestamp()}"}}
        final_state = self.workflow.invoke(initial_state, config=config)
        
        return {
            "personas": final_state["personas"],
            "marketing_strategies": final_state["marketing_strategies"],
            "insights": final_state["data_insights"],
            "metadata": final_state["metadata"],
            "errors": final_state["errors"]
        }
    
    # Workflow nodes
    def _data_analysis_node(self, state: PersonaState) -> PersonaState:
        """Data analysis workflow node"""
        try:
            # Process the raw data
            processed_data = self._preprocess_data(state["raw_data"])
            
            # Generate insights using the data analyst agent
            insights_prompt = f"""
            Analyze this customer data and provide key insights:
            - Data shape: {state["raw_data"].shape}
            - Columns: {list(state["raw_data"].columns)}
            - Sample statistics: {state["raw_data"].describe().to_string()}
            
            Focus on satisfaction patterns, customer segments, and behavioral indicators.
            """
            
            insights = self.data_analyst.run(insights_prompt)
            
            state["processed_data"] = processed_data
            state["data_insights"] = {"analysis": insights, "statistics": processed_data}
            state["messages"].append(AIMessage(content=f"Data analysis completed: {insights}"))
            
        except Exception as e:
            state["errors"].append(f"Data analysis error: {str(e)}")
        
        return state
    
    def _psychology_analysis_node(self, state: PersonaState) -> PersonaState:
        """Psychology analysis workflow node"""
        try:
            psychology_prompt = f"""
            Based on the data analysis insights, identify customer psychology patterns:
            
            Data insights: {state['data_insights']['analysis']}
            
            Analyze:
            1. Behavioral patterns and motivations
            2. Emotional drivers and triggers
            3. Pain points and frustrations
            4. Personality traits indicated by the data
            
            Provide psychological profiles for different customer segments.
            """
            
            psychology_analysis = self.psychologist.run(psychology_prompt)
            
            state["data_insights"]["psychology"] = psychology_analysis
            state["messages"].append(AIMessage(content=f"Psychology analysis completed: {psychology_analysis}"))
            
        except Exception as e:
            state["errors"].append(f"Psychology analysis error: {str(e)}")
        
        return state
    
    def _clustering_node(self, state: PersonaState) -> PersonaState:
        """Clustering workflow node"""
        try:
            # Perform clustering on the processed data
            clustering_results = self._perform_intelligent_clustering(
                state["raw_data"], 
                state["metadata"]["n_personas"]
            )
            
            state["clustering_results"] = clustering_results
            state["messages"].append(AIMessage(content="Customer segmentation clustering completed"))
            
        except Exception as e:
            state["errors"].append(f"Clustering error: {str(e)}")
        
        return state
    
    def _persona_creation_node(self, state: PersonaState) -> PersonaState:
        """Persona creation workflow node"""
        try:
            persona_prompt = f"""
            Create detailed customer personas based on:
            
            Data Insights: {state['data_insights']['analysis']}
            Psychology Analysis: {state['data_insights']['psychology']}
            Clustering Results: {state['clustering_results']}
            
            Create {state['metadata']['n_personas']} distinct personas, each with:
            1. Name and description
            2. Demographics
            3. Psychographics
            4. Behavior patterns
            5. Pain points
            6. Goals and motivations
            7. Size percentage
            
            Make each persona vivid and actionable for marketing teams.
            """
            
            personas_text = self.persona_creator.run(persona_prompt)
            
            # Parse and structure the personas
            personas = self._parse_personas(personas_text, state["clustering_results"])
            
            state["personas"] = personas
            state["messages"].append(AIMessage(content="Customer personas created successfully"))
            
        except Exception as e:
            state["errors"].append(f"Persona creation error: {str(e)}")
        
        return state
    
    def _marketing_strategy_node(self, state: PersonaState) -> PersonaState:
        """Marketing strategy workflow node"""
        try:
            marketing_prompt = f"""
            Create comprehensive marketing strategies for these personas:
            
            Personas: {json.dumps(state['personas'], indent=2)}
            
            For each persona, provide:
            1. Messaging strategy and tone
            2. Preferred communication channels
            3. Content strategy recommendations
            4. Campaign concepts with specific tactics
            5. Key performance indicators (KPIs)
            
            Focus on actionable, specific recommendations.
            """
            
            marketing_strategies = self.marketer.run(marketing_prompt)
            
            state["marketing_strategies"] = self._parse_marketing_strategies(marketing_strategies)
            state["messages"].append(AIMessage(content="Marketing strategies developed"))
            
        except Exception as e:
            state["errors"].append(f"Marketing strategy error: {str(e)}")
        
        return state
    
    def _validation_node(self, state: PersonaState) -> PersonaState:
        """Validation workflow node"""
        try:
            validation_prompt = f"""
            Validate the consistency and quality of these personas and marketing strategies:
            
            Personas: {json.dumps(state['personas'], indent=2)}
            Marketing Strategies: {json.dumps(state['marketing_strategies'], indent=2)}
            
            Check for:
            1. Internal consistency within each persona
            2. Distinctiveness between personas
            3. Alignment between personas and marketing strategies
            4. Actionability of recommendations
            5. Data support for claims
            
            Provide a quality score and any recommended improvements.
            """
            
            validation_results = self.persona_creator.run(validation_prompt)
            
            state["metadata"]["validation"] = validation_results
            state["messages"].append(AIMessage(content="Validation completed"))
            
        except Exception as e:
            state["errors"].append(f"Validation error: {str(e)}")
        
        return state
    
    # Helper methods (tool implementations)
    def _preprocess_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Preprocess the customer data"""
        processed = {}
        
        # Basic statistics
        processed["shape"] = data.shape
        processed["columns"] = list(data.columns)
        processed["dtypes"] = data.dtypes.to_dict()
        processed["missing_values"] = data.isnull().sum().to_dict()
        
        # Numeric column analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            processed["numeric_summary"] = data[numeric_cols].describe().to_dict()
        
        # Categorical column analysis
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            processed["categorical_summary"] = {}
            for col in categorical_cols:
                processed["categorical_summary"][col] = data[col].value_counts().head().to_dict()
        
        return processed
    
    def _perform_intelligent_clustering(self, data: pd.DataFrame, n_clusters: int) -> Dict[str, Any]:
        """Perform intelligent clustering with AI-enhanced analysis"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features for clustering
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {"error": "No numeric columns found for clustering"}
        
        # Clean and prepare data
        clustering_data = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Analyze clusters
        results = {
            "n_clusters": n_clusters,
            "cluster_labels": clusters.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "cluster_sizes": {},
            "cluster_characteristics": {}
        }
        
        # Calculate cluster statistics
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = clusters
        
        for cluster_id in range(n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            results["cluster_sizes"][cluster_id] = len(cluster_data)
            
            # Calculate cluster characteristics
            cluster_chars = {}
            for col in numeric_cols:
                cluster_chars[col] = {
                    "mean": float(cluster_data[col].mean()),
                    "std": float(cluster_data[col].std()),
                    "min": float(cluster_data[col].min()),
                    "max": float(cluster_data[col].max())
                }
            results["cluster_characteristics"][cluster_id] = cluster_chars
        
        return results
    
    def _parse_personas(self, personas_text: str, clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI-generated persona text into structured format"""
        # This is a simplified parser - in practice, you'd use more sophisticated NLP
        personas = {}
        
        # For now, create structured personas based on clustering results
        for cluster_id, size in clustering_results["cluster_sizes"].items():
            total_customers = sum(clustering_results["cluster_sizes"].values())
            percentage = (size / total_customers) * 100
            
            personas[f"persona_{cluster_id}"] = {
                "id": cluster_id,
                "name": f"Customer Segment {cluster_id + 1}",
                "description": f"Generated from AI analysis of customer data",
                "size_percentage": percentage,
                "cluster_characteristics": clustering_results["cluster_characteristics"][cluster_id],
                "ai_analysis": personas_text  # Store the full AI analysis
            }
        
        return personas
    
    def _parse_marketing_strategies(self, strategies_text: str) -> Dict[str, Any]:
        """Parse AI-generated marketing strategies"""
        return {
            "ai_generated_strategies": strategies_text,
            "timestamp": datetime.now().isoformat()
        }
    
    # Tool method implementations
    def _analyze_satisfaction_patterns(self, query: str) -> str:
        """Analyze customer satisfaction patterns"""
        return f"Satisfaction analysis for: {query}"
    
    def _identify_customer_segments(self, query: str) -> str:
        """Identify customer segments"""
        return f"Customer segmentation analysis for: {query}"
    
    def _calculate_statistical_insights(self, query: str) -> str:
        """Calculate statistical insights"""
        return f"Statistical insights for: {query}"
    
    def _find_correlations(self, query: str) -> str:
        """Find correlations in data"""
        return f"Correlation analysis for: {query}"
    
    def _analyze_behavior_patterns(self, query: str) -> str:
        """Analyze behavior patterns"""
        return f"Behavior pattern analysis for: {query}"
    
    def _identify_pain_points(self, query: str) -> str:
        """Identify pain points"""
        return f"Pain point analysis for: {query}"
    
    def _determine_personality_traits(self, query: str) -> str:
        """Determine personality traits"""
        return f"Personality trait analysis for: {query}"
    
    def _assess_emotional_triggers(self, query: str) -> str:
        """Assess emotional triggers"""
        return f"Emotional trigger analysis for: {query}"
    
    def _create_messaging_strategy(self, query: str) -> str:
        """Create messaging strategy"""
        return f"Messaging strategy for: {query}"
    
    def _design_campaign_concepts(self, query: str) -> str:
        """Design campaign concepts"""
        return f"Campaign concepts for: {query}"
    
    def _recommend_channels(self, query: str) -> str:
        """Recommend marketing channels"""
        return f"Channel recommendations for: {query}"
    
    def _create_content_strategy(self, query: str) -> str:
        """Create content strategy"""
        return f"Content strategy for: {query}"
    
    def _synthesize_persona_profile(self, query: str) -> str:
        """Synthesize persona profile"""
        return f"Persona profile synthesis for: {query}"
    
    def _validate_persona_consistency(self, query: str) -> str:
        """Validate persona consistency"""
        return f"Persona validation for: {query}"
    
    def _generate_persona_narrative(self, query: str) -> str:
        """Generate persona narrative"""
        return f"Persona narrative for: {query}"

# Example usage
if __name__ == "__main__":
    # Load sample data
    try:
        data = pd.read_csv("Customer-survey-data.csv")
        print(f"Loaded {len(data)} customer records")
        
        # Initialize the agentic system with open source LLM
        print("Initializing open source AI agents...")
        generator = AgenticPersonaGenerator(
            model="llama3.2:3b",  # or "microsoft/DialoGPT-medium"
            provider="ollama"     # or "huggingface"
        )
        
        # Generate personas
        print("Generating personas with open source AI agents...")
        results = generator.generate_personas(data, n_personas=4)
        
        print("Open Source Agentic AI Persona Generation Complete!")
        print(f"Generated {len(results['personas'])} personas")
        
        # Save results
        with open("open_source_personas.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Results saved to open_source_personas.json")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure you have Ollama running with llama3.2:3b model installed")
        print("Or install HuggingFace transformers: pip install transformers torch")
