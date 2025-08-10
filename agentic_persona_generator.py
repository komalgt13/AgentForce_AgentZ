"""
Agentic AI Customer Persona Generator using LangChain and LangGraph
This system uses multiple AI agents working together to analyze customer data
and generate comprehensive personas with marketing strategies.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Gemini LLM integration
from open_source_llm import GeminiLLMManager

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

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
    messages: List[BaseMessage]
    next_step: str
    errors: List[str]
    metadata: Dict[str, Any]


class AgenticPersonaGenerator:
    """
    Main class for the agentic AI persona generation system using open source LLMs
    """
    
    def __init__(self, model: str = None, api_key: str = None, **kwargs):
        """Initialize the agentic system with Gemini LLM"""
        
        # Initialize Gemini LLM manager
        self.llm_manager = GeminiLLMManager(api_key=api_key)
        
        # Get model configuration from environment or parameters
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        # Initialize LLM and embeddings (use LangChain-compatible wrapper)
        try:
            # Use LangChain-compatible wrapper that falls back to DirectGeminiWrapper
            self.llm = self.llm_manager.get_llm(
                model_name=self.model,
                temperature=kwargs.get('temperature', 0.3),
                max_tokens=kwargs.get('max_tokens', 2048),
                use_langchain=True  # This will try ChatGoogleGenerativeAI, then fallback to compatible wrapper
            )
            
            self.embeddings = self.llm_manager.get_embeddings(
                model_name=kwargs.get('embedding_model', "models/embedding-001")
            )
            
            print(f"SUCCESS: Initialized with Gemini/{self.model} using LangChain-compatible wrapper")
            
        except Exception as e:
            print(f"ERROR: Gemini LLM initialization failed: {str(e)}")
            print("INFO: Make sure GOOGLE_API_KEY is set in your environment")
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
        ## AGENT ROLE AND IDENTITY
        Role: {role}
        Primary Objective: {goal}
        Professional Background: {backstory}
        
        ## COLLABORATION FRAMEWORK
        You are an expert agent within a multi-agent customer persona generation system. Your responsibilities:
        
        ### Core Duties:
        1. **Domain Expertise**: Apply your specialized knowledge to analyze customer data
        2. **Collaborative Intelligence**: Integrate insights from other agents to enhance analysis
        3. **Quality Assurance**: Ensure outputs are comprehensive, actionable, and business-ready
        4. **Data-Driven Insights**: Base all recommendations on quantifiable evidence and patterns
        
        ### Output Requirements:
        - Provide concrete, measurable insights with supporting data points
        - Use professional business language appropriate for executive presentations
        - Structure responses with clear headers and bullet points
        - Include confidence levels and data limitations where relevant
        - Suggest actionable next steps for implementation
        
        ### Communication Protocol:
        - Begin responses with your agent role for clarity
        - Reference specific data points and metrics
        - Highlight key findings and their business implications
        - End with concrete recommendations
        """
        
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Use simpler agent type
            verbose=True,
            max_iterations=3,  # Limit iterations to prevent issues
            early_stopping_method="generate"  # Stop on first generation
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
            "detailed_personas": final_state["personas"],
            "marketing_strategies": final_state["marketing_strategies"],
            "insights": final_state["data_insights"],
            "metadata": final_state["metadata"],
            "errors": final_state["errors"]
        }
    
    def generate_personas_from_csv(self, csv_file_path: str, n_personas: int = 4) -> Dict[str, Any]:
        """
        Generate personas from CSV file path using direct agent execution
        """
        try:
            # Load the CSV data
            data = pd.read_csv(csv_file_path)
            print(f"✅ Loaded {len(data)} customer records from {csv_file_path}")
            
            # Execute the pipeline directly without LangGraph serialization
            print("🤖 Running Data Analysis Agent...")
            data_analysis = self._run_data_analysis_directly(data)
            
            print("🤖 Running Psychology Analysis Agent...")
            psychology_analysis = self._run_psychology_analysis_directly(data, data_analysis)
            
            print("🤖 Running Clustering Agent...")
            clustering_results = self._perform_intelligent_clustering(data, n_personas)
            
            print("🤖 Running Persona Creation Agent...")
            personas = self._run_persona_creation_directly(data_analysis, psychology_analysis, clustering_results)
            
            print("🤖 Running Marketing Strategy Agent...")
            marketing_strategies = self._run_marketing_strategy_directly(personas)
            
            print("🤖 Running Validation Agent...")
            validation_results = self._run_validation_directly(personas, marketing_strategies)
            
            return {
                "detailed_personas": personas,
                "marketing_strategies": marketing_strategies,
                "insights": {
                    "data_analysis": data_analysis,
                    "psychology": psychology_analysis,
                    "clustering": clustering_results
                },
                "metadata": {
                    "n_personas": n_personas,
                    "timestamp": datetime.now().isoformat(),
                    "data_shape": [len(data), len(data.columns)],
                    "csv_file": csv_file_path,
                    "validation": validation_results
                },
                "errors": []
            }
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            print(f"❌ Error in agentic pipeline: {error_msg}")
            return {
                "detailed_personas": {},
                "marketing_strategies": {},
                "insights": {},
                "metadata": {"error": error_msg},
                "errors": [error_msg]
            }
    
    def _run_data_analysis_directly(self, data: pd.DataFrame) -> str:
        """Run data analysis agent directly"""
        insights_prompt = f"""
        ## DATA ANALYSIS REQUEST
        
        **Objective**: Conduct comprehensive customer data analysis for persona generation
        
        **Dataset Overview**:
        - Records: {len(data):,} customers
        - Attributes: {len(data.columns)} data points
        - Features: {', '.join(list(data.columns))}
        
        **Statistical Summary**:
        {data.describe().to_string()}
        
        **Required Analysis**: Provide key insights about customer segments, satisfaction patterns, 
        and behavioral indicators that will inform persona development.
        """
        
        return self.data_analyst.run(insights_prompt)
    
    def _run_psychology_analysis_directly(self, data: pd.DataFrame, data_analysis: str) -> str:
        """Run psychology analysis agent directly"""
        psychology_prompt = f"""
        ## CUSTOMER PSYCHOLOGY ANALYSIS REQUEST
        
        **Data Foundation**: {data_analysis}
        
        **Required Assessment**: Develop psychological profiles for customer segments based on the data patterns.
        Focus on motivations, emotional drivers, pain points, and personality traits.
        """
        
        return self.psychologist.run(psychology_prompt)
    
    def _run_persona_creation_directly(self, data_analysis: str, psychology_analysis: str, clustering_results: Dict) -> Dict:
        """Run persona creation agent directly"""
        persona_prompt = f"""
        ## CUSTOMER PERSONA CREATION REQUEST
        
        **Data Analytics**: {data_analysis}
        **Psychology Insights**: {psychology_analysis}
        **Segmentation**: {json.dumps(clustering_results, indent=2)}
        
        **Task**: Create {len(clustering_results.get('cluster_sizes', {}))} comprehensive customer personas 
        with demographics, behaviors, pain points, and opportunities.
        """
        
        personas_text = self.persona_creator.run(persona_prompt)
        
        # Parse into structured format
        personas = {}
        for i, (cluster_id, size) in enumerate(clustering_results.get('cluster_sizes', {}).items()):
            total = sum(clustering_results['cluster_sizes'].values())
            personas[f"persona_{i+1}"] = {
                "name": f"Customer Segment {i+1}",
                "cluster_id": cluster_id,
                "size_percentage": (size / total) * 100,
                "description": personas_text,
                "characteristics": clustering_results['cluster_characteristics'].get(cluster_id, {})
            }
        
        return personas
    
    def _run_marketing_strategy_directly(self, personas: Dict) -> Dict:
        """Run marketing strategy agent directly"""
        strategy_prompt = f"""
        ## MARKETING STRATEGY DEVELOPMENT REQUEST
        
        **Target Personas**: {json.dumps(personas, indent=2)}
        
        **Task**: Create comprehensive marketing strategies for each persona including 
        channels, messaging, campaigns, and success metrics.
        """
        
        strategies_text = self.marketer.run(strategy_prompt)
        
        return {
            "comprehensive_strategies": strategies_text,
            "timestamp": datetime.now().isoformat()
        }
    
    def _run_validation_directly(self, personas: Dict, marketing_strategies: Dict) -> str:
        """Run validation agent directly"""
        validation_prompt = f"""
        ## VALIDATION REQUEST
        
        **Personas**: {json.dumps(personas, indent=2)}
        **Strategies**: {json.dumps(marketing_strategies, indent=2)}
        
        **Task**: Validate the quality and consistency of personas and marketing strategies.
        """
        
        return self.persona_creator.run(validation_prompt)
    
    # Workflow nodes
    def _data_analysis_node(self, state: PersonaState) -> PersonaState:
        """Data analysis workflow node"""
        try:
            # Convert serialized data back to DataFrame for processing
            if isinstance(state["raw_data"], dict):
                raw_data = pd.DataFrame(state["raw_data"]["data"])
            else:
                raw_data = state["raw_data"]
            
            # Process the raw data
            processed_data = self._preprocess_data(raw_data)
            
            # Generate insights using the data analyst agent
            insights_prompt = f"""
            ## DATA ANALYSIS REQUEST
            
            **Objective**: Conduct comprehensive customer data analysis for persona generation
            
            **Dataset Overview**:
            - Records: {raw_data.shape[0]:,} customers
            - Attributes: {raw_data.shape[1]} data points
            - Features: {', '.join(list(raw_data.columns))}
            
            **Statistical Summary**:
            {raw_data.describe().to_string()}
            
            **Required Analysis**:
            
            ### 1. Customer Satisfaction Patterns
            - Identify satisfaction distribution and key drivers
            - Correlate satisfaction with other customer attributes
            - Flag any concerning satisfaction trends
            
            ### 2. Customer Segmentation Indicators
            - Detect natural groupings based on behavior and demographics
            - Calculate segment sizes and characteristics
            - Identify high-value customer segments
            
            ### 3. Behavioral Insights
            - Purchase frequency patterns and their implications
            - Spending behavior analysis across different customer groups
            - Age and location influences on customer behavior
            
            **Deliverable Format**:
            Provide findings in structured sections with:
            - Key metrics and percentages
            - Business implications for each finding
            - Recommendations for persona development
            - Data quality notes and limitations
            """
            
            insights = self.data_analyst.run(insights_prompt)
            
            state["processed_data"] = processed_data
            state["data_insights"] = {"analysis": insights, "statistics": processed_data}
            state["messages"].append(AIMessage(content=f"Data analysis completed: {insights}"))
            
        except Exception as e:
            error_msg = f"Data analysis error: {str(e)}"
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")
        
        return state
    
    def _psychology_analysis_node(self, state: PersonaState) -> PersonaState:
        """Psychology analysis workflow node"""
        try:
            psychology_prompt = f"""
            ## CUSTOMER PSYCHOLOGY ANALYSIS REQUEST
            
            **Objective**: Develop psychological profiles for customer segments based on data analysis
            
            **Data Analysis Foundation**:
            {state['data_insights']['analysis']}
            
            **Required Psychological Assessment**:
            
            ### 1. Behavioral Pattern Analysis
            - **Purchase Motivations**: What drives customers to buy (necessity, desire, social pressure, etc.)?
            - **Decision-Making Styles**: Quick impulse buyers vs. deliberate researchers
            - **Brand Relationship**: Loyalty patterns and switching triggers
            
            ### 2. Emotional Driver Identification
            - **Primary Emotions**: Fear, joy, security, status, convenience motivating purchases
            - **Emotional Triggers**: Specific situations or feelings that prompt buying behavior
            - **Satisfaction Sources**: What creates positive emotional responses post-purchase
            
            ### 3. Pain Point Psychology
            - **Frustration Sources**: Specific customer journey friction points
            - **Anxiety Factors**: Concerns about price, quality, time, or social perception
            - **Unmet Needs**: Gaps between expectations and current market offerings
            
            ### 4. Personality Trait Mapping
            - **Risk Tolerance**: Conservative vs. adventurous in purchase decisions
            - **Social Influence**: Individual vs. community-driven purchasing
            - **Communication Preferences**: Direct/informational vs. emotional/story-driven messaging
            
            **Deliverable Requirements**:
            - Create distinct psychological profiles for each customer segment
            - Link psychological traits to observable data patterns
            - Provide confidence levels for psychological assessments
            - Include implications for marketing message development
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
            # Convert serialized data back to DataFrame
            if isinstance(state["raw_data"], dict):
                raw_data = pd.DataFrame(state["raw_data"]["data"])
            else:
                raw_data = state["raw_data"]
            
            # Perform clustering on the processed data
            clustering_results = self._perform_intelligent_clustering(
                raw_data, 
                state["metadata"]["n_personas"]
            )
            
            state["clustering_results"] = clustering_results
            state["messages"].append(AIMessage(content="Customer segmentation clustering completed"))
            
        except Exception as e:
            error_msg = f"Clustering error: {str(e)}"
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")
        
        return state
    
    def _persona_creation_node(self, state: PersonaState) -> PersonaState:
        """Persona creation workflow node"""
        try:
            persona_prompt = f"""
            ## CUSTOMER PERSONA CREATION REQUEST
            
            **Objective**: Develop {state['metadata']['n_personas']} comprehensive, actionable customer personas
            
            **Source Intelligence**:
            
            ### Data Analytics Foundation:
            {state['data_insights']['analysis']}
            
            ### Psychological Insights:
            {state['data_insights']['psychology']}
            
            ### Segmentation Results:
            {state['clustering_results']}
            
            **Persona Development Requirements**:
            
            For each of the {state['metadata']['n_personas']} personas, provide:
            
            ### 1. Identity & Demographics
            - **Persona Name**: Memorable, representative name
            - **Age Range**: Specific age brackets with median
            - **Geographic Distribution**: Primary locations and preferences
            - **Income Level**: Annual income ranges and spending capacity
            - **Education & Occupation**: Professional background and expertise levels
            
            ### 2. Psychographic Profile
            - **Core Values**: What matters most to this segment
            - **Lifestyle Choices**: Daily routines, hobbies, interests
            - **Technology Adoption**: Digital comfort level and platform preferences
            - **Social Dynamics**: Family status, social network influence
            
            ### 3. Behavioral Patterns
            - **Purchase Journey**: How they research, evaluate, and buy
            - **Brand Interactions**: Preferred touchpoints and communication frequency
            - **Decision Factors**: Primary criteria influencing purchase decisions
            - **Loyalty Drivers**: What keeps them engaged long-term
            
            ### 4. Challenges & Opportunities
            - **Primary Pain Points**: Specific frustrations with current market options
            - **Unmet Needs**: Gaps between desires and available solutions
            - **Success Metrics**: How they define value and satisfaction
            - **Growth Potential**: Opportunities for increased engagement
            
            ### 5. Segment Metrics
            - **Market Share**: Percentage of total customer base
            - **Revenue Contribution**: Average annual value and lifetime value
            - **Engagement Level**: Interaction frequency and depth
            
            **Quality Standards**:
            - Each persona must be distinct and non-overlapping
            - Include specific, quantifiable characteristics where possible
            - Ensure personas are actionable for marketing, product, and sales teams
            - Provide realistic, human-centered descriptions that teams can visualize
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
            ## MARKETING STRATEGY DEVELOPMENT REQUEST
            
            **Objective**: Create comprehensive, executable marketing strategies for each customer persona
            
            **Target Personas**:
            {json.dumps(state['personas'], indent=2)}
            
            **Strategy Development Framework**:
            
            For each persona, develop a complete marketing approach covering:
            
            ### 1. Brand Positioning & Messaging
            - **Core Value Proposition**: Primary benefit statement for this persona
            - **Messaging Hierarchy**: Primary, secondary, and supporting messages
            - **Tone & Voice**: Communication style that resonates with this segment
            - **Unique Selling Points**: Specific advantages that appeal to this persona
            
            ### 2. Channel Strategy
            - **Primary Channels**: Top 2-3 most effective marketing channels
            - **Secondary Channels**: Supporting touchpoints for reinforcement
            - **Channel Mix Rationale**: Why these channels work for this persona
            - **Budget Allocation**: Recommended spend distribution across channels
            
            ### 3. Content Strategy
            - **Content Types**: Formats that engage this persona (video, articles, infographics, etc.)
            - **Content Themes**: Topic areas that capture attention and drive engagement
            - **Content Calendar**: Optimal timing and frequency for content delivery
            - **Personalization Level**: Degree of customization needed for effectiveness
            
            ### 4. Campaign Concepts
            - **Acquisition Campaigns**: Strategies to attract new customers from this segment
            - **Retention Campaigns**: Approaches to maintain engagement and loyalty
            - **Upsell/Cross-sell**: Opportunities to increase customer value
            - **Seasonal/Event-based**: Time-sensitive campaign opportunities
            
            ### 5. Performance Measurement
            - **Primary KPIs**: Key metrics to track campaign success
            - **Secondary Metrics**: Supporting indicators of performance
            - **Success Benchmarks**: Realistic targets for each metric
            - **Optimization Triggers**: Signals to adjust or pivot strategy
            
            ### 6. Implementation Roadmap
            - **Phase 1 (0-3 months)**: Immediate launch activities
            - **Phase 2 (3-6 months)**: Optimization and scaling activities
            - **Phase 3 (6-12 months)**: Advanced tactics and expansion
            - **Resource Requirements**: Team, budget, and technology needs
            
            **Deliverable Standards**:
            - Strategies must be specific and actionable
            - Include realistic budget estimates and timelines
            - Provide clear success criteria and measurement plans
            - Ensure strategies are differentiated between personas
            - Include risk factors and mitigation approaches
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
            ## PERSONA AND STRATEGY VALIDATION REQUEST
            
            **Objective**: Conduct comprehensive quality assurance on personas and marketing strategies
            
            **Materials for Review**:
            
            ### Customer Personas:
            {json.dumps(state['personas'], indent=2)}
            
            ### Marketing Strategies:
            {json.dumps(state['marketing_strategies'], indent=2)}
            
            **Validation Framework**:
            
            ### 1. Internal Consistency Analysis
            - **Persona Coherence**: Do demographic, psychographic, and behavioral elements align logically?
            - **Data Alignment**: Are persona characteristics supported by the underlying data?
            - **Realistic Representation**: Do personas represent believable, real-world customer archetypes?
            - **Completeness Check**: Are all required persona elements present and well-developed?
            
            ### 2. Differentiation Assessment
            - **Segment Distinctiveness**: Are personas sufficiently different from each other?
            - **Overlap Analysis**: Identify any problematic similarities between personas
            - **Market Coverage**: Do personas collectively represent the full customer spectrum?
            - **Actionability Gap**: Are differences meaningful for marketing execution?
            
            ### 3. Strategy-Persona Alignment
            - **Targeting Accuracy**: Do marketing strategies appropriately match persona characteristics?
            - **Channel-Persona Fit**: Are recommended channels aligned with persona preferences?
            - **Message-Motivation Match**: Do messaging strategies address persona pain points and motivations?
            - **Tactical Relevance**: Are specific tactics appropriate for each persona's behavior patterns?
            
            ### 4. Business Viability Review
            - **Implementation Feasibility**: Can recommended strategies be realistically executed?
            - **Resource Requirements**: Are budget and resource estimates reasonable?
            - **ROI Potential**: Do strategies have clear paths to measurable business impact?
            - **Risk Assessment**: Are potential challenges and mitigation strategies identified?
            
            ### 5. Data Foundation Strength
            - **Statistical Support**: Are claims backed by sufficient data evidence?
            - **Assumption Identification**: Which elements rely on reasonable assumptions vs. hard data?
            - **Confidence Levels**: How reliable are different aspects of the personas?
            - **Data Gaps**: What additional information would strengthen the personas?
            
            **Validation Deliverables**:
            
            ### Overall Quality Score: 
            Rate 1-10 with detailed justification
            
            ### Critical Issues (if any):
            - Issue description
            - Business impact
            - Recommended resolution
            
            ### Enhancement Opportunities:
            - Specific improvement suggestions
            - Priority level (High/Medium/Low)
            - Implementation approach
            
            ### Data Reliability Assessment:
            - High-confidence elements
            - Medium-confidence elements  
            - Low-confidence elements requiring validation
            
            ### Implementation Readiness:
            - Ready-to-execute strategies
            - Strategies requiring additional development
            - Prerequisites for successful implementation
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
