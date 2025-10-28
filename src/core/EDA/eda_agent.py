import os
from typing import Optional

import pandas as pd
from core.EDA.eda_service import EDAService
from core.EDA.eda_tools import EDATools

from core.report.report_generator import ReportBuilder


class EDAAgent:
    """Intelligent chat agent for EDA interaction"""
    
    def __init__(self, df: pd.DataFrame, use_openai: bool = False):
        self.df = df
        self.eda_service = EDAService(df)
        self.tools = EDATools.create_tools(self.eda_service)
        self.llm = self._setup_llm(use_openai)
        self.agent = self._create_agent()
    
    def _setup_llm(self, use_openai: bool):
        """Setup LLM (OpenAI or Ollama)"""
        if use_openai or os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            print("🤖 Using OpenAI GPT-4o-mini\n")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        else:
            from langchain_ollama import ChatOllama
            # Qwen3:4b is THE BEST free model for data analysis and tool calling
            model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
            print(f"🤖 Using Ollama: {model}")
            print("💡 Tip: For best results, ensure you have: ollama pull qwen3:4b\n")
            return ChatOllama(model=model, temperature=0, num_predict=1024)
    
    def _create_agent(self):
        """Create the LangGraph agent"""
        from langgraph.prebuilt import create_react_agent
        
        system_prompt = """You are an expert data analysis assistant specializing in Exploratory Data Analysis (EDA).

YOUR ROLE:
- Help users understand their dataset through comprehensive analysis
- Use available tools to extract insights from the data
- Provide actionable recommendations based on findings
- Explain statistical concepts in simple terms
- Guide users through data quality issues and suggest fixes

AVAILABLE TOOLS:
1. get_dataset_overview - First tool to use for new datasets
2. get_data_quality - Check for missing values and duplicates
3. analyze_column - Deep dive into specific columns
4. get_correlations - Find relationships between features
5. get_automated_insights - Get AI-generated insights
6. generate_full_report - Create comprehensive EDA report
7. compare_columns - Compare multiple columns side-by-side

INSTRUCTIONS:
- Always start with get_dataset_overview for new questions about the dataset
- Use tools systematically: overview → quality → specific analysis
- Provide clear, actionable insights after each analysis
- When you find issues (missing data, outliers, imbalance), suggest solutions
- Format responses clearly with sections and bullet points
- If asked for a full EDA, use generate_full_report
- Don't make up numbers - only use what tools return

RESPONSE STYLE:
- Be concise but informative
- Use emojis sparingly for readability (📊 🔍 💡 ⚠️ ✅)
- Explain statistical terms when needed
- Provide "What this means" sections for complex findings
- Always end with "What should you do next?" suggestions"""

        return create_react_agent(
            self.llm,
            self.tools,
            state_modifier=system_prompt
        )
    
    def chat(self, query: str, verbose: bool = False) -> str:
        """Send a query to the agent"""
        try:
            response = self.agent.invoke(
                {"messages": [("user", query)]},
                {"recursion_limit": 15}
            )
            
            if verbose:
                self._print_tool_activity(response)
            
            # Extract final answer
            messages = response.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    if msg.content and not msg.content.startswith('['):
                        return msg.content
            
            return "No response generated."
        
        except Exception as e:
            return f"❌ Error: {str(e)}\n\nTry rephrasing your question or use 'help' for guidance."
    
    def _print_tool_activity(self, response):
        """Print tool calls for debugging"""
        print("\n" + "="*60)
        print("🔧 TOOL ACTIVITY")
        print("="*60)
        
        messages = response.get("messages", [])
        for msg in messages:
            if hasattr(msg, 'type'):
                if msg.type == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"📞 Tool: {tc['name']}")
                        print(f"   Args: {tc['args']}")
                elif msg.type == 'tool':
                    preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                    print(f"📥 Result: {preview}\n")
        
        print("="*60 + "\n")
    
    def generate_automatic_eda(self, save_path: Optional[str] = None) -> str:
        """Generate complete EDA report automatically (no LLM needed)"""
        print("🔄 Generating comprehensive EDA report...\n")
        
        builder = ReportBuilder(self.eda_service)
        report = (builder
                 .add_metadata_section()
                 .add_quality_section()
                 .add_column_analysis_section()
                 .add_insights_section()
                 .build_markdown())
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Report saved to: {save_path}\n")
        
        return report