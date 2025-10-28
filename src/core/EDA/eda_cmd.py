from core.EDA.eda_agent import EDAAgent


class EDACommandLine:
    """Interactive command-line interface"""
    
    def __init__(self, agent: EDAAgent):
        self.agent = agent
        self.commands = {
            'help': self._help,
            'overview': self._overview,
            'quality': self._quality,
            'columns': self._list_columns,
            'analyze': self._analyze_column,
            'report': self._generate_report,
            'insights': self._get_insights,
            'quit': self._quit,
            'exit': self._quit,
        }
    
    def run(self):
        """Run the interactive CLI"""
        self._print_welcome()
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check if it's a command
                cmd_parts = user_input.lower().split()
                if cmd_parts[0] in self.commands:
                    result = self.commands[cmd_parts[0]](cmd_parts[1:] if len(cmd_parts) > 1 else [])
                    if result == "QUIT":
                        break
                    if result:
                        print(f"\n🤖 Assistant:\n{result}")
                else:
                    # Natural language query to agent
                    response = self.agent.chat(user_input)
                    print(f"\n🤖 Assistant:\n{response}")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _print_welcome(self):
        """Print welcome message"""
        print("\n" + "="*70)
        print("🎯 INTELLIGENT EDA AGENT - Chat with Your Data")
        print("="*70)
        print(f"\n📊 Dataset loaded: {self.agent.df.shape[0]:,} rows × {self.agent.df.shape[1]} columns")
        print("\n💡 Quick Commands:")
        print("  • help       - Show all available commands")
        print("  • overview   - Dataset summary")
        print("  • quality    - Data quality report")
        print("  • columns    - List all columns")
        print("  • report     - Generate full EDA report")
        print("  • insights   - Get automated insights")
        print("  • quit/exit  - Exit the program")
        print("\n🗣️  Or just ask questions naturally:")
        print('  • "What columns have missing values?"')
        print('  • "Analyze the age column"')
        print('  • "Show me correlations above 0.8"')
        print('  • "What should I do about outliers?"')
        print("\n" + "="*70)
    
    def _help(self, args):
        return """Available Commands:
        
🔍 QUICK COMMANDS:
  • overview          - Get dataset overview
  • quality           - Check data quality
  • columns           - List all columns
  • analyze <column>  - Analyze specific column
  • insights          - Get automated insights
  • report [markdown|json] - Generate full report
  • quit / exit       - Exit program

💬 NATURAL LANGUAGE:
Just ask questions! Examples:
  • "Which features are most correlated?"
  • "What's wrong with my data?"
  • "Analyze the age and fare columns"
  • "Give me a complete EDA report"
  • "What should I do about missing values?"

The agent will use appropriate tools automatically!"""
    
    def _overview(self, args):
        return self.agent.chat("Give me a dataset overview")
    
    def _quality(self, args):
        return self.agent.chat("Check data quality and tell me about any issues")
    
    def _list_columns(self, args):
        cols = self.agent.df.columns.tolist()
        return f"Dataset Columns ({len(cols)}):\n" + "\n".join(f"  {i+1}. {col}" for i, col in enumerate(cols))
    
    def _analyze_column(self, args):
        if not args:
            return "❌ Usage: analyze <column_name>"
        col_name = " ".join(args)
        return self.agent.chat(f"Analyze the {col_name} column in detail")
    
    def _generate_report(self, args):
        format_type = args[0] if args else "markdown"
        print("\n⏳ Generating comprehensive EDA report...\n")
        return self.agent.chat(f"Generate a full EDA report in {format_type} format")
    
    def _get_insights(self, args):
        return self.agent.chat("Give me all automated insights and recommendations")
    
    def _quit(self, args):
        print("\n👋 Thank you for using EDA Agent! Goodbye!")
        return "QUIT"
