import pandas as pd

from core.EDA.eda_agent import EDAAgent
from core.EDA.eda_cmd import EDACommandLine

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Intelligent EDA Agent with Chat Interface')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--openai', action='store_true', help='Use OpenAI instead of Ollama')
    parser.add_argument('--auto-report', type=str, help='Generate automatic report and save to file')
    parser.add_argument('--query', type=str, help='Run single query and exit')
    parser.add_argument('--verbose', action='store_true', help='Show tool activity')
    
    args = parser.parse_args()
    
    # Load dataset
    try:
        print(f"\n📂 Loading dataset: {args.csv_file}")
        df = pd.read_csv(args.csv_file)
        print(f"✅ Loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Initialize agent
    agent = EDAAgent(df, use_openai=args.openai)
    
    # Handle different modes
    if args.auto_report:
        # Generate automatic report without LLM
        report = agent.generate_automatic_eda(save_path=args.auto_report)
        print(report[:500] + "...\n")
        print(f"✅ Full report saved to: {args.auto_report}")
    
    elif args.query:
        # Single query mode
        print(f"💬 Query: {args.query}\n")
        response = agent.chat(args.query, verbose=args.verbose)
        print(f"🤖 Response:\n{response}")
    
    else:
        # Interactive chat mode
        cli = EDACommandLine(agent)
        cli.run()


if __name__ == "__main__":
    main()