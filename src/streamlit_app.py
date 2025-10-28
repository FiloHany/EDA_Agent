import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.EDA.eda_agent import EDAAgent
import time
import os

# Set Streamlit page configuration
st.set_page_config(
    page_title="🔬 Intelligent EDA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1d24;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        height: 2.8em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        font-size: 0.95rem;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Chat container - fixed at bottom */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(to top, #0e1117 85%, transparent);
        padding: 1.5rem 2rem 1.5rem 2rem;
        z-index: 999;
        border-top: 1px solid #2d3139;
    }
    
    /* Chat messages container */
    .chat-messages-container {
        margin-bottom: 120px;
        padding-bottom: 2rem;
    }
    
    /* Individual message styling */
    .chat-message {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
        animation: slideIn 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 3px solid #667eea;
    }
    
    .chat-message.assistant {
        background: linear-gradient(135deg, #06b6d415 0%, #48bb7815 100%);
        border-left: 3px solid #06b6d4;
    }
    
    .chat-message .avatar {
        font-size: 1.8rem;
        min-width: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .chat-message .message {
        flex: 1;
        color: #e5e7eb;
        line-height: 1.6;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #374151;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #667eea;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Input field styling */
    .stTextInput input {
        border-radius: 25px;
        border: 2px solid #374151;
        padding: 0.8rem 1.5rem;
        background-color: #1f2937;
        color: white;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #374151;
        border-radius: 12px;
        padding: 2rem;
        background-color: #1a1d24;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background-color: #1f2937;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #1f2937;
        border-radius: 8px;
        border: 1px solid #374151;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1d24;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #374151;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4b5563;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f3f4f6;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #374151;
    }
    
    /* Visualization container */
    .viz-container {
        background: #1f2937;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #374151;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
        padding: 1rem 1.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Divider styling */
    hr {
        border: none;
        border-top: 1px solid #374151;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []
if 'last_file' not in st.session_state:
    st.session_state.last_file = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def display_chat_message(role, content, index):
    """Display a chat message with enhanced styling"""
    avatar = "👤" if role == "user" else "🤖"
    message_class = "user" if role == "user" else "assistant"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="avatar">{avatar}</div>
        <div class="message">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def create_visualization(df, viz_type, settings):
    """Create a visualization based on type and settings"""
    try:
        if viz_type == "histogram":
            fig = px.histogram(df, x=settings['column'], 
                              title=f'Distribution of {settings["column"]}',
                              nbins=30,
                              color_discrete_sequence=['#667eea'])
        elif viz_type == "scatter":
            fig = px.scatter(df, x=settings['x'], y=settings['y'],
                            title=f'{settings["x"]} vs {settings["y"]}',
                            color_discrete_sequence=['#06b6d4'])
        elif viz_type == "box":
            fig = px.box(df, y=settings['column'],
                        title=f'Box Plot of {settings["column"]}',
                        color_discrete_sequence=['#764ba2'])
        elif viz_type == "correlation":
            numeric_cols = df.select_dtypes(include=['number']).columns
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix,
                           title='Correlation Heatmap',
                           color_continuous_scale='RdBu_r',
                           aspect='auto')
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(31, 41, 55, 1)",
            font=dict(color='#e5e7eb'),
            title_font_size=18,
            title_font_color='#f3f4f6',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def process_user_message(user_input):
    """Process user message and get agent response"""
    if not user_input.strip():
        return
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.processing = True
    
    # Get agent response
    try:
        with st.spinner("🤔 Analyzing your data..."):
            response = st.session_state.agent.chat(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.session_state.processing = False

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: #667eea;'>🔬 EDA Studio</h1>", unsafe_allow_html=True)
        st.markdown("---")

        # File Upload Section
        st.markdown("<div class='section-header'>📁 Data Upload</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type="csv",
            help="Upload your CSV file to begin analysis",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            if st.session_state.df is None or uploaded_file.name != getattr(st.session_state.last_file, 'name', None):
                try:
                    with st.spinner("Loading dataset..."):
                        st.session_state.df = pd.read_csv(uploaded_file)
                        st.session_state.agent = EDAAgent(st.session_state.df)
                        st.session_state.last_file = uploaded_file
                        st.session_state.messages = []
                        st.session_state.visualizations = []
                    st.success(f"✅ Loaded {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    st.session_state.df = None
                    return
                
        if st.session_state.df is not None:
            st.markdown("---")
            
            # Quick Actions
            st.markdown("<div class='section-header'>⚡ Quick Actions</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Overview"):
                    process_user_message("Give me a comprehensive dataset overview")
                    st.rerun()
                
                if st.button("🔍 Quality Check"):
                    process_user_message("Perform a detailed data quality check")
                    st.rerun()
            
            with col2:
                if st.button("💡 Insights"):
                    process_user_message("Give me key insights and patterns in the data")
                    st.rerun()
                
                if st.button("📈 Statistics"):
                    process_user_message("Show me statistical summary")
                    st.rerun()
            
            st.markdown("---")
            
            # Visualization Tools
            st.markdown("<div class='section-header'>📊 Visualizations</div>", unsafe_allow_html=True)
            viz_type = st.selectbox(
                "Visualization Type",
                ["histogram", "scatter", "box", "correlation"],
                help="Select the type of visualization to create"
            )
            
            if viz_type in ["histogram", "box"]:
                col = st.selectbox("Select Column", st.session_state.df.columns)
                if st.button("🎨 Generate", key="gen_viz_1"):
                    fig = create_visualization(st.session_state.df, viz_type, {"column": col})
                    if fig:
                        st.session_state.visualizations.append(fig)
                        st.rerun()
            
            elif viz_type == "scatter":
                col1 = st.selectbox("X-axis", st.session_state.df.columns, key="x_axis")
                col2 = st.selectbox("Y-axis", st.session_state.df.columns, key="y_axis")
                if st.button("🎨 Generate", key="gen_viz_2"):
                    fig = create_visualization(st.session_state.df, viz_type, {"x": col1, "y": col2})
                    if fig:
                        st.session_state.visualizations.append(fig)
                        st.rerun()
            
            elif viz_type == "correlation":
                if st.button("🎨 Generate", key="gen_viz_3"):
                    fig = create_visualization(st.session_state.df, viz_type, {})
                    if fig:
                        st.session_state.visualizations.append(fig)
                        st.rerun()
            
            st.markdown("---")
            
            # Export Options
            st.markdown("<div class='section-header'>📥 Export</div>", unsafe_allow_html=True)
            if st.button("📄 Generate Report"):
                with st.spinner("Generating comprehensive report..."):
                    try:
                        report = st.session_state.agent.generate_automatic_eda()
                        st.download_button(
                            "⬇️ Download Report",
                            report,
                            file_name=f"eda_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
            
            # Clear chat button
            st.markdown("---")
            if st.button("🗑️ Clear Chat", help="Clear all chat messages"):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("🔄 Clear Visualizations", help="Remove all visualizations"):
                st.session_state.visualizations = []
                st.rerun()

    # Main content area
    if st.session_state.df is None:
        # Welcome screen
        st.markdown("<h1 style='text-align: center; color: #667eea; margin-top: 2rem;'>🤖 Intelligent EDA Studio</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 1.2rem;'>Your AI-powered data analysis assistant</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1f2937 0%, #111827 100%); 
                        padding: 2rem; border-radius: 12px; margin-top: 3rem;
                        border: 1px solid #374151;'>
                <h3 style='color: #f3f4f6; text-align: center;'>🚀 Get Started</h3>
                <ul style='color: #d1d5db; line-height: 2; font-size: 1.05rem;'>
                    <li>📁 Upload your CSV file using the sidebar</li>
                    <li>⚡ Use quick actions for instant insights</li>
                    <li>💬 Ask questions in natural language</li>
                    <li>📊 Generate interactive visualizations</li>
                    <li>📄 Export comprehensive reports</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Dataset metrics at the top
        st.markdown("<div style='margin-bottom: 1rem;'>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Rows", f"{st.session_state.df.shape[0]:,}")
        with col2:
            st.metric("📋 Columns", st.session_state.df.shape[1])
        with col3:
            missing = st.session_state.df.isna().sum().sum()
            st.metric("⚠️ Missing Values", f"{missing:,}")
        with col4:
            memory = st.session_state.df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("💾 Memory", f"{memory:.1f} MB")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizations section
        if st.session_state.visualizations:
            st.markdown("<div class='section-header'>📊 Generated Visualizations</div>", unsafe_allow_html=True)
            for i, fig in enumerate(st.session_state.visualizations):
                st.markdown("<div class='viz-container'>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button(f"💾 Save HTML", key=f"save_html_{i}"):
                        if not os.path.exists("reports/visualizations"):
                            os.makedirs("reports/visualizations")
                        filename = f"viz_{i+1}_{time.strftime('%Y%m%d_%H%M%S')}.html"
                        fig.write_html(f"reports/visualizations/{filename}")
                        st.success(f"Saved as {filename}")
                
                with col2:
                    if st.button(f"🗑️ Remove", key=f"remove_viz_{i}"):
                        st.session_state.visualizations.pop(i)
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
        
        # Chat messages container
        st.markdown("<div class='chat-messages-container'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>💬 Chat with Your Data</div>", unsafe_allow_html=True)
        
        # Display all messages
        for idx, message in enumerate(st.session_state.messages):
            display_chat_message(message["role"], message["content"], idx)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Fixed chat input at bottom
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)  # Spacer
        
        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                user_input = st.text_input(
                    "Chat Input",
                    placeholder="Ask me anything about your data...",
                    label_visibility="collapsed",
                    key="chat_input"
                )
            with col2:
                submit_button = st.form_submit_button("Send 📤", use_container_width=True)
            
            if submit_button and user_input:
                process_user_message(user_input)
                st.rerun()

if __name__ == "__main__":
    main()