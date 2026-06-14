import os
import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# Use the updated modern OpenAI connection
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables (such as OPENAI_API_KEY from your .env file)
load_dotenv()

st.title("AI Data Analysis Assistant 📊")

# 1. Added 'xlsx' to the allowed types list
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt', 'xlsx'])

# 2. Indented your logic correctly to guarantee 'df' exists before trying to display or use it
if uploaded_file is not None:
    # Handle CSV or generic text files
    if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file) # Removed quotation marks around the variable
        
    # Handle Excel workbooks seamlessly
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file) # Changed read_csv to read_excel

    # Render a visual validation preview to the user
    st.write("### First 5 rows of the uploaded file")
    st.write(df.head())

    st.write("### Ask a question about your data")
    user_question = st.text_input("Enter your question")

    if user_question:
        def create_agent(dataframe):
            # Using ChatOpenAI (gpt-4o-mini is highly cost-effective and accurate for code interpretation)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            # Fixed 'allow_dangerous_code=True' parameter required by modern LangChain
            agent = create_pandas_dataframe_agent(
                llm, 
                dataframe, 
                verbose=True, 
                allow_dangerous_code=True
            )
            return agent
        
        agent = create_agent(df)
        
        with st.spinner("Analyzing data and generating code...."):
            try:
                # Modern LangChain standard uses agent.invoke() instead of agent.run()
                response = agent.invoke({"input": user_question})
                answer = response.get("output", "No answer could be processed.")
                
                st.write("### Answer")
                st.success(answer)
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
else:
    st.info("Please upload a file to start your analysis workflow.")