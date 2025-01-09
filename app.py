import os
import streamlit as st
import mysql.connector
import pandas as pd
from dotenv import load_dotenv

# 1. Ortam değişkenlerini yükle
load_dotenv()

# 2. Gerekli değişkenleri çek
api_key = os.getenv("OPENAI_API_KEY")
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
db_name = os.getenv("DB_NAME")

# -- OpenAI API Key kontrolü --
if not api_key:
    st.error("OpenAI API key missing! Please check your .env file.")
    st.stop()

# -- OpenAI LLM ayarı (ChatOpenAI) --
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage

llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-4",  # veya "gpt-3.5-turbo"
    temperature=0
)

# -- Streamlit sayfa ayarları --
st.set_page_config(page_title="Assistflow.ai Chat with MySQL", page_icon=":speech_balloon:")
st.title("Assistflow.ai Chat with MySQL")

# -- Veritabanına bağlanma --
try:
    connection = mysql.connector.Connect(
        host=db_host,
        user=db_user,
        password=db_pass,
        database=db_name
    )
    st.session_state.db = connection
    st.success("Connected to the database!")
except Exception as e:
    st.error(f"Connection failed: {str(e)}")
    st.stop()

# -- Helper: get database schema info --
def get_schema_info():
    """
    Retrieves table and column information from the connected MySQL database.
    """
    try:
        cursor = st.session_state.db.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        schema_info = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"DESCRIBE {table_name}")
            columns = [col[0] for col in cursor.fetchall()]
            schema_info.append(f"{table_name} ({', '.join(columns)})")
        cursor.close()
        return "Tables: " + ", ".join(schema_info)
    except Exception as e:
        return str(e)

# -- Helper: generate SQL from a natural language query --
def generate_sql(schema_info, query):
    """
    Uses an LLM (ChatOpenAI) to convert a natural language question into an SQL query.
    """

    # Our prompt template text
    template_text = """
    You are an expert in MariaDB SQL. Convert the following natural language
    question into a valid SQL query. The database schema is:
    {schema_info}

    Question: {user_query}

    Return ONLY the SQL query, with no extra text or explanation.
    """

    # Build a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(template_text)

    # Format messages for the chat model
    messages = prompt.format_messages(
        schema_info=schema_info,
        user_query=query
    )

    # Pass the messages to the LLM
    response = llm(messages)

    # Clean up any backticks or extra text
    sql_query = response.content.strip().strip("`")
    return sql_query

# -- Helper: execute SQL query in MySQL database --
def execute_query(query):
    """
    Executes the given SQL query against the MySQL database.
    Returns (results, columns) or (error_message, []) if there's an error.
    """
    try:
        cursor = st.session_state.db.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        cursor.close()
        return results, columns
    except Exception as e:
        return str(e), []

# -- Initialize chat history in session state --
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm an Assistflow.ai SQL assistant. Ask me anything about your database.")
    ]

# -- Display existing chat messages --
for msg in st.session_state.chat_history:
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    else:
        with st.chat_message("user"):
            st.markdown(msg.content)

# -- Streamlit chat input (user types question here) --
user_query = st.chat_input("Ask your question here (e.g. 'How many rows in the users table?') ...")

if user_query:
    # 1. Display user's query in chat
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Get schema info
    schema_info = get_schema_info()

    # 3. Generate SQL from user query
    sql_query = generate_sql(schema_info, user_query)

    # 4. Execute the SQL
    result, columns = execute_query(sql_query)

    # 5. Build the AI response
    if isinstance(result, str):
        # If there's an error (result is a string describing the error)
        error_message = f"**Generated SQL**: `{sql_query}`\n\n**Error**: {result}"
        st.session_state.chat_history.append(AIMessage(content=error_message))
        with st.chat_message("assistant"):
            st.error(error_message)
    else:
        # If the query executed successfully
        if len(result) > 0:
            # Convert to pandas DataFrame
            df = pd.DataFrame(result, columns=columns)
            success_message = f"**Generated SQL**: `{sql_query}`\n\n**Results**:"
            st.session_state.chat_history.append(AIMessage(content=success_message))
            with st.chat_message("assistant"):
                st.markdown(success_message)
                st.dataframe(df)
        else:
            # Query ran but returned no rows
            success_message = f"**Generated SQL**: `{sql_query}`\n\nNo results found."
            st.session_state.chat_history.append(AIMessage(content=success_message))
            with st.chat_message("assistant"):
                st.markdown(success_message)
