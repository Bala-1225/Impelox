import os
import shutil
import re
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Union, List
import operator
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Chroma vectorstore directory
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")

# Load the OpenAI LLM using LangChain
llm_model = ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_api_key, temperature=0.00)

# Initialize Neo4j connection
os.environ['NEO4J_URI'] = os.getenv("NEO4J_URI")
os.environ['NEO4J_USERNAME'] = os.getenv("NEO4J_USERNAME")
os.environ['NEO4J_PASSWORD'] = os.getenv("NEO4J_PASSWORD")
graph = Neo4jGraph()

# Initialize OpenAI embeddings for Chroma
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

#----------------------------------------------------------Define the state structure-----------------------------------------------------------
class State(TypedDict):
    input: str  # Resume text
    chat_history: List[AIMessage]
    agent_outcome: Union[str, None]  # This could hold structured results
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]  # Intermediate steps taken by the agent

# ----------------------------------------------------------------------Extract----------------------------------------------------------------------------

# Function to extract text from a PDF
def extract_pdf_text(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"  # Changed from "/n" to "\n" for proper line breaks
        return text.strip()
    except Exception as e:
        return ""

# ------------------------------------------------------------------load-------------------------------------------------------------------------

# Function to load job descriptions from local PDF files and store them in Chroma VectorDB
def load_job_descriptions_from_files(embeddings_model, directory, file_paths):
    job_descriptions = []

    for file_path in file_paths:
        # Extract text from each specified local PDF file
        jd_text = extract_pdf_text(file_path)
        if jd_text:
            # Create a document from the extracted text
            document = Document(page_content=jd_text)
            job_descriptions.append(document)

    if job_descriptions:
        # Create a vector store for the job descriptions
        store_name = "job_descriptions"
        vectorstore = create_vectorstore(job_descriptions, embeddings_model, store_name)
        return vectorstore  # Return the created vectorstore
    else:
        return None  # Return None if no job descriptions were extracted

#----------------------------------------------------------------chroma-----------------------------------------------------------------------

# Function to create a Chroma vectorstore
def create_vectorstore(documents, embeddings_model, store_name):
    clear_vectorstore_directory(store_name)
    if not os.path.exists(VECTORSTORE_DIR):
        os.mkdir(VECTORSTORE_DIR)
    vectorstore = Chroma.from_documents(
        documents, embeddings_model, persist_directory=os.path.join(VECTORSTORE_DIR, store_name)
    )
    vectorstore.persist()
    return vectorstore

def clear_vectorstore_directory(store_name):
    store_path = os.path.join(VECTORSTORE_DIR, store_name)
    if os.path.exists(store_path):
        try:
            for filename in os.listdir(store_path):
                file_path = os.path.join(store_path, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Delete file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete directory
            os.rmdir(store_path)  # Finally, remove the directory itself
        except Exception as e:
            print(f"Error clearing vectorstore: {e}")

#-------------------------------------------------------------------------------eval------------------------------------------------------

# Function to evaluate the resume against the job description
def evaluate_resume(jd_context, res_context):
    prompt = (
        "You are an expert in rigorously evaluating job applications, with an uncompromising focus on technical skills and relevant experience, which is the most important factor."
        "You have been provided with the job description and resume below."
        "Your role is to strictly assess whether the resume aligns with the job description."
        "Rate the match between the resume and job description on a strict scale from 0 to 100, using the following breakdown:"

        "Experience – 40 marks"
        "Technical Skills – 35 marks"
        "Projects – 15 marks"
        "Certifications – 5 marks"
        "Soft Skills – 5 marks"

        "For Experience, give the highest weight to job experience and relevance to the role. If the candidate does not meet the experience requirement of at least 3 years, they should be disqualified regardless of their technical abilities. If the experience falls under 3-6 years give the score as 30/40.  If the experience falls under 7-9 years give the score as 35/40.  If the experience is more than 9 years give the score as 40/40 "

        "For Technical Skills, closely examine the technologies mentioned in the job description, such as Angular, React, Python, AWS, etc. Examine the relevant skills too. Missing critical technical qualifications or insufficient proficiency should result in a very low score or rejection."

        "For Projects, consider the relevance and scope of their contributions to previous projects and assess how well those align with the job description, particularly in SaaS or full-stack development."

        "For Certifications, check for relevant industry certifications that validate the candidate’s expertise, especially in technologies or areas like AWS, cloud, security, or full-stack development."

        "For Soft Skills, focus on cultural alignment, teamwork, communication skills, and leadership potential as outlined in the job description."

        "Be meticulous in identifying reasons for rejection. Any resume lacking alignment in job experience, technical skills, or other essential requirements should be scored low or rejected outright."
        "Reference the exact criteria from the job description before assigning a score."

        "Justify your decision in detail. Here is the job description and resume:\n\n"
        f"{jd_context}\n\n{res_context}\n\n"

        "Justify your decision in detail for each category, and calculate the total score out of 100."
    )

    response = llm_model(prompt)

    # Extract the score from the response
    score_match = re.search(r"(\d{1,3})/100", response.content)
    score = int(score_match.group(1)) if score_match else 0

    # Proceed with question generation if score > 40
    questions = []
    if score > 40:
        # Refined prompt for generating technical MCQs
        question_prompt = (
            "Based on the following job description and resume, please generate 10 multiple-choice technical interview questions. "
            "The questions should range from intermediate to expert level and should focus on the candidate's skills and experience. "
            "Here is the job description:\n\n"
            f"{jd_context}\n\nAnd here is the candidate's resume:\n\n{res_context}\n\n"
            "Ensure that each question has one correct answer and three distractors. "
            "Please present the questions in the following format:\n"
            "1. Question? \n   a) Option A \n   b) Option B \n   c) Option C \n   d) Option D \n "
            "Generate the questions now."
        )
        question_response = llm_model(question_prompt)
        questions = question_response.content.strip().split("\n")

    return response.content, questions if questions else None

# ---------------------------------------------------------------- Define the Workflow ----------------------------------------------------------

workflow = StateGraph(State)
print("// Initialized StateGraph with AgentState.")
# Add node for extracting text
workflow.add_node("extracted_text", extract_pdf_text)
print("// Extracting text.")

# Add node for loading the file
workflow.add_node("load_file", load_job_descriptions_from_files)
print("// loaded pdf.")

# Add node for evaluating the resume
workflow.add_node("evaluate_resume", evaluate_resume)  # Added evaluation node
print("// Extracting evaluate_resume.")

# Set the entry point for the workflow
workflow.set_entry_point("extracted_text")

# Connect nodes
workflow.add_edge("extracted_text", "load_file")  # Connect to the evaluation node
print("Edge 1")
workflow.add_edge("load_file", "evaluate_resume")  # load_file and evaluate
print("Edge 2")
workflow.add_edge("evaluate_resume", END)  # load_file and evaluate
print("Edge 3")
# ----------------------------------------------------- evaluation result in Neo4j -----------------------------------------------------------
# Function to store the evaluation result in Neo4j
def store_evaluation_in_neo4j(graph, resume_text, jd_text, evaluation_result):
    query = """
    MERGE (r:Resume {content: $resume_content})
    MERGE (jd:JobDescription {content: $jd_content})
    MERGE (e:Evaluation {result: $evaluation_result})
    MERGE (r)-[:MATCHED]->(jd)
    MERGE (r)-[:EVALUATED]->(e)
    """
    graph.run(query, resume_content=resume_text, jd_content=jd_text, evaluation_result=evaluation_result)

# To run the workflow in Streamlit app
if __name__ == "__main__":
    st.title("Resume Evaluation Tool")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    job_files = st.file_uploader("Upload Job Descriptions (PDF)", type=["pdf"], accept_multiple_files=True)

    if st.button("Evaluate"):
        if uploaded_file and job_files:
            resume_text = extract_pdf_text(uploaded_file)
            job_descriptions = load_job_descriptions_from_files(embeddings_model, VECTORSTORE_DIR, job_files)

            if resume_text and job_descriptions:
                for jd in job_descriptions:
                    evaluation_result, questions = evaluate_resume(jd.page_content, resume_text)
                    store_evaluation_in_neo4j(graph, resume_text, jd.page_content, evaluation_result)
                    st.write("Evaluation Result:", evaluation_result)
                    if questions:
                        st.write("Generated Questions:")
                        for question in questions:
                            st.write(question)
            else:
                st.error("Could not extract text from files.")
        else:
            st.warning("Please upload both your resume and job descriptions.")
