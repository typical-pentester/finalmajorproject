from flask import Flask, render_template, request, redirect, url_for, session, flash
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import faiss
import numpy as np
import pickle
import shutil
import time
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import requests
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging
from langchain_groq import ChatGroq
import os
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = 'your_secret_key' 
UPLOAD_FOLDER = "C:\\Users\\nikle\\MAJOR PROJECT\\SOURCE CODE\\Contents"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


groq_api_key = "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"
os.environ["GROQ_API_KEY"] = groq_api_key
chat_groq = ChatGroq(api_key=os.getenv("GROQ_API_KEY"))


embeddings = SentenceTransformer('all-MiniLM-L6-v2')

tool = language_tool_python.LanguageTool("en-US")
nltk.download("words")

nltk.download("punkt_tab")

# Initialize the model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key="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",
    temperature=0.5,
    max_tokens=16000,
)

detailed_prompt = PromptTemplate.from_template(
    "Please answer the following question with as much detail, context, and relevant information as possible:\n\n{question}\n\n"
    "Ensure the response includes explanations, examples, and references when applicable."
)

vectorstore = None
retriever = None

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users 
                      (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()


@app.route('/')
def home():
    return redirect(url_for('index'))

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[0], password): 
            session['username'] = username 
            return redirect(url_for('dashboard'))
                
        else:
            return render_template('login.html', error='Invalid username or password.')

    return render_template('login.html')  

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password) 
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                           (username, hashed_password))
            conn.commit()
            return redirect(url_for('login'))  
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already exists.')
        finally:
            conn.close()

    return render_template('register.html') 


@app.route('/logout')
def logout():
    session.pop('username', None) 
    return redirect(url_for('login'))  




@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('dashboard1.html')

@app.route('/dashboard001', methods=['GET', 'POST'])
def dashboard001():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        vectorstore_name = request.form.get('vectorstore_name')
        pdf_file = request.files.get('pdf_file')

        if not vectorstore_name or not pdf_file:
            print('Vectorstore name and PDF file are required!')
            return redirect(request.url)

        save_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
        pdf_file.save(save_path)
        flash(f'File {pdf_file.filename} uploaded successfully!', 'success')

        # Process PDF to create vector store
        try:
            print("Loading PDF document...")
            loader = PyPDFLoader(save_path)
            data = loader.load()

            print("Splitting document into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=2000, chunk_overlap=200)
            docs = text_splitter.split_documents(data)
            time.sleep(1)

            print("Creating embeddings and vector store...")
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore_groq = FAISS.from_documents(docs, embeddings)
            time.sleep(1)

            # Save the vector store with the provided name
            file_path = f"{vectorstore_name}.pkl"
            print(f"Saving vector store to {file_path}...")
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_groq, f)
            print("Clearing uploaded files...")
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER)  # Recreate the folder after deletion

            flash(f'Vectorstore {vectorstore_name} created successfully!', 'success')
            print("PDF processed and data stored successfully!")

        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            print(f'Error processing file: {str(e)}')
    return render_template('upload.html')

@app.route('/select_vectorstore', methods=['GET', 'POST'])
def select_vectorstore():
    global chain, retriever
    if 'username' not in session:
        return redirect(url_for('login'))

    vectorstores = [f for f in os.listdir() if f.endswith('.pkl')]

    if request.method == 'POST':
        selected_store = request.form.get('selected_store')
        session['selected_vectorstore'] = selected_store
        print(f'Selected vectorstore: {selected_store}')
        vectorstore_file = selected_store
        if os.path.exists(vectorstore_file):

            with open(vectorstore_file, "rb") as f:
                vectorstore = pickle.load(f)
            retriever=vectorstore.as_retriever()
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=retriever
            )            
           
            logging.info(f'vectorstore: {selected_store} has been loaded.....')
        else:
            logging.error("Vector store file does not exist.")

        return redirect(url_for('dashboard001'))

    return render_template('select_vectorstore.html', vectorstores=vectorstores)


@app.route('/ask', methods=['GET', 'POST'])
def ask():
    if 'username' not in session:
        return redirect(url_for('login'))
    if 'selected_vectorstore' not in session:
        flash('Please select a vectorstore first!', 'error')
        return redirect(url_for('select_vectorstore'))

    if 'chain' not in globals():
        flash('Vectorstore not initialized. Please select a vectorstore first.', 'error')
        return redirect(url_for('select_vectorstore'))

    if request.method == 'POST':
        question = request.form['question']
        selected_store = session.get('selected_vectorstore')
        if not selected_store:
            flash('No vectorstore selected!', 'error')
            return redirect(url_for('select_vectorstore'))
        if not question:
            flash('Please enter a question', 'error')
            return render_template('ask.html')
        else:

            # result = chain({"question": question}, return_only_outputs=True)
            result = chain.invoke({"question": question})


        logging.info(f"Asking question: {question}")
        logging.info(f"Selected vectorstore: {selected_store}")
        

        return render_template('ask.html', question=question, answer=result['answer'])

    return render_template('ask.html', question="", answer="")
import random

@app.route('/generate_questions', methods=['GET', 'POST'])
def generate_questions():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        topic = request.form['topic']
        selected_store = session.get('selected_vectorstore')

        if not selected_store:
            flash('No vectorstore selected!', 'error')
            return redirect(url_for('select_vectorstore'))

        if not topic:
            flash('Please enter a topic', 'error')
            return render_template('generate_questions.html')
        results = retriever.get_relevant_documents(topic, k=20)
        # Generate 5 questions and answers related to the topic
        random.shuffle(results)
        results = results[:6]
        qa_pairs = []
        for i, doc in enumerate(results):
            context = doc.page_content
            prompt = (
                f"Based on the following context, generate one question and its answer make sure the answer is long:\n"
                f"Context: {context}\n"
                "Question and Answer:"
            )
            response = llm.predict(prompt)

            if "Question:" in response and "Answer:" in response:
                parts = response.split("Answer:")
                question = parts[0].replace("Question:", "").strip()
                answer = parts[1].strip() if len(parts) > 1 else "No answer provided"
                qa_pairs.append({"question": question, "answer": answer})

        print(qa_pairs)
        return render_template('generate_questions.html', topic=topic, qa_pairs=qa_pairs)
        
    return render_template('generate_questions.html')

valid_words = set(words.words())

def preprocess(text):
    """Lowercase and tokenize the text."""
    return word_tokenize(text.lower())

def is_rubbish(text):
    """Check if the answer is gibberish or nonsense."""
    tokens = preprocess(text)

    if len(tokens) < 3:
        return True  

    valid_count = sum(1 for word in tokens if word in valid_words)
    if (valid_count / len(tokens)) < 0.6:  
        return True  

    return False

def calculate_similarity(student_answer, answer_key):
    """Use Sentence-BERT to compute semantic similarity."""
    student_embedding = embeddings.encode(student_answer, convert_to_tensor=True)
    key_embedding = embeddings.encode(answer_key, convert_to_tensor=True)
    similarity = cosine_similarity([student_embedding.cpu().numpy()], [key_embedding.cpu().numpy()])[0][0]
    return round(similarity * 2, 2)  

def keyword_score(student_answer, answer_key):
    """Check if important words are present."""
    student_tokens = set(preprocess(student_answer))
    key_tokens = set(preprocess(answer_key))
    matched_keywords = student_tokens.intersection(key_tokens)
    score = (len(matched_keywords) / len(key_tokens)) * 1 
    return round(score, 2)

def grammar_score(student_answer):
    """Use LanguageTool to count grammar errors."""
    matches = tool.check(student_answer)
    errors = len(matches)
    if errors == 0:
        return 1.0  
    elif errors <= 2:
        return 0.7  
    else:
        return 0.4  

def coherence_score(student_answer):
    """Simple coherence scoring based on length and sentence structure."""
    sentences = student_answer.split(".")
    if len(sentences) >= 2:
        return 1.0  
    elif len(sentences) ==0:
        return 0
    return 0.5  

def grade_answer(student_answer, answer_key):
    """Grade the student answer based on multiple factors."""

    if is_rubbish(student_answer):
        return 0.0, "Your answer is not relevant to the question."

    sim_score = calculate_similarity(student_answer, answer_key)
    if sim_score < 0.2: 
        return 0.0, "Your answer is not relevant to the question."

    key_score = keyword_score(student_answer, answer_key)
    gram_score = grammar_score(student_answer)
    coh_score = coherence_score(student_answer)

    total_score = round(sim_score + key_score + gram_score + coh_score, 2)
    feedback = f"""
    Content Similarity: {sim_score}/2 ,  Keyword Usage: {key_score}/1 ,  Grammar & Structure: {gram_score}/1 ,  Coherence & Clarity: {coh_score}/1 ,  Final Score: {total_score}/5   """
    
    return total_score, feedback.strip()

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST' and 'topic' in request.form:
        topic = request.form['topic']
        selected_store = session.get('selected_vectorstore')

        if not selected_store:
            flash('No vectorstore selected!', 'error')
            return redirect(url_for('select_vectorstore'))

        if not topic:
            flash('Please enter a topic', 'error')
            return render_template('generate_questions.html')

        results = retriever.get_relevant_documents(topic, k=20)
        # random.shuffle(results)
        # results = results[:6]
        results = random.sample(results, min(5, len(results)))
        qa_pairs = []
        for doc in results:
            context = doc.page_content
            prompt = (
                f"Based on the following context, generate one question and its answer make sure the answer is long:\n"
                f"Context: {context}\n"
                "Question and Answer:"
            )
            response = llm.predict(prompt)
            if "Question:" in response and "Answer:" in response:
                parts = response.split("Answer:")
                question = parts[0].replace("Question:", "").strip()
                answer = parts[1].strip() if len(parts) > 1 else "No answer provided"
                qa_pairs.append({"question": question, "answer": answer})

        session['qa_pairs'] = qa_pairs
        return render_template('quiz.html', topic=topic, qa_pairs=qa_pairs)

    if request.method == 'POST' and 'student_answers' in request.form:
        qa_pairs = session.get('qa_pairs', [])
        student_answers = request.form.getlist('student_answers') 

        graded_results = []
        for i, student_answer in enumerate(student_answers):
            correct_answer = qa_pairs[i]['answer']
            score, feedback = grade_answer(student_answer, correct_answer)
            graded_results.append({"question": qa_pairs[i]['question'], "correct_answer": correct_answer, "student_answer": student_answer, "score": score, "feedback": feedback})

        return render_template('quiz.html', qa_pairs=qa_pairs, graded_results=graded_results)

    return render_template('quiz.html')

if __name__ == '__main__':
    init_db()  
    app.run(debug=False)
