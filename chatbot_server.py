from flask import Flask, render_template_string, request, jsonify
import os
import json
import uuid
import traceback
import math
import sqlite3
import re
import time

# Text extraction imports
from werkzeug.utils import secure_filename
import pymupdf4llm  # Converts PDF to Markdown (preserves tables)
import docx

# Document processing & embeddings
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# HTTP to LLM API
import requests

print("Loading Pakistan Customs AI Agent...")

# --- CONFIGURATION ---
# Fix: Use absolute paths so this works from any terminal location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
INDEX_FOLDER = os.path.join(BASE_DIR, "vector_index")
DB_FILE = os.path.join(INDEX_FOLDER, "customs_data.db")
FAISS_INDEX_FILE = os.path.join(INDEX_FOLDER, "faiss.index")

# Model Config
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384 

# Chunking Config
CHUNK_SIZE = 1200      
CHUNK_OVERLAP = 300    

# LLM API Config (Local) - Checks Env Vars first, defaults to your keys
API_URL = os.environ.get("API_URL", "http://127.0.0.1:1337/v1/chat/completions")
API_KEY = os.environ.get("API_KEY")
LLM_MODEL = "Meta-Llama-3-8B-Instruct_IQ4_XS"

ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# --- DATABASE SETUP (SQLite) ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Table for Document Chunks
    c.execute('''CREATE TABLE IF NOT EXISTS documents 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  doc_uuid TEXT, 
                  filename TEXT, 
                  text_content TEXT, 
                  is_active INTEGER DEFAULT 1)''')
    
    # Table for Chat History (Updated with thread_id)
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  session_id TEXT, 
                  thread_id TEXT,
                  role TEXT, 
                  content TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Table for Threads
    c.execute('''CREATE TABLE IF NOT EXISTS threads 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  thread_id TEXT UNIQUE, 
                  title TEXT, 
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Migration: Check if thread_id column exists in chat_history for older DBs
    c.execute("PRAGMA table_info(chat_history)")
    columns = [info[1] for info in c.fetchall()]
    if "thread_id" not in columns:
        print("Migrating DB: Adding thread_id to chat_history...")
        c.execute("ALTER TABLE chat_history ADD COLUMN thread_id TEXT")
        
    conn.commit()
    conn.close()

init_db()

# --- INITIALIZATION ---
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# In-Memory Cache for FAISS mapping (Index -> DB Row)
# We load the active documents from SQLite into this list so FAISS indices match
doc_cache = []

def load_cache_and_index():
    """Loads active documents from SQLite into memory and initializes FAISS."""
    global index, doc_cache
    
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    # Load all active chunks in order
    rows = conn.execute("SELECT * FROM documents WHERE is_active=1 ORDER BY id").fetchall()
    conn.close()
    
    doc_cache = [dict(r) for r in rows]
    
    # Try to load existing FAISS index
    if os.path.exists(FAISS_INDEX_FILE):
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            print(f"Loaded FAISS index with {index.ntotal} vectors.")
            # Basic sync check
            if index.ntotal != len(doc_cache):
                print("Warning: Index count mismatch. Rebuilding index recommended if issues occur.")
        except Exception:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

load_cache_and_index()

# --- HELPER FUNCTIONS ---

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(filepath):
    ext = filepath.rsplit(".", 1)[1].lower()
    try:
        if ext == "pdf":
            return pymupdf4llm.to_markdown(filepath)
        elif ext == "docx":
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == "txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""
    return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text: return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0: start = 0
        if start >= length: break
    return [c for c in chunks if c]

def add_documents_to_index(file_path, original_filename):
    text = extract_text(file_path)
    if not text:
        return {"added": 0, "error": "No text extracted"}
    
    chunks = chunk_text(text)
    vectors = []
    new_db_entries = []
    
    # Process chunks
    for chunk in chunks:
        vec = embedder.encode(chunk, convert_to_numpy=True)
        norm = np.linalg.norm(vec)
        if norm > 0: vec = vec / norm
        vectors.append(vec.astype("float32"))
        
        doc_uuid = str(uuid.uuid4())
        new_db_entries.append((doc_uuid, original_filename, chunk))
        
    if vectors:
        # 1. Add to SQLite
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.executemany("INSERT INTO documents (doc_uuid, filename, text_content) VALUES (?, ?, ?)", new_db_entries)
        conn.commit()
        conn.close()
        
        # 2. Add to FAISS
        np_vectors = np.vstack(vectors)
        index.add(np_vectors)
        faiss.write_index(index, FAISS_INDEX_FILE)
        
        # 3. Update In-Memory Cache (append new items)
        # We fetch the just-inserted items to ensure we have the correct DB IDs/format
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        # Get the last N items we just added
        recents = conn.execute(f"SELECT * FROM documents WHERE is_active=1 ORDER BY id DESC LIMIT {len(vectors)}").fetchall()
        conn.close()
        
        # We need to append them in the correct order (recents comes out reversed due to DESC)
        for r in reversed(recents):
            doc_cache.append(dict(r))
            
    return {"added": len(vectors)}

def retrieve(query, top_k=4):
    if index.ntotal == 0:
        return []

    # --- Feature: Hybrid Search (Boost HS Codes) ---
    # 1. Check for HS Code pattern (e.g., 8501.1000)
    hs_code_matches = re.findall(r'\b\d{4}\.\d{4}\b', query)
    
    # 2. Vector Search (Semantic)
    q_vec = embedder.encode(query, convert_to_numpy=True)
    norm = np.linalg.norm(q_vec)
    if norm > 0: q_vec = q_vec / norm
    q_vec = np.expand_dims(q_vec.astype("float32"), axis=0)
    
    scores, idxs = index.search(q_vec, top_k * 3) 
    
    results = []
    seen_ids = set()

    # 3. Process Vector Results
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1 or idx >= len(doc_cache): continue
        entry = doc_cache[idx]
        
        # Filter deleted
        if entry.get("is_active") == 0: continue
        
        res_item = {
            "score": float(score),
            "text": entry["text_content"],
            "source": entry["filename"],
            "id": entry["id"]
        }
        results.append(res_item)
        seen_ids.add(entry["id"])

    # 4. Keyword Boost (Exact SQL Match for HS Codes)
    if hs_code_matches:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        for code in hs_code_matches:
            # Find docs containing this code
            rows = conn.execute("SELECT * FROM documents WHERE text_content LIKE ? AND is_active=1", (f"%{code}%",)).fetchall()
            for row in rows:
                if row['id'] not in seen_ids:
                    # Add as a high-relevance result
                    results.append({
                        "score": 2.0, # Artificially high score to float to top
                        "text": row['text_content'],
                        "source": row['filename'],
                        "id": row['id']
                    })
                else:
                    # Boost existing vector result
                    for r in results:
                        if r['id'] == row['id']:
                            r['score'] += 1.0 # Boost
        conn.close()

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

# --- AUTO-INDEXING ON STARTUP ---
def process_existing_uploads():
    print(f"\nScanning '{UPLOAD_FOLDER}' for new files...")
    
    # Get filenames already in DB
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT DISTINCT filename FROM documents WHERE is_active=1").fetchall()
    conn.close()
    existing_files = set(r[0] for r in rows)

    files_on_disk = os.listdir(UPLOAD_FOLDER)
    count = 0
    
    for filename in files_on_disk:
        if allowed_file(filename) and filename not in existing_files:
            print(f"--> Indexing new file: {filename} ...")
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                add_documents_to_index(file_path, filename)
                count += 1
            except Exception as e:
                print(f"    Error: {e}")
    
    if count > 0:
        print(f"Startup Scan: Indexed {count} new files.\n")
    else:
        print("Startup Scan: System up to date.\n")

process_existing_uploads()

# --- HTML TEMPLATE (Enhanced UI with Marked.js) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Pakistan Customs AI Consultant</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Internal:wght@400;600&display=swap" rel="stylesheet">
<style>
  body { font-family: 'Segoe UI', sans-serif; height: 100vh; margin: 0; display: flex; color: #333; background: #f4f6f8; }
  
  /* Sidebar */
  .sidebar { width: 280px; background: #00332a; color: #ecf0f1; display: flex; flex-direction: column; border-right: 1px solid #1a5c50; flex-shrink: 0; }
  .sidebar-header { padding: 20px; border-bottom: 1px solid #1a5c50; }
  .new-chat-btn { background: #004d40; border: 1px solid #4db6ac; color: white; padding: 12px; width: 100%; border-radius: 8px; cursor: pointer; transition: 0.2s; text-align: left; display: flex; align-items: center; gap: 10px; font-weight: 600; }
  .new-chat-btn:hover { background: #00695c; }
  
  .thread-list { flex: 1; overflow-y: auto; padding: 10px; }
  .thread-item { padding: 12px 15px; margin-bottom: 5px; border-radius: 8px; cursor: pointer; transition: 0.2s; display: flex; justify-content: space-between; align-items: center; font-size: 0.95em; color: #b2dfdb; position: relative; }
  .thread-item:hover { background: rgba(255,255,255,0.05); }
  .thread-item.active { background: rgba(255,255,255,0.15); color: white; border: 1px solid rgba(255,255,255,0.1); }
  
  .del-thread-btn { opacity: 0; background: none; border: none; color: #ef9a9a; cursor: pointer; font-size: 1.1em; padding: 0 5px; }
  .thread-item:hover .del-thread-btn { opacity: 1; }
  .del-thread-btn:hover { color: #ff5252; }

  /* Main Chat Area */
  .main-content { flex: 1; display: flex; flex-direction: column; background: #ffffff; position: relative; width: 0; } /* width:0 to fix flex overflow */
  
  .chat-header { padding: 15px 25px; background: white; border-bottom: 1px solid #e0e0e0; display: flex; align-items: center; justify-content: space-between; height: 60px; box-sizing: border-box; }
  .chat-title { font-weight: 600; font-size: 1.1rem; color: #004d40; display: flex; align-items: center; gap: 10px; }
  .header-actions button { background: none; border: 1px solid #ddd; padding: 6px 12px; border-radius: 6px; cursor: pointer; margin-left: 8px; font-size: 0.9em; transition: 0.2s; }
  .header-actions button:hover { background: #f5f5f5; }

  .chat-messages { flex: 1; overflow-y: auto; padding: 30px; display: flex; flex-direction: column; gap: 20px; background: #f9fafb; scroll-behavior: smooth; }
  
  .message { max-width: 80%; padding: 15px 20px; border-radius: 12px; font-size: 15px; line-height: 1.6; box-shadow: 0 1px 2px rgba(0,0,0,0.05); position: relative; }
  .user { align-self: flex-end; background: #004d40; color: white; border-bottom-right-radius: 2px; }
  .bot { align-self: flex-start; background: white; color: #2c3e50; border-bottom-left-radius: 2px; border: 1px solid #e1e4e8; }
  
  /* Markdown Styles */
  .bot table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.9em; }
  .bot th, .bot td { border: 1px solid #ddd; padding: 8px; text-align: left; }
  .bot th { background-color: #f1f8e9; color: #33691e; }
  .bot pre { background: #f4f6f8; padding: 10px; border-radius: 5px; overflow-x: auto; }
  
  .chat-input-area { padding: 20px 30px; background: white; border-top: 1px solid #eee; display: flex; gap: 15px; align-items: center; }
  #userInput { flex: 1; padding: 15px; border-radius: 25px; border: 1px solid #ced4da; font-size: 15px; outline: none; background: #f8f9fa; transition: 0.2s; }
  #userInput:focus { border-color: #004d40; background: white; box-shadow: 0 0 0 3px rgba(0,77,64,0.1); }
  #sendBtn { width: 45px; height: 45px; border-radius: 50%; background: #004d40; color: white; border: none; cursor: pointer; display: flex; justify-content: center; align-items: center; font-size: 1.2em; transition: 0.2s; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
  #sendBtn:hover { transform: translateY(-2px); background: #00695c; }
  #sendBtn:disabled { background: #ccc; cursor: default; transform: none; }

  /* Modal */
  .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; justify-content: center; align-items: center; }
  .modal-content { background: white; width: 500px; max-width: 90%; max-height: 80vh; border-radius: 12px; padding: 25px; display: flex; flex-direction: column; box-shadow: 0 10px 25px rgba(0,0,0,0.2); }
  .file-list { overflow-y: auto; flex: 1; margin-top: 15px; max-height: 300px; border: 1px solid #eee; border-radius: 8px; }
  .file-item { display: flex; justify-content: space-between; padding: 10px 15px; border-bottom: 1px solid #eee; background: white; }
  .file-item:last-child { border-bottom: none; }
  .file-item:hover { background: #f9f9f9; }
</style>
</head>
<body>

<!-- Sidebar -->
<div class="sidebar">
  <div class="sidebar-header">
     <button class="new-chat-btn" onclick="createNewThread()">
       <span>+</span> New Chat
     </button>
  </div>
  <div class="thread-list" id="threadList">
     <!-- Populated by JS -->
  </div>
  <div style="padding: 20px; font-size: 0.75em; text-align: center; color: #5aa192; border-top: 1px solid #1a5c50;">
     Pakistan Customs AI<br>Powered by RAG
  </div>
</div>

<!-- Main Interface -->
<div class="main-content">
  <div class="chat-header">
    <div class="chat-title" id="headerTitle">
      <span id="threadName">Welcome</span>
      <span style="font-size:0.8em; color:#999; font-weight:400; cursor:pointer;" onclick="editTitle()">✎</span>
    </div>
    <div class="header-actions">
      <button onclick="toggleModal()">📁 Knowledge Base</button>
      <button onclick="clearChat()" title="Reset Context">🧹 Reset</button>
    </div>
  </div>

  <div class="chat-messages" id="chatMessages">
    <!-- Messages Go Here -->
  </div>

  <div class="chat-input-area">
    <input type="text" id="userInput" placeholder="Ask about HS Codes, Duties..." onkeypress="handleKeyPress(event)" />
    <button id="sendBtn" onclick="sendMessage()">➤</button>
  </div>
</div>

<!-- Knowledge Base Modal -->
<div class="modal-overlay" id="kbModal">
  <div class="modal-content">
    <div style="display:flex; justify-content:space-between; font-weight:bold; color:#004d40; font-size:1.1em; margin-bottom:10px;">
      <span>📚 Knowledge Base</span>
      <span style="cursor:pointer; color:#888" onclick="toggleModal()">×</span>
    </div>
    <div style="font-size:0.9em; color:#666;">Manage PDF/Docs for RAG context.</div>
    <div class="file-list" id="docList"></div>
    <div style="margin-top:20px; display:flex; gap:10px">
       <input id="fileInput" type="file" style="display:none" onchange="uploadFile()"/>
       <button style="background:#004d40; color:white; border:none; padding:10px 15px; border-radius:6px; flex:1; cursor:pointer;" onclick="document.getElementById('fileInput').click()">+ Upload File</button>
    </div>
  </div>
</div>

<script>
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const modal = document.getElementById('kbModal');
const threadListEl = document.getElementById('threadList');
const threadNameEl = document.getElementById('threadName');

let currentThreadId = null;

// Init
window.onload = async function() {
    await loadThreads();
    // If we have threads, load the first one, else create one
    if(document.querySelectorAll('.thread-item').length === 0) {
        await createNewThread();
    } else {
        // Just select the first one if not selected?
        // Actually loadThreads should handle UI render, we just need to decide which to active.
        // Let's assume loadThreads sets up the list. We need to pick one.
        // We'll trust the order and pick the top one.
        const firstId = threadListEl.firstElementChild.dataset.id;
        if(firstId) selectThread(firstId);
    }
};

async function loadThreads() {
    const res = await fetch('/threads');
    const data = await res.json();
    threadListEl.innerHTML = '';
    data.threads.forEach(t => {
        const div = document.createElement('div');
        div.className = 'thread-item';
        div.dataset.id = t.id;
        div.onclick = (e) => {
             // Avoid triggering select when clicking delete
             if(!e.target.classList.contains('del-thread-btn')) selectThread(t.id);
        };
        div.innerHTML = `
           <span style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1;">${t.title}</span>
           <button class="del-thread-btn" onclick="deleteThread('${t.id}')">×</button>
        `;
        threadListEl.appendChild(div);
    });
}

async function createNewThread() {
    const res = await fetch('/threads', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({title: 'New Chat'})
    });
    const data = await res.json();
    await loadThreads();
    selectThread(data.id);
}

async function selectThread(id) {
    currentThreadId = id;
    
    // UI Update
    document.querySelectorAll('.thread-item').forEach(el => el.classList.remove('active'));
    const activeEl = document.querySelector(`.thread-item[data-id='${id}']`);
    if(activeEl) {
        activeEl.classList.add('active');
        threadNameEl.innerText = activeEl.querySelector('span').innerText;
    }

    // Load History
    const res = await fetch(`/history?thread_id=${id}`);
    const data = await res.json();
    
    chatMessages.innerHTML = '';
    if (data.history.length === 0) {
        const welcomeText = `<strong>Welcome to the Pakistan Customs AI Consultant!</strong><br><br>
        I am an advanced AI specialized in Pakistan Customs Law. My capabilities include:<br>
        • 🔍 <strong>HS Code Classification</strong> for your products.<br>
        • 💰 <strong>Duty & Tax Calculation</strong> based on FBR Tariffs.<br>
        • 📜 <strong>SRO & Policy Research</strong> using official documents.<br><br>
        <em>Upload your research documents or ask me a question to begin.</em>`;
        appendMsg(welcomeText, false, false);
    } else {
        data.history.forEach(msg => {
            appendMsg(msg.content, msg.role === 'user', false);
        });
    }
}

async function deleteThread(id) {
    if(!confirm('Delete this chat permanently?')) return;
    await fetch(`/threads/${id}`, {method:'DELETE'});
    
    if(currentThreadId === id) {
        currentThreadId = null;
        chatMessages.innerHTML = '';
        threadNameEl.innerText = "Deleted";
    }
    await loadThreads();
    // If we deleted the active one, switch to another
    if(currentThreadId === null && threadListEl.firstElementChild) {
        selectThread(threadListEl.firstElementChild.dataset.id);
    }
}

async function editTitle() {
    if(!currentThreadId) return;
    const newTitle = prompt("Rename chat:", threadNameEl.innerText);
    if(newTitle) {
        await fetch('/rename_thread', {
            method:'POST', 
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({thread_id: currentThreadId, title: newTitle})
        });
        await loadThreads();
        selectThread(currentThreadId); // Refresh title in sidebar
    }
}

// --- Modified Existing Functions ---

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;
    if (!currentThreadId) { alert("Please select or create a chat first."); return; }
    
    appendMsg(text, true);
    userInput.value = '';
    sendBtn.disabled = true;

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({message: text, thread_id: currentThreadId})
        });
        const data = await res.json();
        const responseText = data.response || "Error: " + (data.error || "No response");
        appendMsg(responseText, false, true); 
    } catch (e) {
        appendMsg("Connection Error.", false, false);
    }
    sendBtn.disabled = false;
    // Auto-rename thread if first message? Implementation optional but nice.
}

async function clearChat(){
  if(!currentThreadId) return;
  if(confirm("Clear history for this chat?")) {
      await fetch('/clear',{
        method:'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({thread_id: currentThreadId})
      });
      chatMessages.innerHTML = '';
      appendMsg("Chat Reset.", false);
  }
}

// --- Utils (Same as before) ---
function appendMsg(text, isUser, animate=true) {
  const div = document.createElement('div');
  div.className = 'message ' + (isUser ? 'user' : 'bot');
  chatMessages.appendChild(div);

  if (isUser) {
    div.innerText = text;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return;
  }
  
  const rawHTML = marked.parse(text);
  if (!animate) {
    div.innerHTML = rawHTML;
    chatMessages.scrollTop = chatMessages.scrollHeight;
  } else {
    div.innerHTML = rawHTML;
    div.style.opacity = 0;
    let op = 0.1;
    const timer = setInterval(function () {
        if (op >= 1){ clearInterval(timer); }
        div.style.opacity = op;
        op += op * 0.1;
    }, 10);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

function handleKeyPress(e){ if (e.key === 'Enter') sendMessage(); }

// Modal Stuff
function toggleModal() {
  modal.style.display = (modal.style.display === 'flex') ? 'none' : 'flex';
  if(modal.style.display === 'flex') refreshDocList();
}
modal.addEventListener('click', e => { if(e.target === modal) toggleModal(); });

async function uploadFile(){
  const file = document.getElementById('fileInput').files[0];
  if(!file) return;
  const fd = new FormData();
  fd.append('file', file);
  alert("Uploading... please wait.");
  const res = await fetch('/upload', {method:'POST', body: fd});
  const data = await res.json();
  if(data.ok) { alert("Uploaded!"); refreshDocList(); }
  else { alert("Error: " + data.error); }
}

async function deleteFile(filename) {
    if(!confirm("Delete " + filename + "?")) return;
    await fetch('/delete', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({filename: filename})
    });
    refreshDocList();
}

async function refreshDocList(){
  const res = await fetch('/docs');
  const data = await res.json();
  const list = document.getElementById('docList');
  list.innerHTML = '';
  data.docs.forEach(d => {
      list.innerHTML += `<div class="file-item"><span>📄 ${d}</span><span style="color:red;cursor:pointer" onclick="deleteFile('${d}')">🗑️</span></div>`;
  });
}
</script>
</body>
</html>
"""

# --- ROUTES ---

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        thread_id = data.get('thread_id')
        session_id = request.remote_addr 
        
        if not thread_id:
            return jsonify({"error": "Missing thread_id"}), 400

        # 1. Save User Msg to SQLite
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT INTO chat_history (session_id, thread_id, role, content) VALUES (?, ?, ?, ?)", 
                     (session_id, thread_id, "user", user_message))
        conn.commit()
        conn.close()

        # 2. Hybrid Retrieval
        retrieved = retrieve(user_message, top_k=4)
        context_pieces = [f"Source: {r['source']}\n{r['text']}\n---" for r in retrieved]
        context_text = "\n\n".join(context_pieces) if context_pieces else "No context found."

        # 3. Prompt Construction
        system_msg = {
            "role": "system",
            "content": (
                "You are an expert Customs Clearing Agent for Pakistan (Pakistan Customs AI Consultant). "
                "Your goal is to provide accurate duties and taxes based on the provided FBR Tariff/SRO context. "
                "CRITICAL INSTRUCTION: If the user asks for details about an HS Code (PCT Code), you MUST format your response EXACTLY as the following Markdown table. "
                "Do not summarize or skip fields. If a value is not found in context, write 'Not Specification in Uploaded Docs'.\n\n"
                "**HS CODE STRUCTURAL LOGIC:**\n"
                "- The **first 4 digits** (Heading) determine the **Main Description** and **Explanatory Notes**.\n"
                "- The **last 4 digits** (Subheading) determine the specific **Short Description**.\n"
                "- Example: For 8517.7900, '8517' gives the Main Description/Notes, and '7900' gives the specific item description.\n\n"
                "**Required Table Format:**\n"
                "| Field | Value |\n"
                "| :--- | :--- |\n"
                "| **PCT CODE** | [Insert Code] |\n"
                "| **Short Description** | [Specific description based on full 8 digits] |\n"
                "| **Main Description** | [Broader category description based on first 4 digits] |\n"
                "| **UoM** | [Unit of Measurement] |\n"
                "| **Customs Duty** | [% Rate] |\n"
                "| **Additional CD** | [% Rate or SRO] |\n"
                "| **Regulatory Duty** | [% Rate or SRO] |\n"
                "| **Sales Tax** | [% Rate] |\n"
                "| **Value Addition Tax** | [Usually 3% unless specified] |\n"
                "| **Advance Income Tax** | [WHT Rate for Filer/Non-Filer] |\n"
                "| **PTAs / FTAs** | [List exemptions like China, Malaysia, Turkiye with rates] |\n\n"
                "**EXPLANATORY NOTES**\n"
                "[Provide detailed notes derived from the **First 4 Digits** (Heading) of the HS Code. Include item inclusions/exclusions.]\n\n"
                "If the query is NOT about a specific HS Code, answer normally using the context."
                "\n\nCONTEXT DATA:\n" + context_text
            )
        }

        # 4. Fetch History for LLM Context (Scoped to Thread)
        conn = sqlite3.connect(DB_FILE)
        hist_rows = conn.execute("SELECT role, content FROM chat_history WHERE thread_id=? ORDER BY id DESC LIMIT 6", (thread_id,)).fetchall()
        conn.close()
        
        history_msgs = [{"role": r[0], "content": r[1]} for r in hist_rows][::-1]
        messages_for_api = [system_msg] + history_msgs

        # 5. Call LLM
        payload = {
            "model": LLM_MODEL,
            "messages": messages_for_api,
            "temperature": 0.1,
            "max_tokens": 800
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        ai_response = "Error parsing response."
        if resp.status_code == 200:
            ai_response = resp.json().get("choices", [{}])[0].get("message", {}).get("content", ai_response)

        # 6. Save Assistant Msg to SQLite
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT INTO chat_history (session_id, thread_id, role, content) VALUES (?, ?, ?, ?)", 
                     (session_id, thread_id, "assistant", ai_response))
        conn.commit()
        conn.close()

        return jsonify({"response": ai_response})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    session_id = request.remote_addr
    thread_id = request.args.get('thread_id')
    
    conn = sqlite3.connect(DB_FILE)
    if thread_id:
        rows = conn.execute("SELECT role, content FROM chat_history WHERE thread_id=? ORDER BY id ASC", (thread_id,)).fetchall()
    else:
        # Fallback or global history (optional, currently just by session IP if no thread)
        # But with sidebar we likely always want a thread.
        # Let's return empty or all for this IP if no thread specified?
        # Better to return empty list if no thread to avoid confusion.
        rows = []
        
    conn.close()
    return jsonify({"history": [{"role": r[0], "content": r[1]} for r in rows]})

@app.route('/threads', methods=['GET'])
def get_threads():
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT thread_id, title, created_at FROM threads ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify({"threads": [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]})

@app.route('/threads', methods=['POST'])
def create_thread():
    data = request.json or {}
    title = data.get('title', 'New Chat')
    thread_id = str(uuid.uuid4())
    
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO threads (thread_id, title) VALUES (?, ?)", (thread_id, title))
    conn.commit()
    conn.close()
    
    return jsonify({"id": thread_id, "title": title})

@app.route('/threads/<thread_id>', methods=['DELETE'])
def delete_thread_route(thread_id):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM threads WHERE thread_id=?", (thread_id,))
    conn.execute("DELETE FROM chat_history WHERE thread_id=?", (thread_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

@app.route('/rename_thread', methods=['POST'])
def rename_thread():
    data = request.json
    thread_id = data.get('thread_id')
    new_title = data.get('title')
    
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE threads SET title=? WHERE thread_id=?", (new_title, thread_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route('/clear', methods=['POST'])
def clear():
    data = request.json
    thread_id = data.get('thread_id')
    
    conn = sqlite3.connect(DB_FILE)
    if thread_id:
        conn.execute("DELETE FROM chat_history WHERE thread_id=?", (thread_id,))
    else:
        # If no thread specified, maybe don't delete anything or use IP?
        # Safety: require thread_id
        pass
        
    conn.commit()
    conn.close()
    return jsonify({"status": "cleared"})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files: return jsonify({"ok": False, "error": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"ok": False, "error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        
        # Add to Index and DB
        result = add_documents_to_index(save_path, filename)
        return jsonify({"ok": True, "filename": filename, "indexed": result}), 200
    return jsonify({"ok": False, "error": "Invalid file type"}), 400

@app.route('/docs', methods=['GET'])
def docs():
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT DISTINCT filename FROM documents WHERE is_active=1").fetchall()
    conn.close()
    return jsonify({"docs": sorted([r[0] for r in rows])})

@app.route('/delete', methods=['POST'])
def delete_file():
    data = request.json
    filename = data.get('filename')
    
    # Soft delete in SQLite
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE documents SET is_active=0 WHERE filename=?", (filename,))
    conn.commit()
    conn.close()
    
    # Reload cache to update search results immediately
    load_cache_and_index()
    
    # Remove physical file
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path): os.remove(path)
    
    return jsonify({"ok": True})

if __name__ == '__main__':
    print("Starting Pakistan Customs RAG Chatbot on http://localhost:8088")
    app.run(debug=True, use_reloader=False, host='localhost', port=8088)