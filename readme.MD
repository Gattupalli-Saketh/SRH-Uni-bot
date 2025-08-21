


```markdown
# 📚 RAG Agent for SRH University  

This project is a **Retrieval-Augmented Generation (RAG) agent** built to assist students and staff at **SRH University** with answering queries related to the university, courses, admissions, and general information.  

It combines the power of **LLaMA 3.2** for natural language understanding and **Sentence Transformers** for efficient embedding-based retrieval.  

---

## 🚀 Features  
- **Question Answering**: Ask university-related questions and get precise answers.  
- **RAG Pipeline**: Combines a vector database with LLaMA 3.2 for context-aware responses.  
- **Semantic Search**: Uses Sentence Transformer embeddings to retrieve the most relevant documents.  
- **Scalable**: Can be expanded with more university documents (course guides, policies, FAQs).  

---

## 🛠️ Tech Stack  
- **LLM**: LLaMA 3.2  
- **Embedding Model**: Sentence Transformer  
- **Frameworks**: Python 
- **Vector Store**: Chroma  
- **Data**: University docs, FAQs, guidelines  

---

---

## ⚙️ Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/Gattupalli-Saketh/SRH-Uni-bot.git
   cd srh-Uni-bot
````

2. Create a virtual environment & install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download or prepare university documents and place them inside the `data/` folder.

---

## ▶️ Usage

Run the RAG agent:

```bash
python RAG/front.py
```

Ask a question:

```bash
>>> "What are the admission requirements for Data Science Master's?"
```

Example Output:

```
The admission requirements include a bachelor's degree in a relevant field, English language proficiency, and supporting documents such as transcripts and a CV.
```

---

## 📌 Future Improvements

* Deploy as a **Streamlit / FastAPI app** for easy access.
* Add **multilingual support** (German & English).
* Integrate with **university portal** for real-time updates.
* Improve retrieval with **hybrid search (BM25 + embeddings)**.

---

## 🤝 Contributing

Pull requests and feature suggestions are welcome!

---

## 📄 License

MIT License

```

---

Do you want me to also **add badges (Python, LangChain, LLaMA, Sentence Transformers)** and a **usage diagram** for the README? That would make it look more professional.
```
