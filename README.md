# 🥗 AI Nutrition Assistant — Multimodal RAG System  
**Localized Dietary Guidance using Retrieval-Augmented Generation (RAG) and Multimodal AI**

---

##  Overview
The **AI Nutrition Assistant** is an intelligent dietary advisor that provides **accurate, localized, and culturally relevant nutrition information** for **Pakistani foods and recipes**.  
Unlike generic global nutrition apps, this system uses a **custom dataset** from *“Dining Along the Indus”* — an official **Nestlé Pakistan x Government of Pakistan** publication containing authentic local recipes and nutritional data.

It combines **Retrieval-Augmented Generation (RAG)** with **multimodal AI**, enabling:
- Context-aware **nutrition Q&A**
- **Image + text** based meal analysis

This project bridges the gap between **global AI models and local dietary data**, promoting **AI-driven health awareness in Pakistan**.

---

##  Motivation
Pakistan faces rising challenges in nutrition and dietary misinformation.  
Most AI nutrition systems rely on Western food datasets, producing irrelevant advice for South Asian users.

This project aims to:
- Localize AI nutrition knowledge for Pakistani users  
- Enable multimodal (image + text) analysis for meals  
- Demonstrate **RAG + Multimodal AI** for real-world, context-specific applications  

---

##  Key Features
###  Chat Mode
- Natural language nutrition Q&A  
- Retrieves data from *Dining Along the Indus* dataset  
- Provides calorie breakdowns, macro analysis, and recipe suggestions  

###  Multimodal Mode
- Accepts **image + text input**  
- Analyzes meal images and gives contextual health/nutrition feedback  

### ⚙️ System Capabilities
- **OCR Extraction:** Parses and cleans PDF-based nutritional data  
- **RAG Pipeline:** Combines embeddings + retrieval for contextual accuracy  
- **Gradio Interface:** Dual-tab UI for chat and multimodal analysis  

---

##  Tech Stack
| Category | Tools / Libraries |
|-----------|-------------------|
| **LLMs & RAG** | LangChain · FAISS · OpenAI Embeddings |
| **Data Processing** | Python · PyPDF2 · Tesseract OCR · Pandas |
| **Interface** | Gradio (Dual-tab Interface) |
| **AI Models** | Mistral / OpenAI API |
| **Backend (Optional)** | Flask / FastAPI |

---

##  System Architecture

```bash
+---------------------------+
|        User Input         |
|     (Text / Image + Text) |
+------------+--------------+
             |
             v
+---------------------------+
|      Input Processing     |
| OCR / Preprocessing / Embeddings |
+------------+--------------+
             |
             v
+---------------------------+
|     FAISS Vector Store    |
|  (Custom Pakistani Dataset) |
+------------+--------------+
             |
             v
+---------------------------+
|     LLM (RAG Pipeline)    |
|  Context Retrieval + Response |
+------------+--------------+
             |
             v
+---------------------------+
|       Gradio UI Output    |
+---------------------------+

```


---

##  Dataset Source
 **Dining Along the Indus (Nestlé Pakistan x Government of Pakistan)**  
 [Download PDF](https://www.nestle.pk/sites/g/files/pydnoa361/files/2020-09/Dining%20Along%20The%20Indus%20-%20PDF.pdf)

Data extracted via OCR and structured parsing — includes:
- Recipe names  
- Ingredients  
- Nutritional facts per serving (calories, protein, fat, carbohydrates)  

---

##  How It Works
1. **Data Extraction:** PDF processed with OCR → recipes + nutrition tables parsed  
2. **Vectorization:** Data embedded via OpenAI embeddings → FAISS vector store  
3. **RAG Retrieval:** User query → contextual match → relevant data injected  
4. **Response Generation:** Model produces context-based response  
5. **Multimodal Input:** Optional image upload for food analysis  

---

##  Demo Interface
**Main Tabs:**
-  **Nutrition Q&A** — text-based chat interface  
-  **Image + Text Analyzer** — upload meal images for AI evaluation  

*(Screenshots or short GIFs can be added here)*  

---

##  Example Queries
- “How many calories are in one serving of Chicken Biryani?”  
- “Create a 1,500-calorie meal plan using Pakistani recipes.”  
- “Is this breakfast healthy?” *(with image upload)*  

---

##  Installation
```bash
# Clone this repository
git clone https://github.com/yourusername/ai-nutrition-assistant.git
cd ai-nutrition-assistant

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

```

Then open your browser and visit http://localhost:7860 to interact with the interface.

---

## Pictures
### **Chat bot based**
<img width="1919" height="875" alt="Nutrition Assitant 1(text)" src="https://github.com/user-attachments/assets/294610eb-7563-4a70-8280-232631dcf5c5" />
<img width="1919" height="870" alt="Nutrition Assitant 2 (text)" src="https://github.com/user-attachments/assets/7fdb178e-4019-4474-a6b4-a2142958b42a" />
<img width="1919" height="874" alt="Nutrition Assitant 3 (text)" src="https://github.com/user-attachments/assets/4b268eb3-3221-42f0-bc67-fe85e38b9171" />
<img width="1276" height="466" alt="Nutrition Assitant 4 (text)" src="https://github.com/user-attachments/assets/7867b3ee-57d7-4358-af56-1200048ad8b6" />

### **Multimodal Mode**
<img width="1904" height="869" alt="Nutrition Assistant 1(multi)" src="https://github.com/user-attachments/assets/e21e7c60-597d-42b4-b46c-3ee63f78ef4f" />
<img width="1462" height="870" alt="Nutrition Assistant 2(multi)" src="https://github.com/user-attachments/assets/5c0302c8-5010-4e8f-851a-5acf8c385fa6" />
<img width="1212" height="879" alt="Nutrition Assistant 3(multi)" src="https://github.com/user-attachments/assets/b11b8463-393b-4e61-bc82-ed8e280bfbbd" />
<img width="1322" height="792" alt="Nutrition Assistant 4(multi)" src="https://github.com/user-attachments/assets/73033163-2b4a-4338-bbc8-23264bfaf1f1" />


---
## Future Work
- Integration with Agentic AI systems for personalized meal planning
- Add voice input and mobile-friendly UI
- Expand dataset to include regional cuisines

---
## Author
- Muhammad Abdul Rehman Salahuddin
- BS Artificial Intelligence — Shifa Tameer-e-Millat University
- Islamabad, Pakistan
- muhammadabdulrehmansalahuddin@gmail.com

---

## License
This project is licensed under the MIT License.
Dataset credit: Nestlé Pakistan & Government of Pakistan (for research and educational use only).
