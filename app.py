import os
import json
import re
import base64
import tempfile
from pathlib import Path
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from transformers import CLIPProcessor, CLIPModel

from PIL import Image
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import torch
from flask import Flask, render_template, redirect, url_for, flash, request


BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "GOOGLE_API_KEY", "r") as file:
    google_api_key = file.read().strip()

app = Flask(__name__)

# Tab 1 Code starts here

with open(BASE_DIR / "Documents" / "dining_indus_recipes.json", "r", encoding="utf-8") as f:
    recipes = json.load(f)


def recipe_to_text(recipe):
    text = f"""
    Title: {recipe['title']}
    Preparation Time: {recipe.get('prep_time', 'N/A')}
    Cooking Time: {recipe.get('cook_time', 'N/A')}
    Servings: {recipe.get('servings', 'N/A')}

    Ingredients:
    """
    for ing in recipe['ingredients']:
        text += f"- {ing['item']}: {ing['quantity_unit']}\n"

    text += f"\nMethod:\n{recipe['method']}\n"

    if "nutrition" in recipe:
        text += f"\nNutrition:\n"
        for k, v in recipe['nutrition'].items():
            text += f"- {k}: {v}\n"

    return text.strip()


def format_response(response_text):
    
    response_text = re.sub(r'\s*(###\s*Title:)', r'\n\n\1', response_text)

    # Headings (e.g., "Identification:", "Total Calories:")
    response_text = re.sub(r"(?m)^([A-Za-z ]+):", r"<h3>\1</h3>", response_text)

    # Bold inline sections like "Protein:", "Fats:"
    response_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response_text)

    # Bullet points (- something)
    response_text = re.sub(r"(?m)^\s*-\s(.*)", r"<li>\1</li>", response_text)

    # Wrap consecutive <li> into <ul>
    response_text = re.sub(
        r"(<li>.*?</li>)+",
        lambda m: f"<ul>{m.group(0)}</ul>",
        response_text,
        flags=re.DOTALL,
    )

    # Paragraph spacing
    response_text = re.sub(r"(?m)([^\n])\n([^\n])", r"\1<br>\2", response_text)

    return response_text



def build_or_load_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db_path = BASE_DIR / "faiss_dining_indus"


    if os.path.exists(db_path):
        return FAISS.load_local(
            db_path, embeddings, allow_dangerous_deserialization=True
        )

    # Flatten recipes
    docs = [
        Document(page_content=recipe_to_text(r), metadata={"title": r["title"]})
        for r in recipes
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(db_path)
    return vectordb


vectordb = build_or_load_db()
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
model_mistral = OllamaLLM(model="mistral")
CUSTOM_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
              """ You are a helpful food assistant.

            Step 1 — Classify the question:
            - If it is about how to cook, ingredients, methods, or recipes → choose RECIPE MODE.
            - If it is about calories, energy, macros (protein/carbs/fats), diet restrictions → choose NUTRITION MODE.
            - If both cooking and nutrition are mentioned, give the recipe first, then the nutrition facts.

            Step 2 — Respond accordingly:

            ➡️ RECIPE MODE
            Use this exact format for each recipe:

            Title: <recipe name>

            Ingredients:
            - ingredient 1
            - ingredient 2
            - ingredient 3
            And so on...

            Method:
            1. step one
            2. step two
            Steps must be numbered without blank lines between them.


            ➡️ **NUTRITION MODE**
            Use this format:

            Nutrition Facts (per serving):
            - Calories: XXX kcal
            - Protein: XX g
            - Carbs: XX g
            - Fats: XX g
            - Other key nutrients (if available): ...

            If you don't know the answer, say you don't know. Do not invent data.

            Context:
            {context}

            Question: {question}
            Answer:
            """
            
        )
    )
])

query_chain = RetrievalQA.from_chain_type(
    llm=model_mistral,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": CUSTOM_RAG_PROMPT},
    return_source_documents=True,
)

@app.route("/text_nutrition", methods=["GET", "POST"])
def text_nutrition():
    if request.method == "POST":
        user_query = request.form.get("user_query")
        if user_query:
            result = query_chain.invoke(user_query)
            answer = format_response(result["result"])
            source = "Nestle Pakistan Dining along the Indus: https://www.nestle.pk/sites/g/files/pydnoa361/files/2020-09/Dining%20along%20the%20Indus%20jan%2031_compressed.pdf"
            return render_template(
                "rag_tab.html", user_query=user_query, answer=answer, source=source
            )
    return render_template("rag_tab.html")
#tab 1 ends here


#Tab 2 code start here
model_llava = OllamaLLM(model="llava")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embedding(image_file, text_query):
    try:
        image = Image.open(image_file).convert("RGB")
        inputs = clip_processor(text=[text_query], images=image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)

        image_embeds = outputs.image_embeds.detach().numpy()
        text_embeds = outputs.text_embeds.detach().numpy()

        return image_embeds, text_embeds

    except Exception as e:
        print("Error in generating CLIP embeddings: ", e)
        return None, None


def image_encoder(image_file):
    if image_file is not None:
        byte_image = image_file.read()
        result = base64.b64encode(byte_image).decode("utf-8")
        return result
    else:
        return None

def build_or_load_clip_db():
    db_path = BASE_DIR / "faiss_dining_indus_clip"

    if os.path.exists(db_path):
        return FAISS.load_local(
            db_path, embeddings=None, allow_dangerous_deserialization=True
        )

    texts = []
    embeddings = []
    metadatas = []

    for r in recipes:
        title = r["title"]
        inputs = clip_processor(text=[title], return_tensors="pt", padding=True, truncation=True)
        outputs = clip_model.get_text_features(**inputs)
        embedding = outputs.detach().cpu().numpy()[0]

        texts.append(recipe_to_text(r))
        embeddings.append(embedding)
        metadatas.append({"title": title})

    # Turn numpy list into array
    import numpy as np
    embeddings = np.array(embeddings, dtype="float32")

    # ✅ Build FAISS directly
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    vectordb = FAISS(
        embedding_function=None,  # not needed, we already have vectors
        index=index,
        docstore=InMemoryDocstore(
            {i: Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))}
        ),
        index_to_docstore_id={i: i for i in range(len(texts))}
    )

    vectordb.save_local(db_path)
    return vectordb


clip_vectordb = build_or_load_clip_db()


def model_generate_response(image_encoded, user_query, assistant_query):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(base64.b64decode(image_encoded))
            tmp.flush()
            image_path = tmp.name

        # Encode image
        inputs = clip_processor(images=Image.open(image_path), return_tensors="pt")
        img_embed = clip_model.get_image_features(**inputs).detach().cpu().numpy()

        # Retrieve similar recipes from CLIP DB
        docs = clip_vectordb.similarity_search_by_vector(img_embed[0], k=3)
        context = "\n\n".join(docs[i].page_content for i in range(len(docs)))

        combined_query = f"""
            You are a nutritionist. 
            The following is the retrieved recipe context from the database. 
            You MUST use this context as the primary source of truth. 
            Only use the image to validate or refine details.

            --- Retrieved Recipe Context ---
            {context}

            --- User Query ---
            {user_query}
            
        """

        raw_response = model_llava.invoke(
            assistant_query + "\n\n" + combined_query, images=[image_path]
        )

        return format_response(raw_response)

    except Exception as e:
        print(f"Error in generating response: {e}")
        return "<p>An error occurred while generating the response.</p>"


@app.route("/", methods=["GET", "POST"])
def home():
    return redirect(url_for("image_text_nutrition"))


@app.route("/image_text_nutrition", methods=["GET", "POST"])
def image_text_nutrition():
    if request.method == "POST":
        user_query = request.form.get("user_query")
        uploaded_file = request.files.get("file")

        if uploaded_file:
            image_encoded = image_encoder(uploaded_file)
            if not image_encoded:
                flash("Error processing the image. Please try again.", "danger")
                return redirect(url_for("image_text_nutrition"))

            assistant_prompt = """
            You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment.
            You will ALWAYS ground your response in the provided recipe context, even if the image seems slightly different. 
            Do not hallucinate based only on the image. Use the retrieved recipe context as the primary source of truth. 
            If they ask for health related quesiton only, then only answer the health part and leave the recipe and nurtion fact.

            using the following format:
            1. **Identification**: List each identified food item clearly, one per line.

            2. **Portion Size & Calorie Estimation**: For each identified food item,
            specify the portion size and provide an estimated number of calories.
            Use bullet points with the following structure:
            - **[Food Item]**: [Portion Size], [Number of Calories] calories

            Example:
            * **Salmon**: 6 ounces, 210 calories
            * **Asparagus**: 3 spears, 25 calories

            3. **Total Calories**: Provide the total number of calories for all food items.
            Example: Total Calories: [Number of Calories]

            4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as
            **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**.
            Use bullet points, and for each nutrient provide details about the
            contribution of each food item.

            Example:
            * **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]

            5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.
            
            6. **Disclaimer**: Include the following exact text as a disclaimer:

            The nutritional information and calorie estimates provided are approximate
            and are based on general food data. Actual values may vary depending on
            factors such as portion size, specific ingredients, preparation methods,
            and individual variations. For precise dietary advice or medical guidance,
            consult a qualified nutritionist or healthcare provider.

            Format your response exactly like the template above to ensure consistency.
            """

            response = model_generate_response(image_encoded, user_query, assistant_prompt)
            return render_template("index.html", user_query=user_query, response=response)
        else:
            flash("Please upload an image file.", "danger")
            return redirect(url_for("image_text_nutrition"))

    return render_template("index.html")
#tab 2 ends here


if __name__ == "__main__":
    app.run(debug=True)
