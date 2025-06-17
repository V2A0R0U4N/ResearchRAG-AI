# ResearchRAG-AI
![image](https://github.com/user-attachments/assets/f11f6e51-9f09-4939-ace6-128b07c368b3)



## Features

- Add one or more article URLs to fetch content dynamically.
- Process content using **LangChain's UnstructuredURLLoader**.
- Clean, split, and embed article content using **HuggingFace Sentence Transformers**.
- Store embedding vectors using **FAISS** for efficient semantic retrieval.
- Interact with **DeepSeek LLM** through a sleek Streamlit interface to ask research questions.
- Get structured answers and source-linked insights.
- Reuse embeddings with a locally stored FAISS `.pkl` file.

## 🔧 Installation

1. **Clone this repository** to your local machine:
   ```bash
   git clone https://github.com/your-username/ResearchRAG-AI.git
   
2. Navigate to the project folder:
   ```bash
   cd ResearchRAG-AI
3. Install the required packages using pip:
  ```bash
  pip install -r requirements.txt
```
4. Create a .env file in the root directory and add your DeepSeek API key:
   ```bash
   OPENAI_API_KEY=your_deepseek_api_key
   OPENAI_API_BASE=https://api.deepseek.com
   ```

## 🚀 Usage

1. Run the Streamlit App
   ```bash
   streamlit run main.py
   ```
2. The app will open in your browser.
3. In the sidebar:
   
   - Enter article URLs (up to 3 or more if extended).

   - Click Analyze Articles to fetch, split, embed, and index the content.

4. On the right side:

   - Enter your research question.

   - Click Submit Question to get answers based on the processed articles.

  ## 🧪 Example Articles Used<br>
  
  -  https://www.moneycontrol.com/news/business/markets/taking-stock-market-ends-higher-amid-positive-global-cues-12568811.html  
  - https://www.econlib.org/library/Enc/StockMarket.html

## Project Structure
- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- faiss_store_openai.pkl: A pickle file to store the FAISS index.
- .env: Configuration file for storing your OpenAI API key.


