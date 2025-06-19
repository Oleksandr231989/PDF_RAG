# 📄 PDF RAG Q&A System

A powerful web-based application that allows you to upload PDF documents and ask questions about them using AI. Get intelligent answers with visual page references!

## 🎯 Features

- **📤 PDF Upload**: Upload any PDF document
- **🤖 AI Q&A**: Ask questions in natural language
- **📄 Page Citations**: Get answers with specific page references
- **🖼️ Visual Pages**: See actual PDF pages with charts and graphs
- **💬 Chat History**: Track all your questions and answers
- **📊 Analytics**: Document statistics and analysis

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Model**: OpenAI GPT-4o Mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: ChromaDB
- **PDF Processing**: PyPDF2, pdf2image
- **Text Processing**: LangChain

## 🔑 Requirements

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **PDF Document**: Any PDF file you want to analyze

## 🚀 How to Use

1. **Enter API Key**: Add your OpenAI API key in the sidebar
2. **Upload PDF**: Choose and upload your PDF document
3. **Process Document**: Click "Process PDF" to analyze the document
4. **Ask Questions**: Go to "Ask Questions" tab and start querying
5. **View Results**: Get AI answers with relevant page visuals

## 💡 Sample Questions

- "What is the financial performance?"
- "What are the key metrics and KPIs?"
- "What challenges are mentioned?"
- "Show me market analysis information"
- "What are the main conclusions?"

## 🏃‍♂️ Running Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/pdf-rag-app.git
cd pdf-rag-app

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install poppler-utils

# Run the app
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy your app
5. Share the URL with others!

## 📊 How It Works

1. **PDF Processing**: Extracts text and converts pages to images
2. **Text Chunking**: Breaks document into manageable pieces
3. **Vector Embeddings**: Converts text to numerical representations
4. **Semantic Search**: Finds relevant content based on meaning
5. **AI Answers**: Generates responses using GPT-4o Mini
6. **Visual Pages**: Shows original PDF pages with charts and formatting

## 🔧 Configuration

The app uses the following default settings:
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Embedding Model**: text-embedding-3-small
- **LLM Model**: GPT-4o Mini
- **Max Pages Displayed**: 3 per query

## 🎉 Perfect for

- 📈 Business reports analysis
- 📋 Research paper exploration
- 📊 Financial document review
- 📝 Technical manual queries
- 🔍 Legal document search

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/pdf-rag-app/issues) page
2. Create a new issue if needed
3. Provide detailed information about the problem

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenAI](https://openai.com/) for AI capabilities
- [LangChain](https://langchain.com/) for RAG implementation
- [ChromaDB](https://www.trychroma.com/) for vector storage

---

**Built with ❤️ for better document analysis**
