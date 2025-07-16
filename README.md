# 🦙 LlamaIndex RAG Plugin for Dify

Advanced RAG (Retrieval-Augmented Generation) tool powered by LlamaIndex for local document querying and knowledge extraction.

## 🚀 Features

- **Local-First Design**: Works entirely with local files and Ollama
- **Advanced RAG**: Semantic search with document indexing
- **Ollama Integration**: Supports multiple local LLM models
- **Fencing Club Optimized**: Special handling for fencing terminology
- **Sales Detection**: Identifies potential sales opportunities
- **Multi-Format Support**: PDF, TXT, MD, DOCX files

## 📦 Installation

1. **Go to Dify Studio** → Settings → Plugins
2. **Choose "Install from GitHub"**
3. **Enter Repository URL:** `https://github.com/YOUR_USERNAME/dify-llamaindex-plugin`
4. **Select Version:** `v1.0.0`
5. **Install Plugin**

## 🔧 Configuration

### Required Parameters
- **Query**: Your question or search query
- **Document Path**: Path to your documents (e.g., `/data/fencing_docs`)

### Optional Parameters
- **Model**: Ollama model (llama3.2, mistral:7b, phi3:mini, codellama:7b)
- **Top K**: Number of results to retrieve (1-10)
- **Response Mode**: context_only, synthesized, tree_summarize, compact
- **Sales Detection**: Enable/disable sales opportunity detection
- **Debug Mode**: Enable detailed logging

## 🎯 Usage Example

### Basic Query
```
Query: "What are the basic fencing techniques?"
Document Path: "/data/fencing_docs"
Model: "llama3.2"
Top K: 3
```

### Advanced Query
```
Query: "How to improve sabre footwork?"
Document Path: "/data/fencing_docs/techniques_advanced"
Model: "mistral:7b"
Response Mode: "synthesized"
Sales Detection: true
```

## 📁 Document Organization

Recommended directory structure:
```
/data/fencing_docs/
├── weapons_basics/
├── weapons_advanced/
├── rules_and_regulations/
├── equipment_and_gear/
├── techniques_basic/
├── techniques_advanced/
├── competition_guide/
├── training_programs/
└── general/
```

## 🔍 How It Works

1. **Document Indexing**: Creates semantic indexes using LlamaIndex
2. **Query Processing**: Preprocesses queries with fencing-specific terms
3. **Semantic Search**: Finds most relevant document chunks
4. **Response Generation**: Uses Ollama to synthesize answers
5. **Sales Detection**: Identifies potential business opportunities

## 🛠️ Technical Details

- **Framework**: LlamaIndex 0.12.49
- **LLM Integration**: Ollama (local)
- **Embeddings**: Ollama embeddings
- **Vector Store**: SimpleVectorStore (local)
- **File Support**: PDF, TXT, MD, DOCX, RTF

## 🎪 Sample Documents

The plugin includes sample fencing documents for testing:
- Foil, Épée, and Sabre basics
- Equipment guides
- Basic techniques
- Rules and regulations

## 🔐 Security Features

- **Path Restrictions**: Only allowed paths can be accessed
- **Input Validation**: All inputs are sanitized
- **Local Processing**: No external API calls
- **File Size Limits**: Prevents memory exhaustion

## 🚀 Performance

- **Initial Indexing**: 10-30 seconds (depends on document size)
- **Cached Queries**: 1-3 seconds
- **New Queries**: 3-8 seconds
- **Memory Usage**: Configurable with chunk size settings

## 📊 Requirements

- **Dify**: v1.5.1+
- **Ollama**: Running locally with models
- **Python**: 3.12+
- **Memory**: 2GB recommended

## 🎯 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Issues and pull requests welcome! This plugin is designed to be extensible and customizable.

---

**Ready to revolutionize your document querying with AI!** 🦙✨