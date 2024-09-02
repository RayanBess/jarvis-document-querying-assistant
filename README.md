# Project: Voice-Activated Document Querying

This project enables voice recognition to query documents using Retrieval Augmented Generation (RAG). It leverages the **Ollama DolphinPhi** model for query processing, **Parler TTS** for text-to-speech, and **Whisper** for speech-to-text conversion.

## Features
- **Voice Recognition**: Use voice commands to interact with your documents.
- **Retrieval Augmented Generation**: Enhance your queries with context-aware document retrieval.
- **Natural Language Processing**: Employ advanced models like DolphinPhi for processing and understanding queries.
- **Speech-to-Text & Text-to-Speech**: Seamlessly convert spoken language to text and vice versa.

## Setup Instructions

1. **Install Dependencies**
   - Run the following command to install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

2. **Download Necessary Models**
   - Install the required Hugging Face models by following the instructions provided by the models' documentation.

3. **Configure API Keys**
   - Ensure you have valid API keys for Hugging Face and OpenAI.
   - Input your API keys in the appropriate configuration files or environment variables.

4. **Run the Application**
   - Start the application by running:
     ```bash
     python3 main.py
     ```
   - Once the application is running, it will start listening for voice commands.

5. **Using Voice Commands**
   - To initiate a query, say **"Hey Jarvis"**, followed by your question or command.
   - The system will process your request and respond accordingly.

## Enjoy!
Explore and make the most out of voice-activated document querying.
