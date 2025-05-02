# Message Traffic Summarization with LLMs

This project demonstrates a text-to-SQL framework that enables users to query a relational database using natural language. This is as part of our Data Science capstone project at the University of Nebraska at Omaha. Our system integrates a fine-tuned T5 model with Retrieval-Augmented Generation (RAG) to convert user queries into SQL.

## Project Goals

- Allow users to ask database questions in natural English and get desired results
- Fine-tune a pre-trained T5 model on a domain-specific set of question-and-sql pairs
- Use RAG to tell model which table/columns it should be using to make the user's query
- Deploy an API on AWS on our EC2 instance demonstrate our working prototype

## How it works

- **LLM tranfer learning**: Fine-tuned [T5 model](https://huggingface.co/cssupport/t5-small-awesome-text-to-sql) from Hugging Face
- **RAG**: Uses [FAISS](https://ai.meta.com/tools/faiss/) and [SentenceTransformer](https://huggingface.co/sentence-transformers) give model relevant schema info
- **Deployment**: API build with flask and hosted on AWS EC2
- **Example Dataset**: Cars database from [Spider 1.0](https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset)

## Technologies

- Python, SQLite
- Hugging Face Transformers and SentenceTransformers
- FAISS (Facebook AI Similarity Search)
- Flask (for deployment)
- AWS EC2 (for hosting our API)

## Usage

To try out our model

- Open terminal e.g. git bash
- Use [curl command](https://www.geeksforgeeks.org/curl-command-in-linux-with-examples/) to send a POST request
- Here is an example curl request send a POST request to the deployed API endpoint with a natural language query. The response will contain the generated SQL and retrieved values

```
curl -X POST http://my-aws-ip-address:5000/query \
  -H "Content-Type: application/json" \
  -d '{"nl_query": "How many cars are there?"}' | jq
```

---
