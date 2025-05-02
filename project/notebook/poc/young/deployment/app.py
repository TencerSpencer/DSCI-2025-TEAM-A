# ### Imports

import re
import sqlite3

import faiss
import numpy as np
import pandas as pd
import spacy
import torch
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sentence_transformers import SentenceTransformer
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ### Load necessary data and models


# db_location = "../eda/code/cyoung_eda.db"  # soccer dataset
db_location = "data/car_1.sqlite"
schema_location = "data/car_metadata.xlsx"


def get_column_metadata():
    return pd.read_excel(schema_location, sheet_name="schema_metadata")[
        "full_column_metadata"
    ].tolist()


def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_model_and_tokenizer(use_saved_model=True):
    model_name = "cssupport/t5-small-awesome-text-to-sql"
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if use_saved_model:
        saved_model_path = "model_checkpoints/fine-tuned-model"
        model = T5ForConditionalGeneration.from_pretrained(saved_model_path).to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    return tokenizer, model


# ### Vector embedding and similarity search


# python -m spacy download en_core_web_sm


def find_optimal_k(nl_query):
    min_k_value = 1
    doc = spacy.load("en_core_web_sm")(nl_query)
    keywords = []
    for token in doc:
        if token.pos_ in ["NOUN", "ADJ"] and token.text.lower() not in STOP_WORDS:
            keywords.append(token)
    optimal_k = max(min_k_value, len(keywords))
    print(f"Keywords in natural-language query: {keywords}")
    print(f"Optimal k is {optimal_k}")
    return max(min_k_value, optimal_k)


def convert_text_to_vector_embedding(text, embedding_model):
    if isinstance(text, str):
        text = [text]
    return embedding_model.encode(text)


def search_similar_columns(
    nl_query_embedding, column_embeddings, raw_column_text, num_of_search_results=None
):
    # build the index
    vector_size = column_embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_size)
    index.add(np.array(column_embeddings))

    # perform search
    D, I = index.search(np.array(nl_query_embedding), k=num_of_search_results)

    # convert results from embeddings back to natural language
    return [raw_column_text[i] for i in I[0]]


# ### Convert search results to create table statements with only those columns


# returns create statement for given table
def get_table_create_statement(table_name):
    conn = sqlite3.connect(db_location)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    )
    statement = cursor.fetchone()[0]
    conn.close()
    return statement


# returns create table statements with only the columns that are relevant
def filter_table_create_statement(relevant_columns_info):
    table_columns_map = {}
    for relevant_column_info in relevant_columns_info:
        table = relevant_column_info.split(",")[0].split(":")[1].strip()
        column = relevant_column_info.split(",")[1].split(":")[1].strip()
        if table not in table_columns_map:
            table_columns_map[table] = []
        table_columns_map[table].append(column)
    print(table_columns_map)
    filtered_table_create_statements = []

    for table_name, relevant_columns in table_columns_map.items():
        raw_sql = get_table_create_statement(table_name)

        filtered_lines = []
        for line in raw_sql.split("\n"):
            line_segments = [
                s.strip().replace('"', "").replace(",", "") for s in line.split('" ')
            ]
            column_name = line_segments[0] if len(line_segments) > 0 else None
            column_details = line_segments[1] if len(line_segments) > 1 else None
            is_line_without_columns = (
                line_segments[0].upper().startswith("CREATE") or line_segments[0] == ")"
            )

            if is_line_without_columns:
                continue

            if "FOREIGN" in line:
                if any(col in line for col in relevant_columns):
                    filtered_lines.append(line.strip())

            if column_name.strip() in relevant_columns:
                filtered_lines.append(f"{column_name} {column_details}")

        table_def = (
            f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(filtered_lines) + "\n)"
        )
        filtered_table_create_statements.append(table_def)

    return "\n\n".join(filtered_table_create_statements)


def build_prompt(nl_query, schema_context):
    return f"tables:\n{schema_context}\nquery for: {nl_query}"


def convert_nlq_to_sql(prompt, tokenizer, model):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    with torch.no_grad():
        output = model.generate(**inputs, max_length=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def execute_sql(query):
    conn = sqlite3.connect(db_location)
    cur = conn.cursor()
    try:
        cur.execute(query)
        results = cur.fetchall()
        return results
    except Exception as e:
        return f"SQL Error: {e}"
    finally:
        conn.close()


def text_to_sql(nl_query: str, num_of_search_results=None):
    if num_of_search_results is None:
        num_of_search_results = find_optimal_k(nl_query)
    # load data and models
    column_texts = get_column_metadata()
    embedder = get_embedding_model()
    tokenizer, model = get_model_and_tokenizer()

    # convert to vector embeddings
    column_vectors = convert_text_to_vector_embedding(column_texts, embedder)
    query_vector = convert_text_to_vector_embedding(nl_query, embedder)

    # find relevant colums
    similar_columns = search_similar_columns(
        query_vector, column_vectors, column_texts, num_of_search_results
    )
    # create input prompt
    schema_text = filter_table_create_statement(similar_columns)
    prompt = build_prompt(nl_query, schema_text)
    print(f"\nInput prompt:\n{prompt}")

    # convert to sql
    sql = convert_nlq_to_sql(prompt, tokenizer, model)
    return sql


def print_results(natural_language_query, generated_sql, retrevied_value):
    padding = 30
    print("\n")
    print(f"{'Natural language query:':<{padding}}{natural_language_query}")
    print(f"{'Generated SQL query:':<{padding}}{generated_sql}")
    print(f"{'Query results:':<{padding}}{retrevied_value}")


app = Flask(__name__)
limiter = Limiter(key_func=lambda: "global", default_limits=["10 per minute"])
limiter.init_app(app)


@app.route("/query", methods=["POST"])
@limiter.limit("5 per minute")
def query():
    natural_language_query = request.json.get("nl_query", "")

    generated_sql = text_to_sql(natural_language_query)
    query_results = execute_sql(generated_sql)
    print_results(natural_language_query, generated_sql, query_results)

    return jsonify(
        {
            "nl_query": natural_language_query,
            "generated_sql": generated_sql,
            "query_results": query_results,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
