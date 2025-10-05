#!/usr/bin/env python
# coding: utf-8
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import List, Union

# -----------------------------
# Environment & Config
# -----------------------------
load_dotenv()

# Neo4j config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Gemini API key
GENAI_APIKEY = os.getenv("GENAI_APIKEY")

# -----------------------------
# Neo4j Driver Initialization
# -----------------------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -----------------------------
# Gemini Client Initialization
# -----------------------------
gemini_client = genai.Client(api_key=GENAI_APIKEY)

# -----------------------------
# Embedding Function
# -----------------------------
def get_gemini_embedding(
    model_name: str,
    text: Union[str, List[str]],
    output_dimension: int = 768,
    is_query: bool = True
) -> List[float]:
    """
    Generates embeddings for a given text or list of texts using Gemini API.

    Args:
        model_name (str): Name of the Gemini embedding model.
        text (str or list of str): Input text(s) to generate embeddings for.
        output_dimension (int): Dimension of the output embedding vector.
        is_query (bool): If True, sets task_type='RETRIEVAL_QUERY', else 'RETRIEVAL_DOCUMENT'.

    Returns:
        List[float] or List[List[float]]: Embedding vector(s) for the input text(s).
    """
    try:
        if isinstance(text, str):
            contents = [text]
        elif isinstance(text, list):
            contents = text
        else:
            raise TypeError("Input `text` must be a string or a list of strings.")

        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dimension
        )

        response = gemini_client.models.embed_content(
            model=model_name,
            contents=contents,
            config=config
        )

        embeddings = [e.values for e in response.embeddings]

        return embeddings[0] if isinstance(text, str) else embeddings

    except Exception as e:
        print(f"[Embedding Error] {e}")
        return [] if isinstance(text, str) else [[] for _ in text]

# -----------------------------
# Context Retrieval Function
# -----------------------------
def retrieve_context(
    query_text: str,
    score_threshold: float = 0.75,
    top_k: int = 5,
    embedding_model: str = "text-embedding-004"
) -> List[dict]:
    """
    Retrieve contextual patient cases similar to the query from Neo4j.

    Args:
        query_text (str): Query describing symptoms or patient conditions.
        score_threshold (float): Minimum cosine similarity score for results.
        top_k (int): Maximum number of top results to return.
        embedding_model (str): Gemini embedding model to use.

    Returns:
        List[dict]: List of retrieved case dictionaries.
    """
    query_embedding = get_gemini_embedding(
        model_name=embedding_model,
        text=query_text,
        output_dimension=768,
        is_query=True
    )

    cypher_query = """
    CALL db.index.vector.queryNodes('clinicalNote_embedding_index', $top_k, $query_embedding)
    YIELD node AS note, score
    WITH note, score
    WHERE score > $score_threshold
    OPTIONAL MATCH (p:Patient)-[:HAS_NOTE]->(note)
    OPTIONAL MATCH (p)-[:HAS_SYMPTOM]->(s:Symptom)
    OPTIONAL MATCH (p)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
    OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition)
    OPTIONAL MATCH (p)-[:HAS_RISK_FACTOR]->(r:RiskFactor)
    OPTIONAL MATCH (p)-[:HAS_TEST]->(t:Test)
    RETURN
        p.name AS patient_name,
        p.age AS age,
        p.gender AS gender,
        collect(DISTINCT s.symptom) AS symptoms,
        collect(DISTINCT d.diagnosis) AS diagnoses,
        collect(DISTINCT c.condition) AS conditions,
        collect(DISTINCT r.risk_factor) AS risk_factors,
        collect(DISTINCT {test_name: t.test_name, value: t.value, unit: t.unit}) AS lab_tests,
        note.note_text AS clinical_note,
        score
    ORDER BY score DESC
    LIMIT $top_k
    """

    with driver.session() as session:
        results = session.run(
            cypher_query,
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold
        )
        records = [r.data() for r in results]
    return records
# -----------------------------
# Diagnosis Generation
# -----------------------------
def generate_diagnosis(
    query_text: str,
    top_k: int = 15,
    is_user: bool = False,
    embedding_model: str = "text-embedding-004",
    llm_model: str = "gemini-2.5-flash"
) -> str:
    """
    Generate a contextualized diagnosis suggestion using retrieved graph evidence.

    Args:
        query_text (str): User query describing symptoms or condition.
        top_k (int): Number of top context cases to retrieve.
        is_user (bool): If True, hide evidence in the LLM prompt.
        embedding_model (str): Embedding model name.
        llm_model (str): Gemini LLM model name.

    Returns:
        str: LLM-generated diagnosis suggestion.
    """
    retrieved_context = retrieve_context(
        query_text=query_text,
        top_k=top_k,
        embedding_model=embedding_model
    )

    retrieved_evidence = ""
    for r in retrieved_context:
        lab_summary = ", ".join(
            [f"{t['test_name']}: {t['value']} {t.get('unit', '')}" for t in r['lab_tests'] if t['test_name']]
        ) if r['lab_tests'] else "N/A"

        retrieved_evidence += f"""
Case Profile:
- Age: {r.get('age', 'N/A')}, Gender: {r.get('gender', 'N/A')}
- Symptoms: {', '.join(r['symptoms']) if r['symptoms'] else 'N/A'}
- Diagnoses: {', '.join(r['diagnoses']) if r['diagnoses'] else 'N/A'}
- Conditions: {', '.join(r['conditions']) if r['conditions'] else 'N/A'}
- Risk Factors: {', '.join(r['risk_factors']) if r['risk_factors'] else 'N/A'}
- Lab Tests: {lab_summary}
- Clinical Note: {r['clinical_note']}
"""

    if is_user:
        prompt = f"""You are a medical reasoning assistant. The following background information contains insights from 
previous real-world medical cases similar to the patient's situation. 
Use that information to guide your reasoning, but do not mention it directly.

Patient Query: "{query_text}"

Background Medical Knowledge:
{retrieved_evidence}

Provide the most probable diagnosis or recommendation for the patient's query, ensuring your answer is medically sound and evidence-informed.
"""
    else:
        prompt = f"""You are a clinical reasoning assistant specialized in interpreting medical knowledge graphs.
You are provided with structured graph evidence from Neo4j, including symptoms, diagnoses, risk factors, and lab tests.

User Query:
{query_text}

Retrieved Graph Evidence:
{retrieved_evidence}

Instructions:
1. Analyze the evidence carefully.
2. Identify diseases consistent with the patient's symptoms, labs, and risk factors.
3. Provide a diagnosis suggestion with step-by-step reasoning, citing evidence nodes/attributes.
4. Do not invent evidence; rely only on provided context.
"""

    try:
        response = gemini_client.models.generate_content(
            model=llm_model,
            contents=prompt
        )
        return response.text

    except Exception as e:
        print(f"[Diagnosis Generation Error] {e}")
        return "Error generating diagnosis."

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":


    query = "I am 27 years old, I am having chest pain and shortness of breath"
    print(f"Generating the output the sample query: {query}")

    print("=== Diagnosis with Evidence (Debug) ===")
    result_debug = generate_diagnosis(query, is_user=False)
    print(result_debug)

    print("\n=== Diagnosis for User (Evidence Hidden) ===")
    result_user = generate_diagnosis(query, is_user=True)
    print(result_user)

    # Testing loop
    is_test_case = True
    print(f"This is test case loop")
    while is_test_case:


        user_input = input("Please enter your query (enter `exit` to exit):")

        if user_input.lower().strip() == "exit":
            print("Existing the test case")
            is_test_case = False
        else:
            print(f"Generating the output")
            print("=== Diagnosis with Evidence (Debug) ===")
            result_debug = generate_diagnosis(query, is_user=False)
            print(result_debug)
            print()

            print("\n=== Diagnosis for User (Evidence Hidden) ===")
            result_user = generate_diagnosis(query, is_user=True)
            print(result_user)

