#!/usr/bin/env python
# coding: utf-8



import os
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from typing import List, Dict
from neomodel import StructuredNode, StringProperty, IntegerProperty, FloatProperty, ArrayProperty, RelationshipTo, config
from neo4j import GraphDatabase
from dotenv import load_dotenv
from google import genai
from google.genai import types
import textwrap


# Load environment variables
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

config.DATABASE_URL = f"bolt://{NEO4J_USER}:{NEO4J_PASSWORD}@localhost:7687"

# Gemini
GENAI_APIKEY = os.getenv("GENAI_APIKEY")

# Neo4j driver for vector queries
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Gemini client
gemini_client = genai.Client(api_key=GENAI_APIKEY)


# -----------------------------
# Load Synthetic Data
# -----------------------------
data_dir = "./data/realistic/"

patients_df = pd.read_csv(f"{data_dir}patients_realistic.csv")
symptoms_df = pd.read_csv(f"{data_dir}symptoms_realistic.csv")
conditions_df = pd.read_csv(f"{data_dir}conditions_realistic.csv")
notes_df = pd.read_csv(f"{data_dir}clinical_notes_realistic.csv")
diagnoses_df = pd.read_csv(f"{data_dir}diagnoses_realistic.csv")
tests_df = pd.read_csv(f"{data_dir}lab_tests_realistic.csv")
riskfactors_df = pd.read_csv(f"{data_dir}risk_factors_realistic.csv")


# -----------------------------
# Neo4j Node Models
# -----------------------------
class Symptom(StructuredNode):
    symptom = StringProperty(unique_index=True)

class Condition(StructuredNode):
    condition = StringProperty(unique_index=True)

class Diagnosis(StructuredNode):
    diagnosis = StringProperty(unique_index=True)

class RiskFactor(StructuredNode):
    risk_factor = StringProperty(unique_index=True)

class Test(StructuredNode):
    test_name = StringProperty()
    value = StringProperty()
    unit = StringProperty(required=False)

class ClinicalNote(StructuredNode):
    note_text = StringProperty()
    embedding = ArrayProperty(FloatProperty())

class Patient(StructuredNode):
    patient_id = IntegerProperty(unique_index=True)
    name = StringProperty()
    age = IntegerProperty()
    gender = StringProperty()

    has_symptom = RelationshipTo('Symptom', 'HAS_SYMPTOM')
    has_condition = RelationshipTo('Condition', 'HAS_CONDITION')
    has_diagnosis = RelationshipTo('Diagnosis', 'HAS_DIAGNOSIS')
    has_risk_factor = RelationshipTo('RiskFactor', 'HAS_RISK_FACTOR')
    has_test = RelationshipTo('Test', 'HAS_TEST')
    has_note = RelationshipTo('ClinicalNote', 'HAS_NOTE')


# -----------------------------
# Embedding Function
# -----------------------------
def get_gemini_embedding(
    model_name: str,
    texts: List[str],
    output_dimension: int,
    is_query: bool = True
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Gemini embedding model.

    Args:
        model_name: Gemini embedding model name.
        texts: List of strings to embed.
        output_dimension: Embedding vector size.
        is_query: True for query embedding, False for document embedding.

    Returns:
        List of embeddings (list of floats) for each text.
    """
    try:
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        config_obj = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dimension
        )
        response = gemini_client.models.embed_content(
            model=model_name,
            contents=texts,
            config=config_obj
        )
        # Return embeddings as list of lists
        embeddings = [item.values for item in response.embeddings]
        return embeddings
    except Exception as e:
        print(f"[Embedding Error]: {e}")
        return [[] for _ in texts]


# -----------------------------
# Data Ingestion
# -----------------------------
def ingest_data():
    """
    Ingest synthetic patient data, clinical notes, and relationships into Neo4j.
    """
    print("Starting data ingestion...")

    # 1. Create Patients
    for _, row in tqdm(patients_df.iterrows(), total=len(patients_df), desc="Patients"):
        patient = Patient.get_or_create({
            "patient_id": int(row.patient_id),
            "name": row.name,
            "age": int(row.age),
            "gender": row.gender
        })

    # 2. Create Symptoms
    for _, row in tqdm(symptoms_df.iterrows(), total=len(symptoms_df), desc="Symptoms"):
        patient = Patient.nodes.get(patient_id=int(row.patient_id))
        symptom = Symptom.get_or_create({"symptom": row.symptom})
        patient.has_symptom.connect(symptom[0])

    # 3. Create Conditions
    for _, row in tqdm(conditions_df.iterrows(), total=len(conditions_df), desc="Conditions"):
        patient = Patient.nodes.get(patient_id=int(row.patient_id))
        condition = Condition.get_or_create({"condition": row.condition})
        patient.has_condition.connect(condition[0])

    # 4. Create Diagnoses
    for _, row in tqdm(diagnoses_df.iterrows(), total=len(diagnoses_df), desc="Diagnoses"):
        patient = Patient.nodes.get(patient_id=int(row.patient_id))
        diagnosis = Diagnosis.get_or_create({"diagnosis": row.diagnosis})
        patient.has_diagnosis.connect(diagnosis[0])

    # 5. Risk Factors
    for _, row in tqdm(riskfactors_df.iterrows(), total=len(riskfactors_df), desc="Risk Factors"):
        patient = Patient.nodes.get(patient_id=int(row.patient_id))
        risk = RiskFactor.get_or_create({"risk_factor": row.risk_factor})
        patient.has_risk_factor.connect(risk[0])

    # 6. Lab Tests
    for _, row in tqdm(tests_df.iterrows(), total=len(tests_df), desc="Lab Tests"):
        patient = Patient.nodes.get(patient_id=int(row.patient_id))
        test = Test.get_or_create({
            "test_name": row.test_name,
            "value": str(row.value),
            "unit": str(row.unit) if row.unit else None
        })
        patient.has_test.connect(test[0])

    # 7. Clinical Notes + Embeddings
    notes_list = notes_df['note_text'].tolist()
    note_embeddings = get_gemini_embedding("text-embedding-004", notes_list, 768, is_query=False)

    for i, (_, row) in enumerate(tqdm(notes_df.iterrows(), total=len(notes_df), desc="Clinical Notes")):
        patient = Patient.nodes.get(patient_id=int(row.patient_id))
        note = ClinicalNote(note_text=row.note_text, embedding=note_embeddings[i]).save()
        patient.has_note.connect(note)

    print("Data ingestion complete.")


# -----------------------------
# Retrieve Contextual Cases
# -----------------------------
def retrieve_context(query_text: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top-k patient cases similar to the query using vector search.

    Args:
        query_text: Text describing patient symptoms/query.
        top_k: Number of top similar cases to retrieve.

    Returns:
        List of dictionaries containing patient info, symptoms, diagnoses, conditions, risk factors, lab tests, and clinical note.
    """
    # Generate embedding
    query_embedding = get_gemini_embedding("text-embedding-004", [query_text], 768, is_query=True)[0]

    cypher_query = """
    CALL db.index.vector.queryNodes('clinicalNote_embedding_index', $top_k, $query_embedding)
    YIELD node AS note, score
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
        results = session.run(cypher_query, query_embedding=query_embedding, top_k=top_k)
        records = [r.data() for r in results]
    return records

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Ingest data into Neo4j
    ingest_data()

    # Create vector index in neo4j
    with driver.session() as session:
        session.run("""
        CREATE VECTOR INDEX clinicalNote_embedding_index IF NOT EXISTS
        FOR (n: ClinicalNote)
        ON (n.embedding)
        OPTIONS {
            indexConfig: {
            `vector.dimensions`:768,
            `vector.similarity_function`: 'cosine'
            }
        }
        """)
    print("Vector index created successfully!")

    is_test_case = True

    while is_test_case:
        print(f"This is test case loop, to see the related nodes for a given query")

        user_input = input("Please enter your query (enter `exit` to exit):")

        if user_input.lower().strip() == "exit":
            print("Existing the test case")
            is_test_case = False
        else:
            print(f"Retrieving similar nodes for a given query")
            # Retrieve context
            context_results = retrieve_context(user_input)
            print(f"{len(context_results)} similar documents found")
            print("Retrieved Context:")
            for r in context_results:
                print(r)
                print()
                