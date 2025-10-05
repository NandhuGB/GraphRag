# GraphRAG for Patient Diagnosis

This project implements a Graph-based Retrieval Augmented Generation (GraphRAG) pipeline using **Neo4j** as a knowledge graph store and the **Google Gemini API** for embedding and large language model (LLM) reasoning. The goal is to assist in patient diagnosis by retrieving relevant historical patient case data from the knowledge graph based on a patient's symptoms and factors, and then using an LLM to generate a contextualized diagnosis suggestion.

## 🚀 Key Components

  * **Knowledge Graph (Neo4j):** Stores structured medical data, including patients, symptoms, diagnoses, conditions, risk factors, lab tests, and clinical notes.
  * **Vector Indexing:** Clinical notes are embedded and indexed in Neo4j using the `ClinicalNote.embedding` property for fast vector similarity search.
  * **Retrieval (RAG):** A user query is converted into an embedding, which is used to retrieve similar historical cases (clinical notes and their connected entities) from the graph.
  * **Generation (LLM):** The retrieved graph entities and relationships are formatted as evidence and passed to the Gemini LLM to generate a reasoned diagnosis suggestion.

## 💾 Prerequisites

1.  **Neo4j Instance:** A running Neo4j database (local or AuraDB).
2.  **Python:** Python 3.9+
3.  **API Keys:**
      * **Neo4j:** URI, Username, and Password.
      * **Google AI (Gemini):** API Key for embedding and generation models.

## ⚙️ Setup Instructions

### 1\. Project Structure

Ensure you have the following file structure (the data files are assumed to be in the `./data/realistic/` directory, which were used in the provided solution):

```
GraphRAG-Diagnosis/
├── data/
│   └── realistic/
│       ├── conditions_realistic.csv
│       ├── clinical_notes_realistic.csv
│       ├── diagnoses_realistic.csv
│       ├── patients_realistic.csv
│       ├── risk_factors_realistic.csv
│       ├── symptoms_realistic.csv
│       └── lab_tests_realistic.csv
├── .env
├── main_rag_pipeline.py  (The provided second script for RAG)
└── ingestion_script.py (The provided first script for ingestion)
```

### 2\. Environment Variables

Create a file named `.env` in the root directory and populate it with your credentials:

```bash
# .env file content
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your_neo4j_password"

GENAI_APIKEY="your_google_gemini_api_key"
```

### 3\. Create conda environment

Conda environment can be created by running following command:

```bash
  conda env create -f conda_environment.yml
```

### 4\. Data Ingestion and Indexing

Run the ingestion script (which uses `neomodel` and the `get_gemini_embedding` function) to populate the Neo4j graph and create the vector index.

> **Note:** The provided solution splits ingestion and RAG into two parts. The first part handles ingestion and indexing. Ensure the database is cleared before re-running if necessary.

1.  **Run Ingestion (First part of the provided solution):**

    ```bash
    python ingestion_script.py
    ```

    This script performs the following:

      * Creates all the patient, factor, and clinical note nodes.
      * Generates embeddings for all `ClinicalNote` nodes using `text-embedding-004`.
      * Establishes all the relationships (e.g., `Patient-[:HAS_SYMPTOM]->Symptom`).
      * Creates the necessary **Vector Index** in Neo4j.

## 📋 Knowledge Graph Schema

The schema is designed to connect multiple contributing factors to a central `Patient` node, facilitating comprehensive diagnostic retrieval.

### Node Types (Labels)

| Node Label | Properties | Description |
| :--- | :--- | :--- |
| **`Patient`** | `patient_id` (INT), `name` (STRING), `age` (INT), `gender` (STRING) | The central entity representing a patient. |
| **`Symptom`** | `symptom` (STRING) | A reported patient symptom (e.g., 'cough', 'fever'). |
| **`Condition`** | `condition` (STRING) | A pre-existing or chronic medical condition (e.g., 'Diabetes', 'Hypertension'). |
| **`Diagnosis`**| `diagnosis` (STRING) | The confirmed or suspected final diagnosis. |
| **`RiskFactor`**| `risk_factor` (STRING) | Lifestyle or family history factors (e.g., 'smoking', 'family history of heart disease'). |
| **`Test`** | `test_name` (STRING), `value` (STRING), `unit` (STRING) | Results of lab or imaging tests. |
| **`ClinicalNote`**| `note_text` (STRING), `embedding` (LIST\<FLOAT\>) | Unstructured clinical text; embedded for RAG. |

### Relationship Types

| Relationship Type | Connecting Nodes | Description |
| :--- | :--- | :--- |
| **`:HAS_SYMPTOM`** | `(Patient) -> (Symptom)` | Patient exhibits a symptom. |
| **`:HAS_CONDITION`**| `(Patient) -> (Condition)`| Patient has a prior medical condition. |
| **`:HAS_DIAGNOSIS`**| `(Patient) -> (Diagnosis)`| Patient received this diagnosis. |
| **`:HAS_RISK_FACTOR`**| `(Patient) -> (RiskFactor)`| Patient has a specific risk factor. |
| **`:HAS_TEST`** | `(Patient) -> (Test)` | Patient has a specific test result. |
| **`:HAS_NOTE`** | `(Patient) -> (ClinicalNote)`| Clinical documentation for the patient. |

## 🧪 GraphRAG Pipeline Execution

The `main_rag_pipeline.py` script implements the full RAG workflow.

### 1\. Embedding Generation

The user's input query is converted into a query embedding using the `get_gemini_embedding` function and the **`text-embedding-004`** model.

### 2\. Neo4j Vector Search and Retrieval

The query embedding is used in a Cypher query with the `db.index.vector.queryNodes` procedure to find the top $k$ most similar `ClinicalNote` nodes.

**Cypher Query (Simplified):**

```cypher
CALL db.index.vector.queryNodes('clinicalNote_embedding_index', $top_k, $query_embedding)
YIELD node AS note, score
// ... filter by score > $score_threshold ...
OPTIONAL MATCH (p:Patient)-[:HAS_NOTE]->(note)
OPTIONAL MATCH (p)-[:HAS_SYMPTOM]->(s:Symptom)
// ... follow all relationships to gather the complete context
RETURN p.*, collect(s.symptom), collect(d.diagnosis), note.note_text, score
```

This retrieves the **full context** surrounding the similar clinical notes: the patient's demographics, symptoms, risk factors, conditions, and past diagnoses.

### 3\. LLM Reasoning for Diagnosis

The retrieved structured context is formatted into a **`Background Medical Knowledge`** section. This context, along with the original `User Query`, is passed to the **`gemini-2.5-flash`** LLM.

The LLM is prompted to act as a **medical reasoning assistant** to generate a probable diagnosis and a step-by-step reasoning based *only* on the provided evidence.

### 4\. Running the Pipeline

Run the main script for interactive testing:

```bash
python main_rag_pipeline.py
```

The script will first demonstrate the diagnosis generation for a sample query (`"I am 27 years old, I am having chest pain and shortness of breath"`) in both debug (evidence-cited) and user (evidence-hidden) modes. It then enters an interactive loop.

**Sample Interactive Session:**

```
This is test case loop
Please enter your query (enter `exit` to exit): **45-year-old female, long history of smoking, presenting with persistent cough and fatigue.**

Generating the output
=== Diagnosis with Evidence (Debug) ===
// ... LLM output with explicit citation of retrieved cases/symptoms ...

=== Diagnosis for User (Evidence Hidden) ===
// ... LLM output with the final diagnosis suggestion and reasoning ...

Please enter your query (enter `exit` to exit): exit
Existing the test case
```

-----
