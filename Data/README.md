## Data Preparation
This folder contains all datasets, intermediate outputs, and preprocessing steps used in the QMSS–IBM Practicum Project.

Our project focuses on supporting the HHS Enterprise Performance Life Cycle (EPLC) process using generative AI and Retrieval-Augmented Generation (RAG), by building a structured knowledge base from publicly available documents and templates. 

## 1. Overview
Data Source: 
- EPLC Templates: https://web.archive.org/web/20240609100355/https:/www2.cdc.gov/cdcup/library/templates/default.htm#sthash.UcHHkg85.cHHkg856.dpbs
- EPLC Policy: https://www.hhs.gov/web/governance/digital-strategy/it-policy-archive/policy-for-information-technology-enterprise-performance.html

Raw Data Format: docx, xls

Final Output Format: JSON

Templates Size: 17 documents

## 2. Folder Contents
### 📥 Extractiond
| File | Description |
|------|-------------|
| `xxx.docx` | Original Templates. |
| `xxx.json` | Data after chunked and flattened. |

### 🧹 Cleaning
| File | Description |
|------|-------------|
| `xxx_embedding.json` | Texts converted into embeddings. |
| `xxx Phase DB.py` | Codes that turn embeddings into Vector DB. |

### 📦 Final Outputs
| File | Description |
|------|-------------|
| `chroma_db_xxx` | Final Vector DB that are ready to perform generative AI. |


## 3. Q&A Retrieval Dataset

The Q&A source of truth is the cleaned EPLC framework JSON under
`EPLC Framework/EPLC Cleaned Data` plus
`HHS EPLC Website/HHS EPLC Website.json`. The older `*embeddings*.json` files
and databases are retained as legacy experiment artifacts.

The current Q&A pipeline:

1. Preserves paragraph boundaries that represent roles, activities, and
   deliverables.
2. Splits only oversized paragraphs with a sentence-aware word window.
3. Adds a derived section overview for multi-part sections.
4. Stores document title, section number, section title, chunk type, source
   file, and schema version with every chunk.
5. Writes to a new versioned Chroma directory and refuses to overwrite an
   existing index.

```bash
python "Coding/Q&A/qna_data_pipeline.py" prepare
python "Coding/Q&A/qna_data_pipeline.py" build
python "Coding/Q&A/evaluate_retrieval.py" \
  --index "Data/Vector DataBase/qna_v4_bge_base_en_v1_5" \
  --model "BAAI/bge-base-en-v1.5"
```

## 4. Embedding Model Decision

The current Q&A index uses `BAAI/bge-base-en-v1.5` with 768-dimensional,
normalized embeddings and cosine distance. It was selected by comparing the
same 28 positive retrieval questions across the legacy index, BGE-large
candidate indexes, and BGE-base candidate indexes. Six out-of-domain questions
are used to calibrate the refusal threshold.

The full reports are stored under `evaluation/results`. The v4 BGE-base index
reaches Recall@5 of 1.0 on the checked-in positive set. A minimum cosine
similarity of 0.61 rejects all six checked-in negative questions while
retaining 96.43% of positive questions at the threshold.
