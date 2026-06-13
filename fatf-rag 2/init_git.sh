#!/usr/bin/env bash
# Seed a clean, incremental commit history, then push to your remote.
# Run once from the repo root on your own machine:  bash init_git.sh
set -e

rm -rf .git
git init -q
git add .gitignore requirements.txt .env.example src/__init__.py src/config.py
git commit -q -m "scaffold: project config, requirements, env template"

git add src/ingest.py
git commit -q -m "ingest: page-tagged chunking of the FATF PDF"

git add src/embeddings.py src/vectorstore.py
git commit -q -m "retrieval: local embeddings + FAISS vector store"

git add src/retriever.py
git commit -q -m "retrieval: dense/BM25/hybrid retriever with score fusion"

git add src/llm.py src/pipeline.py scripts/__init__.py scripts/build_index.py
git commit -q -m "pipeline: pluggable LLM + end-to-end orchestration + index builder"

git add cli.py app/streamlit_app.py
git commit -q -m "entrypoints: CLI and Streamlit web app"

git add eval/ tests/ notebooks/
git commit -q -m "eval: test set, metrics + ablation, offline tests, notebook"

git add README.md data/ENG_REC.pdf init_git.sh
git commit -q -m "docs: README with design rationale, eval interpretation, AI-usage disclosure"

echo "Done. Now add your remote and push, e.g.:"
echo "  git remote add origin git@github.com:<you>/fatf-rag.git"
echo "  git branch -M main && git push -u origin main"
git log --oneline
