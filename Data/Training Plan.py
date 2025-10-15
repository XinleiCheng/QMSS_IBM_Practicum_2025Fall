#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 17:06:13 2025

@author: adonischeng
"""
from docx import Document
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import re

doc = Document("Training Plan.docx")


full_text = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

sections = {}
current_section = None
record = False  
for line in full_text:
    # start only after the real "1.0 Introduction" appears
    if re.match(r"^1\.0\s+Introduction$", line):
        record = True

    if not record:
        continue  # skip preface, version history, table of contents

    # detect section headings
    if re.match(r"^\d+(\.\d+)*\s+", line):
        current_section = line
        sections[current_section] = []
    elif current_section:
        sections[current_section].append(line)

#print(full_text[:10])  

cleaned_data = []

for section, content in sections.items():
    text = " ".join(content)
    text = re.sub(r"<.*?>", "", text)  # remove placeholders
    text = re.sub(r"\s+", " ", text)  # normalize spaces

#Summary
    sentences = sent_tokenize(text)
    summary = " ".join(sentences[:])

    cleaned_data.append({
        "phase": "Development",
        "deliverable": "Training Plan",
        "section_title": section,
        "related_eplc_reference": "EPLC Framework – Development Phase",
        "source": "CDC UP Training Plan Template",
        "summary": summary
        
    })

df = pd.DataFrame(cleaned_data)
df.to_csv("training_plan_cleaned.csv", index=False)
