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

doc = Document("Training Plan.docx")


full_text = []
for para in doc.paragraphs:
    if para.text.strip():  
        full_text.append(para.text.strip())


#print(full_text[:10])  

import re #(used for detecting patterns like "1.0", "2.0" etc.)

sections = {}
current_section = None

for line in full_text:
    if re.match(r"^\d+\.\d*", line):  # detects lines like 1.0, 2.0, etc.
        current_section = line
        sections[current_section] = []
    elif current_section:
        sections[current_section].append(line)

cleaned_data = []

for section, content in sections.items():
    text = " ".join(content)
    text = re.sub(r"<.*?>|\[.*?\]", "", text)  # remove placeholders
    text = re.sub(r"\s+", " ", text)  # normalize spaces

    # Take first 2–3 sentences as summary
    sentences = sent_tokenize(text)
    summary = " ".join(sentences[:2])

    cleaned_data.append({
        "phase": "Development",
        "deliverable": "Training Plan",
        "section_title": section,
        "summary": summary,
        "related_eplc_reference": "EPLC Framework – Development Phase",
        "source": "CDC UP Training Plan Template"
    })

