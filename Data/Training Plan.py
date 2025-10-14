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


