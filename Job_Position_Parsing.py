import random
import json
import pandas as pd
import numpy as np
import re
import string
import os
import spacy
import re
from tqdm import *
import string


nlp = spacy.load('en_core_web_lg')


# See position type in txt file

pos=positions

about=pos[["description"]].rename(columns={"description":"text"})
req=pos[["responsibilities"]].rename(columns={"responsibilities":"text"})

about["label"]="about"
req["label"]="responsibilities"

# Some data modifications
alll=about.join(req,rsuffix="_1").dropna().reset_index(drop=True)
about_fin=alll[["text","label"]]
req_fin=alll[["text_1","label_1"]]

# Setting starting and endig indexes of About and Requirements
az=[]
for i in about_fin.text.index:
    word=about_fin.text[i].split()[-1]
    az.append([0,int(about_fin.text[i].index(word)+len(word))])

bz=[]
for i in about_fin.text.index:
    word=req_fin.text_1[i].split()[-1]
    bz.append([1,int(req_fin.text_1[i].index(word)+len(word))])

fz=[]
for i in range(len(az)):
    fz.append([(az[i][1]+bz[i][0]),(az[i][1]+bz[i][1])])

pre=pd.DataFrame(about_fin.text+" "+req_fin.text_1).reset_index().join(pd.DataFrame(pd.Series(az),columns=["about_text"]).join(pd.DataFrame(pd.Series(fz),columns=["req_text"])))
final=pre.rename(columns={0:"text"})

entities=[]
for i in final.about_text:
    i.append("about")
    entities.append(i)

for i in final.req_text:
    i.append("req")
    entities.append(i)

#def train_spacy():

nlp = spacy.blank('en')  # create blank Language class
# create the built-in pipeline components and add them to the pipeline
# nlp.create_pipe works for built-ins that are registered with spaCy
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)

# add labels
for i,j in zip(final.about_text, final.req_text):
    ner.add_label(i[2])
    ner.add_label(j[2])

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(20):
        print("Statring iteration " + str(itn))
        final_1=final.sample(200)
        losses = {}
        bz=[]
        for i,j in zip(final_1.about_text, final_1.req_text):
            az={"entities":[(i[0],i[1],i[2]),(j[0],j[1],j[2])]}
            bz.append(az)
        for text, annotations in zip(final_1.text,bz):
            nlp.update(
                [text],  # batch of texts
                [annotations],  # batch of annotations
                drop=0.2,  # dropout - make it harder to memorise data
                sgd=optimizer,  # callable to update weights
                losses=losses)
        print(losses)
nlp.to_disk("./modelik_our.vol2")


text="""To breathe some fresh air into our dynamically growing team,
we are looking for BRIGHT and COOL team members to join our Communications
department for an Internship, which will last for 3 month. Interns may
receive a job offer based on their performance during their internship.
If you are a current undergraduate and graduate student pursuing degrees
in a related field (Communications, Journalism etc), this opportunity if for YOU!
Involvement in various Communications projects
Activities and assignments related to media monitoring processes
Assistance in creation of communications materials, media texts, and articles.
Generation of ideas and coordination works"""


#test the model and evaluate it

nlp = spacy.load("modelik_our.vol2")

doc_text=nlp(txt)
for doc in doc_text.ents:
    print("\n",doc.label_,"\n",doc.text,"neeext")

# Tipical output

## about
# To breathe some fresh air into our dynamically growing team,
# we are looking for BRIGHT and COOL team members to join our Communications
# department for an Internship, which will last for 3 month. Interns may
# receive a job offer based on their performance during their internship.
# If you are a current undergraduate and graduate student pursuing degrees
# in a related field (Communications, Journalism etc), this opportunity if for YOU!
#
## responsibility
# Involvement in various Communications projects
# Activities and assignments related to media monitoring processes
# Assistance in creation of communications materials, media texts, and articles.
# Generation of ideas and coordination works
#
