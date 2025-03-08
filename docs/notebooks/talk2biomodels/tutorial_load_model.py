#!/usr/bin/env python
# coding: utf-8

# Working with SBML models using basico

# In[1]:


import basico


# In[2]:


# load SBML model
model = basico.load_model("./Dwivedi_Model537_annotated.xml")


# In[3]:


# get model species
species = basico.get_species()
species.head(5)


# In[4]:


# some basic concepts:
"""
SBML definition is based on species, compartments, reactions, and functions.
"""

# other important functions

basico.get_reactions()             # Get information about reactions
basico.get_compartments()          # Get information about compartments
basico.get_species()               # Get information about species (metabolites)
basico.get_functions()             # Get information about functions
basico.get_model_name()            # Get name of the model


# MIRIAM stands for "Minimum Information Required in the Annotation of Models." It's a standardized framework for annotating computational models in biology, particularly for systems biology models like those used with COPASI and basico.
# Here's what MIRIAM provides:
# 
# Structured Identifiers: MIRIAM provides a way to link model components (species, reactions, compartments) to external databases using standardized URIs. For example, linking a metabolite to its ChEBI ID or a reaction to an Enzyme Commission number.
# Controlled Vocabulary: It establishes a set of defined relationships between model components and external resources using qualifiers like "is," "isVersionOf," "isDescribedBy," etc.
# Registry of Resources: MIRIAM maintains a registry of recognized data resources that can be referenced in model annotations.
# 
# In basico, MIRIAM annotations are implemented through functions like `get_miriam_annotation()` and `set_miriam_annotation()`.

# In[5]:


# get annotations
basico.get_miriam_annotation(name="IL6")

