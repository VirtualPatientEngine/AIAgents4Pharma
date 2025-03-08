#!/usr/bin/env python
# coding: utf-8

# # From SBML to BKGs
# 
# **Objectives**: 
# 
# Annotate a SBML

# In[ ]:


# Installations

get_ipython().system('pip -q install copasi_basico==0.78')


# In[ ]:


import basico


# In[ ]:


model = basico.load_model("./Dwivedi_Model537_empty.xml")
species = basico.get_species()
species


# In[ ]:


# The empty model does not have annotation for species
basico.get_miriam_annotation(name="IL6")


# In[ ]:


# However, we do have miriam annorations for the model itself
basico.get_miriam_annotation()


# In[ ]:


# The above information already tells use we can consider a SBML node type to a BKG with 
# This node already is connected to a Disease, Gene, Biological Process, and Reactom


# In[ ]:


# using AMAS-sb
# !pip -q install AMAS-sb


# In[ ]:


# !recommend_species Dwivedi_Model537_empty.xml


# ## contextualizing the species based on the paper

# In[ ]:


species.display_name.to_list()


# In[ ]:


# used a good quality text extractor and the species name in a reasoning model to get the following map
# from display names to name and description


# In[1]:


species_dict = {
    'IL6{serum}': {
        'name': 'Interleukin-6 in serum',
        'description': 'Interleukin-6 cytokine in blood circulation; elevated in Crohn\'s disease'
    },
    'IL6{liver}': {
        'name': 'Interleukin-6 in liver',
        'description': 'Interleukin-6 in liver compartment'
    },
    'IL6{gut}': {
        'name': 'Interleukin-6 in gut',
        'description': 'Interleukin-6 in gut compartment; significantly increased in intestinal mucosa in Crohn\'s'
    },
    'R': {
        'name': 'Membrane-bound IL-6 receptor',
        'description': 'Membrane-bound IL-6 receptor (IL-6Rα); binds IL-6 for classical signaling'
    },
    'sR{serum}': {
        'name': 'Soluble IL-6 receptor in serum',
        'description': 'Soluble IL-6 receptor in blood; enables trans-signaling'
    },
    'sR{liver}': {
        'name': 'Soluble IL-6 receptor in liver',
        'description': 'Soluble IL-6 receptor in liver compartment'
    },
    'sR{gut}': {
        'name': 'Soluble IL-6 receptor in gut',
        'description': 'Soluble IL-6 receptor in gut compartment'
    },
    'gp130{liver}': {
        'name': 'Glycoprotein 130 in liver',
        'description': 'Membrane-bound glycoprotein 130 co-receptor in liver cells'
    },
    'gp130{gut}': {
        'name': 'Glycoprotein 130 in gut',
        'description': 'Membrane-bound glycoprotein 130 co-receptor in gut cells'
    },
    'sgp130{serum}': {
        'name': 'Soluble glycoprotein 130 in serum',
        'description': 'Soluble glycoprotein 130 in blood; natural inhibitor of IL-6 trans-signaling'
    },
    'sgp130{liver}': {
        'name': 'Soluble glycoprotein 130 in liver',
        'description': 'Soluble glycoprotein 130 in liver compartment'
    },
    'sgp130{gut}': {
        'name': 'Soluble glycoprotein 130 in gut',
        'description': 'Soluble glycoprotein 130 in gut compartment'
    },
    'R_IL6': {
        'name': 'IL-6/IL-6Rα complex',
        'description': 'Complex of membrane-bound IL-6 receptor with IL-6'
    },
    'sR_IL6{serum}': {
        'name': 'IL-6/sIL-6Rα complex in serum',
        'description': 'Complex of soluble IL-6 receptor with IL-6 in blood'
    },
    'sR_IL6{liver}': {
        'name': 'IL-6/sIL-6Rα complex in liver',
        'description': 'Complex of soluble IL-6 receptor with IL-6 in liver'
    },
    'sR_IL6{gut}': {
        'name': 'IL-6/sIL-6Rα complex in gut',
        'description': 'Complex of soluble IL-6 receptor with IL-6 in gut'
    },
    'R_IL6_gp130{liver}': {
        'name': 'IL-6/IL-6Rα/gp130 complex in liver',
        'description': 'Tripartite complex of IL-6, IL-6Rα, and gp130 in liver'
    },
    'R_IL6_gp130{gut}': {
        'name': 'IL-6/IL-6Rα/gp130 complex in gut',
        'description': 'Tripartite complex of IL-6, IL-6Rα, and gp130 in gut'
    },
    'sR_IL6_sgp130{serum}': {
        'name': 'IL-6/sIL-6Rα/sgp130 complex in serum',
        'description': 'Inhibitory complex of IL-6, soluble IL-6 receptor, and sgp130 in blood'
    },
    'sR_IL6_sgp130{liver}': {
        'name': 'IL-6/sIL-6Rα/sgp130 complex in liver',
        'description': 'Inhibitory complex in liver compartment'
    },
    'sR_IL6_sgp130{gut}': {
        'name': 'IL-6/sIL-6Rα/sgp130 complex in gut',
        'description': 'Inhibitory complex in gut compartment'
    },
    'Ractive{liver}': {
        'name': 'Activated IL-6 receptor complex in liver',
        'description': 'Activated IL-6 receptor complex in liver cells'
    },
    'Ractive{gut}': {
        'name': 'Activated IL-6 receptor complex in gut',
        'description': 'Activated IL-6 receptor complex in gut cells'
    },
    'STAT3{liver}': {
        'name': 'STAT3 in liver',
        'description': 'Signal transducer and activator of transcription 3 in liver cells'
    },
    'STAT3{gut}': {
        'name': 'STAT3 in gut',
        'description': 'STAT3 in gut cells'
    },
    'pSTAT3{liver}': {
        'name': 'Phosphorylated STAT3 in liver',
        'description': 'Phosphorylated STAT3 in liver cells; indicates active IL-6 signaling'
    },
    'pSTAT3{gut}': {
        'name': 'Phosphorylated STAT3 in gut',
        'description': 'Phosphorylated STAT3 in gut cells; elevated in Crohn\'s disease'
    },
    'CRP{liver}': {
        'name': 'C-reactive protein in liver',
        'description': 'C-reactive protein produced within liver cells'
    },
    'CRP{serum}': {
        'name': 'C-reactive protein in serum',
        'description': 'C-reactive protein in blood; key inflammatory biomarker in Crohn\'s'
    },
    'CRPExtracellular': {
        'name': 'Extracellular C-reactive protein',
        'description': 'Secreted CRP not yet in circulation'
    },
    'geneProduct': {
        'name': 'STAT3-induced gene product',
        'description': 'Generic product of genes activated by pSTAT3 signaling'
    },
    'Ab{serum}': {
        'name': 'Therapeutic antibody in serum',
        'description': 'Therapeutic antibody (anti-IL6 or anti-IL6Rα) in blood'
    },
    'Ab{liver}': {
        'name': 'Therapeutic antibody in liver',
        'description': 'Therapeutic antibody in liver compartment'
    },
    'Ab{gut}': {
        'name': 'Therapeutic antibody in gut',
        'description': 'Therapeutic antibody in gut compartment'
    },
    'Ab{peripheral}': {
        'name': 'Therapeutic antibody in peripheral tissues',
        'description': 'Therapeutic antibody in peripheral tissues'
    },
    'Ab_R': {
        'name': 'Antibody-receptor complex',
        'description': 'Antibody bound to membrane-bound IL-6 receptor'
    },
    'Ab_sR{serum}': {
        'name': 'Antibody-sIL-6Rα complex in serum',
        'description': 'Antibody bound to soluble IL-6 receptor in blood'
    },
    'Ab_sR{liver}': {
        'name': 'Antibody-sIL-6Rα complex in liver',
        'description': 'Antibody bound to soluble IL-6 receptor in liver'
    },
    'Ab_sR{gut}': {
        'name': 'Antibody-sIL-6Rα complex in gut',
        'description': 'Antibody bound to soluble IL-6 receptor in gut'
    },
    'Ab_sR_IL6{serum}': {
        'name': 'Antibody-IL-6/sIL-6Rα complex in serum',
        'description': 'Antibody bound to IL-6/sIL-6Rα complex in blood'
    },
    'Ab_sR_IL6{liver}': {
        'name': 'Antibody-IL-6/sIL-6Rα complex in liver',
        'description': 'Antibody bound to IL-6/sIL-6Rα complex in liver'
    },
    'Ab_sR_IL6{gut}': {
        'name': 'Antibody-IL-6/sIL-6Rα complex in gut',
        'description': 'Antibody bound to IL-6/sIL-6Rα complex in gut'
    },
    'CRP Suppression (%)': {
        'name': 'CRP suppression percentage',
        'description': 'Percentage reduction in CRP levels after treatment'
    },
    'CRP (% of baseline)': {
        'name': 'CRP percentage of baseline',
        'description': 'CRP levels as percentage of pre-treatment baseline'
    }
}


# In[3]:


import json
with open('species_dict.json', 'w') as f:
    json.dump(species_dict, f)


# ## Approach 1: Use Bio-Ontology API for entity recoginition
# We use the ontolgoies of PrimeKG for search

# In[14]:


# download the nodes.tab in csv format file locally from
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
import pandas as pd
df = pd.read_csv('./nodes.csv')
df.node_source.drop_duplicates().tolist()


# **Note:** These naming is a bit different on bioongotlogy

# In[6]:


PRIMEKG_ONTOLOGIES = [
        'NCBITAXON',     # NCBI Taxonomy
        'DRON',          # Drug Ontology (alternative to DrugBank)
        'HP',            # Human Phenotype Ontology
        'MONDO',         # Monarch Disease Ontology
        'GO-PLUS',       # Gene Ontology Plus
        'MEDDRA',        # Medical Dictionary for Regulatory Activities (alternative to CTD)
        'RXNORM',        # RxNorm
        'UBERON'         # Uber Anatomy Ontology
    ]


# In[ ]:


primekg_to_bioontology_map = {
       "NCBI": "NCBITAXON",
       "DrugBank": "DRON",
       "HPO": "HP",
       "MONDO": "MONDO",
       "MONDO_grouped": "MONDO",
       "GO": "GO-PLUS",
       "CTD": "MEDDRA",
       "REACTOME": "RXNORM",  # Note: not exact equivalent
       "UBERON": "UBERON"
   }


# In[4]:


from bioontology_api import enrich_species_dict_with_ontologies


# In[5]:


import getpass

bioontology_api_key = getpass.getpass(prompt="Enter your API key: ")


# In[8]:


species_dict_annotated = enrich_species_dict_with_ontologies(species_dict, bioontology_api_key, PRIMEKG_ONTOLOGIES)


# In[21]:


species_dict_annotated['IL6{serum}']


# In[9]:


with open('species_dict_annotated.json', 'w') as f:
    json.dump(species_dict_annotated, f)


# ### Connecting to PrimeKG
# Since this is a direct use of the source ontologies of PrimeKG, we can match the ids to that of the PrimeKG

# In[ ]:





# ## Approach 2: using UMLS codes
# we can use scispacy to find umls codes 
# 
# ```
# !pip -q install spacy scispacy
# !pip -q install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
# ```
# 
# But it the installation is incompatible with this repo, so we did it on a different platform.
# 
# The code to run this can be access via:
# 
# ```
# from extract_umls import extract_umls_code
# ```

# The results are stored as `species_dict_umls.json` file.

# In[20]:


with open('species_dict_umls.json', 'r') as f:
    species_dict_umls = json.load(f)
species_dict_umls['IL6{serum}']


# ### Connecting to PrimeKG
# 
# PrimeKG provides umls mappings for its nodes under the new publicaiton
# 
# Su, X., Messica, S., Huang, Y., Johnson, R., Fesser, L., Gao, S., Sahneh, F. and Zitnik, M., 2025. Multimodal Medical Code Tokenizer. arXiv preprint arXiv:2502.04397.
