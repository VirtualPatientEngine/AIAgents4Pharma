"""
Test cases for utils/pubchem_utils.py
"""

from ..utils import pubchem_utils

def test_drugbank_id2pubchem_cid():
    """
    Test the drugbank_id2pubchem_cid method.

    The DrugBank ID for Alclometasone is DB00240.
    The PubChem CID for Alclometasone is 5311000.
    """
    drugbank_id = "DB00240"
    pubchem_cid = pubchem_utils.drugbank_id2pubchem_cid(drugbank_id)
    assert pubchem_cid == 5311000
