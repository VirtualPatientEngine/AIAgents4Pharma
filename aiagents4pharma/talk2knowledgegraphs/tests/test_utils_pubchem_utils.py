"""
Test cases for utils/pubchem_utils.py
"""

from ..utils import pubchem_utils

def test_cas_rn2pubchem_cid():
    """
    Test the casRN2pubchem_cid function.

    The CAS RN for ethyl cabonate is 105-58-8.
    The PubChem CID for ethyl cabonate is 7766.
    """
    casrn = "105-58-8"
    pubchem_cid = pubchem_utils.cas_rn2pubchem_cid(casrn)
    assert pubchem_cid == 7766

def test_external_id2pubchem_cid():
    """
    Test the external_id2pubchem_cid function.

    The DrugBank ID for Alclometasone is DB00240.
    The PubChem CID for Alclometasone is 5311000.

    The CTD ID for Butylated Hydroxyanisole is D002083
    The PubChem CID for Butylated Hydroxyanisole is 24667.
    """
    drugbank_id = "DB00240"
    pubchem_cid = pubchem_utils.external_id2pubchem_cid('drugbank', drugbank_id)
    assert pubchem_cid == 5311000

    ctd_id = "D002083"
    pubchem_cid = pubchem_utils.external_id2pubchem_cid(
                    'comparative toxicogenomics database',
                    ctd_id)
    assert pubchem_cid == 24667

def test_pubchem_cid_description():
    """
    Test the pubchem_cid_description function.

    The PubChem CID for Alclometasone is 5311000.
    The description for Alclometasone starts with 
        "Alclometasone is a prednisolone compound having an alpha-chloro substituent".
    """
    pubchem_cid = 5311000
    description = pubchem_utils.pubchem_cid_description(pubchem_cid)
    assert description.startswith(
        "Alclometasone is a prednisolone compound having an alpha-chloro substituent")
