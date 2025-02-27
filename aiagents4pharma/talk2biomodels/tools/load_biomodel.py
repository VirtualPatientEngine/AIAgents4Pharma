#!/usr/bin/env python3

"""
Function for loading the BioModel.
"""

from typing import Annotated, Any, Union
from pydantic import BaseModel, BeforeValidator
from ..models.basico_model import BasicoModel

def ensure_biomodel_id(value: Any) -> Any:
    """
    Ensure that the biomodel_id is an integer or a string starting with 'BIOMD' or 'MODEL'.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str) and (value.startswith("BIOMD") or value.startswith("MODEL")):
        return value
    raise ValueError("biomodel_id must be an integer or a string starting with 'BIOMD' or 'MODEL'.")

class ModelData(BaseModel):
    """
    Base model for the model data.
    """
    biomodel_id: Annotated[Union[int, str], BeforeValidator(ensure_biomodel_id)] = None
    # sbml_file_path: Optional[str] = None
    use_uploaded_sbml_file: bool = False

def load_biomodel(sys_bio_model, sbml_file_path=None):
    """
    Load the BioModel.
    """
    model_object = None
    if sys_bio_model.biomodel_id:
        model_object = BasicoModel(biomodel_id=sys_bio_model.biomodel_id)
    elif sbml_file_path:
        model_object = BasicoModel(sbml_file_path=sbml_file_path)
    return model_object
