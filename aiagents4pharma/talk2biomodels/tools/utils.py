#!/usr/bin/env python3

"""
Utility functions for T2B tools.
"""

import basico

def get_model_units(model_object):
    """
    Get the units of the model.

    Args:
        model_object: The model object.

    Returns:
        dict: The units of the model.
    """
    model_units = basico.model_info.get_model_units(model=model_object.copasi_model)
    model_units_y = model_units['quantity_unit']
    model_units_x = model_units['time_unit']
    return {'y_axis_label': model_units_y, 'x_axis_label': model_units_x}
