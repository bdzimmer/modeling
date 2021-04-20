"""

Utilties for working with my own scene JSON format.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

# Model fields:

# * name - name of model
# * filename - filename to load, usually obj
# * color - default / overall color for the model
# * auto_smooth_angle - if present, enable auto smoothing at the specified angle
# * transformation - object with 'rotation' and 'translation' fields. Defaults
#     to identity transformation if not present.
# * children - a list of additional model objects that should be positioned
#     relative to this one
