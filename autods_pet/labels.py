"""TotalSegmentator label IDs used by autods_pet.

Label IDs correspond to TotalSegmentator v2.x class mapping
(Wasserthal et al., Radiology: AI, 2023).  If upgrading
TotalSegmentator to a new major version, verify these IDs
against the updated label map.
"""

from __future__ import annotations

# Lumbar vertebrae
L3 = 29
L4 = 28
L5 = 27

# Thoracic vertebrae
T4 = 40
T5 = 39
T6 = 38
T7 = 37
T8 = 36

# Organs
AORTA = 52
HEART = 51
LIVER = 5
BRAIN = 90

# Long bones
FEMUR_L = 75
FEMUR_R = 76
HUMERUS_L = 69
HUMERUS_R = 70

# Convenience groups
LUMBAR_L3_L5 = [L3, L4, L5]
THORACIC_T4_T8 = list(range(T8, T4 + 1))  # [36, 37, 38, 39, 40]
