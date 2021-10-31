
import numpy as np
from __classes__ import TranslationClass
from __validation__ import ValidationTranslation

class Translation(TranslationClass):

    def __init__(self, translation):
        ValidationTranslation(translation)
        if translation is None:
            self._t = np.zeros(3)

    def __str__(self):
        return str(self._t)
