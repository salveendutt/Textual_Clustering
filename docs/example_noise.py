import sys
import os
import pandas as pd
# Add src/ to sys.path if running script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from noise_strategy import *

original_text = """Hey, I am a bunch of texts.
I like to be grouped with similar texts.
If you put me in the wrong group, I'll feel lost.
So, find a way to cluster me with others like me!"""
text_series = pd.Series(original_text)

# # Adding no Noise
# print("==============================================")
# print("Adding no Noise")
# changed_text = NoNoise().apply(text_series)
# print(changed_text.tolist())

# # Adding random chars noise
# print("==============================================")
# print("Adding random characters noise")
# changed_text = AddRandomCharsNoise().apply(text_series)
# print(changed_text.tolist())

# # Adding random words noise
# print("==============================================")
# print("Adding random words noise")
# changed_text = AddRandomWordsNoise().apply(text_series)
# print(changed_text.tolist())

# # Delete random words noise
# print("==============================================")
# print("Deleting random words noise")
# changed_text = DeleteRandomWordsNoise().apply(text_series)
# print(changed_text.tolist())

# # Shuffle sentances noise
# print("==============================================")
# print("Shuffling sentances noise")
# changed_text = ShuffleSentencesNoise(noise_level=0.5).apply(text_series)
# print(changed_text.tolist())

# # Replace with synonyms noise
# print("==============================================")
# print("Replacing with synonyms noise")
# changed_text = ReplaceWithSynonymsNoise().apply(text_series)
# print(changed_text.tolist())

# # Replace with antonyms noise
# print("==============================================")
# print("Replacing with antonyms noise")
# changed_text = ReplaceWithAntonymsNoise().apply(text_series)
# print(changed_text.tolist())

# All noise strategies
print("==============================================")
noise_strategies = [AddRandomCharsNoise(), AddRandomWordsNoise(), DeleteRandomWordsNoise(), ShuffleSentencesNoise(noise_level=0.5), ReplaceWithSynonymsNoise(), ReplaceWithAntonymsNoise()]
print("All noise strategies")
changed_text = CompositeNoise(noise_strategies).apply(text_series)
print(changed_text.tolist())