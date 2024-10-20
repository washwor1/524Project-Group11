from get_novels import get_novels
from preprocess_data import process_all_files
from feature_engineering import extract_features
from modeling import model_data
from ui import run_ui
import os


data_dir = f"{os.path.dirname(__file__)}/data"

# Step 1: Get the novels from the database, they are saved to data_dir (default './data')
# get_novels(data_dir)

# Step 2: Break all novels into individual sections and store them in individualized csv
process_all_files(data_dir)

# Step 3: Get the features from all the data we got from Step 2
embeddings_index = extract_features(data_dir)

# Step 4: Use the features extracted from step 4 to train a ML model
model = model_data()

positive_text = "Ralph who had started at the word “burglar” again spoke to her two or three times. His words did not seem to reach her ears; it was like knocking against a closed door. It was better to keep silent and wait for his revenge. He remained silent therefore in his corner, disconcerted by the way the adventure had turned out, but in his heart of hearts charmed and full of hope. What a delightful creature, original and ravishing, mysterious and so frank! And gifted with what keen powers of observation! What an insight she had shown into his motives! How she had revealed the slight errors which his contempt of danger sometimes allowed him to commit! In the matter of those two initials— He took his hat from the rack, tore the silk lining out of it, stepped into the corridor, and threw it out of the window. Then he also laid himself at full length on the seat, buried his head in his two pillows and fell into an idle reverie. Life wore a rosy hue. His note-case was full of notes easily gained. Twenty profitable plans that he could certainly carry out jostled one another in his ingenious brain, and next morning he would awake with the pleasing sight of a charming girl in the corner facing him. He dwelt on this thought with uncommon pleasure; presently in a doze he saw her beautiful blue eyes, the color of heaven. Then a strange thing happened. Slowly, to his surprise, they changed color and became green, the color of the sea. He was no longer quite sure whether it was the eyes of the English girl or of the Parisienne that gazed at him in this half-light. Then the young Parisienne was smiling at him, a charming smile. In"
# positive_text = "\"She did not trust me,\" Noël Dorgeroux continued. \"Oh, I had so many disappointments, so many lamentable failures! Do you remember, Victorien, do you remember my experiment on intensive germination by means of electric currents, my experiments with oxygen and all the rest, all the rest, not one of which succeeded? The pluck it called for! But I never lost faith for a minute! . . . One idea in particular buoyed me up and I came back to it incessantly, as though I were able to penetrate the future. You know to what I refer, Victorien: it appeared and reappeared a score of times under different forms, but the principle remained the same. It was the idea of utilizing the solar heat. It's all there, you know, in the sun, in its action upon us, upon cells, organisms, atoms, upon all the more or less mysterious substances that nature has placed at our disposal. And I attacked the problem from every side. Plants, fertilizers, diseases of men and animals, photographs: for all these I wanted the collaboration of the solar rays, utilized by the aid of special processes which were mine alone, my secret and nobody else's.\" My uncle Dorgeroux was talking with renewed eagerness; and his eyes shone feverishly. He now held forth without interrupting himself: \"I will not deny that there was an element of chance about my discovery. Chance plays its part in everything. There never was a discovery that did not exceed our inventive effort; and I can confess to you, Victorien, that I do not even now understand what has happened. No, I can't explain it by a long way; and I can only just believe it. But, all the same, if I had not sought in that direction, the thing would not have"
print(positive_text)
run_ui(positive_text, embeddings_index, model)