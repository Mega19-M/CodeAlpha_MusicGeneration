import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras.utils import to_categorical

# 1. Parse MIDI Files & Extract Notes
notes = []
print("Parsing MIDI files...")

for file in glob.glob("midi_songs/*.mid"):
    try:
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
            
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    except Exception as e:
        print(f"Error parsing {file}: {e}")

# Fallback: Create mock note sequence if no local MIDI files are present
if len(notes) < 50:
    print("Using built-in sample note sequence for demo...")
    sample_pitches = ['C4', 'E4', 'G4', 'C5', 'G4', 'E4', 'C4', 'D4', 'F4', 'A4', 'D5', 'A4', 'F4', 'D4']
    notes = sample_pitches * 20

# 2. Prepare Sequences for Training
pitchnames = sorted(set(item for item in notes))
n_vocab = len(pitchnames)
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

sequence_length = 10
network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length, 1):
    seq_in = notes[i : i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)
X = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
y = to_categorical(network_output, num_classes=n_vocab)

# 3. Build & Train LSTM Model
model = Sequential([
    LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(n_vocab),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
print("Training AI Music Model...")
model.fit(X, y, epochs=15, batch_size=32, verbose=1)

# 4. Generate New Music Sequence
start = np.random.randint(0, len(network_input) - 1)
pattern = list(network_input[start])
prediction_output = []

print("Generating music notes...")
for note_index in range(100):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    
    pattern.append(index)
    pattern = pattern[1:]

print("Generated notes sequence:")
print(prediction_output)

# 5. Convert Generated Output to MIDI File
offset = 0
output_notes = []

for pattern_item in prediction_output:
    if ('.' in pattern_item) or pattern_item.isdigit():
        notes_in_chord = pattern_item.split('.')
        chord_notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            chord_notes.append(new_note)
        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern_item)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
    offset += 0.5

midi_stream = stream.Stream(output_notes)
output_filename = 'generated_music.mid'
midi_stream.write('midi', fp=output_filename)
print(f"Success! Music generated and saved as '{output_filename}'.")


# Install FluidSynth and a SoundFont for playback
!apt-get update -qq
!apt-get install -y fluidsynth
!pip install midi2audio pyfluidsynth

# Download a standard General MIDI SoundFont
!wget https://a320.free.fr/SF2/FluidR3_GM.sf2 -O SoundFont.sf2 -q || true

from midi2audio import FluidSynth
from IPython.display import Audio

# Convert MIDI to WAV
fs = FluidSynth('SoundFont.sf2')
fs.midi_to_audio('generated_music.mid', 'generated_music.wav')

# Play the WAV file in Colab
Audio('generated_music.wav')
