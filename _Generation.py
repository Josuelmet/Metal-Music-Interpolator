import guitarpro
from guitarpro import *
from matplotlib import pyplot as plt
import mgzip
import numpy as np
import os
from os.path import join
import pickle
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten

from _Decompressor import SongWriter



# Define some constants:


# PITCH[i] = the pitch associated with midi note number i.
# For example, PITCH[69] = 'A4'
PITCH = {val : str(GuitarString(number=0, value=val)) for val in range(128)}
# MIDI[string] = the midi number associated with the note described by string.
# For example, MIDI['A4'] = 69.
MIDI  = {str(GuitarString(number=0, value=val)) : val for val in range(128)}






# Generation helper methods:
def thirty_seconds_to_duration(count):
    if count % 3 == 0:
        # If the note is dotted, do 32 / (i * 2/3), and return isDotted = True.
        return (48//count, True)
    else:
        # If the note is not dotted, to 32 / i, and return isDotted = False.
        return (32//count, False)

    
def quantize_thirty_seconds(value):

    # 32nd-note values of each fundamental type of note (not including 64th-notes, of course).
    vals = np.array([32, # whole
                     24, # dotted half
                     16, # half
                     12, # dotted quarter
                     8,  # quarter
                     6,  # dotted eigth
                     4,  # eigth
                     3,  # dotted sixteenth
                     2,  # sixteenth
                     1]) # thirty-second
    
    list_out = []

    for v in vals:
        if v <= value:
            list_out.append(thirty_seconds_to_duration(v))
            value -= v
            
    return np.array(list_out)




def adjust_to_4_4(prediction_output):
    '''
    Adjust prediction output to be in 4/4 time.
    Then, separate the beats into measures.
    '''

    # This will be the prediction output
    new_prediction_output = []


    time = 0
    for beat in prediction_output:

        # Calculate the fraction of a measure encompassed by the current beat / chord.
        beat_time = (1 / beat[1]) * (1 + 0.5 * beat[2])

        # Calculate the fraction of a measure taken up by all notes in the measure.
        # Calculate any residual time to see if this measure (in 4/4 time) is longer than 1 measure.
        measure_time = time + beat_time
        leftover_time = (measure_time) % 1

        # If the measure count (i.e., the measure integer) has changed and there is significant left-over beat time:
        if (int(measure_time) > int(time)) and (leftover_time > 1/128):

            # Calculate the initial 32nd notes encompassed by this beat in the current measure.
            this_measure_thirty_seconds = int(32 * (1 - time % 1))
            # Calculate the remaining 32nd notes encompassed by this beat in the next measure.
            next_measure_thirty_seconds = int(32 * leftover_time)

            # Get the Duration object parameters for this measure and the next measure.
            this_measure_durations = quantize_thirty_seconds(this_measure_thirty_seconds)
            next_measure_durations = quantize_thirty_seconds(next_measure_thirty_seconds)


            #print(f'{{ {32 / beat[1]}')
            for duration_idx, duration in enumerate(this_measure_durations):
                time += (1 / duration[0]) * (1 + 0.5 * duration[1])

                #print(time, '\t', time * 32)

                chord = beat[0] if duration_idx == 0 else 'tied'

                new_prediction_output.append((chord, duration[0], duration[1], beat[3]))


            for duration in next_measure_durations:
                time += (1 / duration[0]) * (1 + 0.5 * duration[1])

                #print(time, '\t', time * 32)

                new_prediction_output.append(('tied', duration[0], duration[1], beat[3]))


            continue


        time += beat_time
        new_prediction_output.append((beat[0], beat[1], beat[2], beat[3]))

        #print(time, '\t', time * 32)


    '''
    # Code for debugging
    
    time = 0
    time2 = 0
    idx = 0

    for idx2, beat2 in enumerate(new_prediction_output[:100]):
        beat = prediction_output[idx]

        if time == time2:
            print(beat[0], '\t', time, '\t\t', beat2[0], '\t', time2)

            idx += 1

            time += (1 / beat[1]) * (1 + 0.5 * beat[2])

        else:
            print('\t\t\t\t', beat2[0], '\t', time2)



        time2 += (1 / beat2[1]) * (1 + 0.5 * beat2[2])
    ''';
    
    # Use the previously calculated cumulative time as the number of measures in the new 4/4 song.
    num_measures = int(np.ceil(time))

    song = np.empty(num_measures, dtype=object)

    time = 0
    m_idx = 0

    timestamps = []

    for beat in new_prediction_output:
        #print(time)
        timestamps.append(time)

        m_idx = int(time)

        if song[m_idx] is None:

            song[m_idx] = [beat]
        else:
            song[m_idx].append(beat)


        time += (1 / beat[1]) * (1 + 0.5 * beat[2])


    print(f'4/4 adjusted correctly: {set(range(num_measures)).issubset(set(timestamps))}')
    
    return song







class Generator:
    def __init__(self, num_tracks_to_generate=5, as_fingerings=True, sequence_length=100):
        with mgzip.open(join('data', 'notes_data.pickle.gz'), 'rb') as filepath:
            self.notes = pickle.load(filepath)
            self.note_to_int = pickle.load(filepath)
            self.int_to_note = pickle.load(filepath)
            self.n_vocab = pickle.load(filepath)
            self.NUM_TRACKS_TO_GENERATE = num_tracks_to_generate
            self.as_fingerings = as_fingerings
            self.sequence_length = sequence_length

        with mgzip.open(join('data', 'track_data.pickle.gz'), 'rb') as filepath:
            self.track_data = pickle.load(filepath)

        self.model = keras.models.load_model('minigpt')

        self.ints = np.array([self.note_to_int[x] for x in self.notes])
        
        
        
    def generate_track(self, track_idx=None):

        if track_idx is None:
            # Choose a random track
            track_idx = np.random.choice(len(self.track_data))

        # Get the note indices corresponding to the beginning and ending of the track
        song_note_idx_first = self.track_data.loc[track_idx]['noteStartIdx']
        song_note_idx_last = self.track_data.loc[track_idx+1]['noteStartIdx']

        # Choose a random starting point within the track
        start_idx = np.random.randint(low=song_note_idx_first,
                                      high=song_note_idx_last)

        # Choose a number of initial notes to select from the track, at most 100.
        #num_initial_notes = np.random.choice(min(100, song_note_idx_last - start_idx))
        num_initial_notes = np.random.choice(min(100, song_note_idx_last - start_idx))

        # Select the initial notes (tokens)
        start_tokens = [_ for _ in self.ints[start_idx:start_idx+num_initial_notes]]


        max_tokens = 100



        def sample_from(logits, top_k=10):
            logits, indices = tf.math.top_k(logits, k=top_k, sorted=True)
            indices = np.asarray(indices).astype("int32")
            preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
            preds = np.asarray(preds).astype("float32")
            return np.random.choice(indices, p=preds)

        num_tokens_generated = 0
        tokens_generated = []

        while num_tokens_generated <= max_tokens:
            pad_len = self.sequence_length - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.sequence_length]
                sample_index = self.sequence_length - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)

        generated_notes = [self.int_to_note[num] for num in np.concatenate((start_tokens, tokens_generated))]

        return track_idx, generated_notes
    
    
    
    def generate_track_batch(self, artist=None):

        self.track_indices = np.zeros(self.NUM_TRACKS_TO_GENERATE)
        self.tracks = np.zeros(self.NUM_TRACKS_TO_GENERATE, dtype=object)


        for i in tqdm(range(self.NUM_TRACKS_TO_GENERATE)):
            if artist is None:
                idx, t = self.generate_track()
            else:
                idx, t = self.generate_track(track_idx=np.random.choice(list(self.track_data[self.track_data.artist==artist].index)))
            self.track_indices[i] = idx
            self.tracks[i] = t
            
            
            
    def save_tracks(self, filepath='_generation.gp5'):

        songWriter = SongWriter(initialTempo=self.track_data.loc[self.track_indices[0]]['tempo'])

        for idx in range(len(self.tracks)):
            new_track = adjust_to_4_4(self.tracks[idx])

            # Get the tempo and tuning (lowest string note) of the song:
            #print(          track_data.loc[track_indices[idx]])
            tempo         = self.track_data.loc[self.track_indices[idx]]['tempo']
            instrument    = self.track_data.loc[self.track_indices[idx]]['instrument'] 
            name          = self.track_data.loc[self.track_indices[idx]]['song']
            lowest_string = self.track_data.loc[self.track_indices[idx]]['tuning']

            if not self.as_fingerings:
                # Get all the unique pitch values from the new track
                pitchnames = set.union(*[set([beat[0].split('_')[0] for beat in measure]) for measure in new_track])
                pitchnames.discard('rest') # Ignore rests
                pitchnames.discard('tied') # Ignore tied notes
                pitchnames.discard('dead') # Ignore dead/ghost notes
                lowest_string = min([MIDI[pitch] for pitch in pitchnames]) # Get the lowest MIDI value / pitch
                lowest_string = min(lowest_string, MIDI['E2']) # Don't allow any tunings higher than standard.


            # Standard tuning
            tuning = {1: MIDI['E4'],
                      2: MIDI['B3'],
                      3: MIDI['G3'],
                      4: MIDI['D3'],
                      5: MIDI['A2'],
                      6: MIDI['E2']}

            if lowest_string <= MIDI['B1']:
                # 7-string guitar case
                tuning[7] = MIDI['B1']
                downtune = MIDI['B1'] - lowest_string
            else:
                # downtune the tuning by however much is necessary.
                downtune = MIDI['E2'] - lowest_string

            tuning = {k: v - downtune for k, v in tuning.items()} # Adjust to the new tuning

            # Write the track to the song writer
            songWriter.decompress_track(new_track, tuning, tempo=tempo, instrument=instrument, name=name, as_fingerings=self.as_fingerings)



        songWriter.write(filepath)
        print('Finished')

    

    
    
    
    
    

'''       
        

def init_generator():
    global NUM_TRACKS_TO_GENERATE, notes, note_to_int, int_to_note, n_vocab, track_data, model, ints
    
    with mgzip.open('data\\notes_data.pickle.gz', 'rb') as filepath:
        notes = pickle.load(filepath)
        note_to_int = pickle.load(filepath)
        int_to_note = pickle.load(filepath)
        n_vocab = pickle.load(filepath)

    with mgzip.open('data\\track_data.pickle.gz', 'rb') as filepath:
        track_data = pickle.load(filepath)

    #with mgzip.open('output\\generated_songs.pickle.gz', 'rb') as filepath:
    #    track_indices = pickle.load(filepath)
    #    tracks = pickle.load(filepath)

    model = keras.models.load_model('minigpt')

    ints = np.array([note_to_int[x] for x in notes])
    
    
    

def generate_track(track_idx=None):
    global track_data, ints, int_to_note
    
    if track_idx is None:
        # Choose a random track
        track_idx = np.random.choice(len(track_data))

    # Get the note indices corresponding to the beginning and ending of the track
    song_note_idx_first = track_data.loc[track_idx]['noteStartIdx']
    song_note_idx_last = track_data.loc[track_idx+1]['noteStartIdx']

    # Choose a random starting point within the track
    start_idx = np.random.randint(low=song_note_idx_first,
                                high=song_note_idx_last)

    # Choose a number of initial notes to select from the track, at most 100.
    #num_initial_notes = np.random.choice(min(100, song_note_idx_last - start_idx))
    num_initial_notes = np.random.choice(min(100, song_note_idx_last - start_idx))

    # Select the initial notes (tokens)
    start_tokens = [_ for _ in ints[start_idx:start_idx+num_initial_notes]]


    max_tokens = 100



    def sample_from(logits, top_k=10):
        logits, indices = tf.math.top_k(logits, k=top_k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    num_tokens_generated = 0
    tokens_generated = []

    while num_tokens_generated <= max_tokens:
        pad_len = maxlen - len(start_tokens)
        sample_index = len(start_tokens) - 1
        if pad_len < 0:
            x = start_tokens[:maxlen]
            sample_index = maxlen - 1
        elif pad_len > 0:
            x = start_tokens + [0] * pad_len
        else:
            x = start_tokens
        x = np.array([x])
        y, _ = model.predict(x)
        sample_token = sample_from(y[0][sample_index])
        tokens_generated.append(sample_token)
        start_tokens.append(sample_token)
        num_tokens_generated = len(tokens_generated)

    generated_notes = [int_to_note[num] for num in np.concatenate((start_tokens, tokens_generated))]

    return track_idx, generated_notes




def generate_track_batch(artist=None):
    global track_indices, tracks, NUM_TRACKS_TO_GENERATE, track_data

    track_indices = np.zeros(NUM_TRACKS_TO_GENERATE)
    tracks = np.zeros(NUM_TRACKS_TO_GENERATE, dtype=object)


    for i in tqdm(range(NUM_TRACKS_TO_GENERATE)):
        if artist is None:
            idx, t = generate_track()
        else:
            idx, t = generate_track(track_idx=np.random.choice(list(track_data[track_data.artist==artist].index)))
        track_indices[i] = idx
        tracks[i] = t
        
        
        
        
        
# Generation helper methods:
def thirty_seconds_to_duration(count):
    if count % 3 == 0:
        # If the note is dotted, do 32 / (i * 2/3), and return isDotted = True.
        return (48//count, True)
    else:
        # If the note is not dotted, to 32 / i, and return isDotted = False.
        return (32//count, False)

    
def quantize_thirty_seconds(value):

    # 32nd-note values of each fundamental type of note (not including 64th-notes, of course).
    vals = np.array([32, # whole
                     24, # dotted half
                     16, # half
                     12, # dotted quarter
                     8,  # quarter
                     6,  # dotted eigth
                     4,  # eigth
                     3,  # dotted sixteenth
                     2,  # sixteenth
                     1]) # thirty-second
    
    list_out = []

    for v in vals:
        if v <= value:
            list_out.append(thirty_seconds_to_duration(v))
            value -= v
            
    return np.array(list_out)




def adjust_to_4_4(prediction_output):
    
    #Adjust prediction output to be in 4/4 time.
    #Then, separate the beats into measures.
    

    # This will be the prediction output
    new_prediction_output = []


    time = 0
    for beat in prediction_output:

        # Calculate the fraction of a measure encompassed by the current beat / chord.
        beat_time = (1 / beat[1]) * (1 + 0.5 * beat[2])

        # Calculate the fraction of a measure taken up by all notes in the measure.
        # Calculate any residual time to see if this measure (in 4/4 time) is longer than 1 measure.
        measure_time = time + beat_time
        leftover_time = (measure_time) % 1

        # If the measure count (i.e., the measure integer) has changed and there is significant left-over beat time:
        if (int(measure_time) > int(time)) and (leftover_time > 1/128):

            # Calculate the initial 32nd notes encompassed by this beat in the current measure.
            this_measure_thirty_seconds = int(32 * (1 - time % 1))
            # Calculate the remaining 32nd notes encompassed by this beat in the next measure.
            next_measure_thirty_seconds = int(32 * leftover_time)

            # Get the Duration object parameters for this measure and the next measure.
            this_measure_durations = quantize_thirty_seconds(this_measure_thirty_seconds)
            next_measure_durations = quantize_thirty_seconds(next_measure_thirty_seconds)


            #print(f'{{ {32 / beat[1]}')
            for duration_idx, duration in enumerate(this_measure_durations):
                time += (1 / duration[0]) * (1 + 0.5 * duration[1])

                #print(time, '\t', time * 32)

                chord = beat[0] if duration_idx == 0 else 'tied'

                new_prediction_output.append((chord, duration[0], duration[1]))


            for duration in next_measure_durations:
                time += (1 / duration[0]) * (1 + 0.5 * duration[1])

                #print(time, '\t', time * 32)

                new_prediction_output.append(('tied', duration[0], duration[1]))


            continue


        time += beat_time
        new_prediction_output.append((beat[0], beat[1], beat[2]))

        #print(time, '\t', time * 32)


    
    # Code for debugging
    
    #time = 0
    #time2 = 0
    #idx = 0

    #for idx2, beat2 in enumerate(new_prediction_output[:100]):
    #    beat = prediction_output[idx]

    #    if time == time2:
    #        print(beat[0], '\t', time, '\t\t', beat2[0], '\t', time2)

    #        idx += 1

    #        time += (1 / beat[1]) * (1 + 0.5 * beat[2])

    #    else:
    #        print('\t\t\t\t', beat2[0], '\t', time2)



    #    time2 += (1 / beat2[1]) * (1 + 0.5 * beat2[2])
    
    
    # Use the previously calculated cumulative time as the number of measures in the new 4/4 song.
    num_measures = int(np.ceil(time))

    song = np.empty(num_measures, dtype=object)

    time = 0
    m_idx = 0

    timestamps = []

    for beat in new_prediction_output:
        #print(time)
        timestamps.append(time)

        m_idx = int(time)

        if song[m_idx] is None:

            song[m_idx] = [beat]
        else:
            song[m_idx].append(beat)


        time += (1 / beat[1]) * (1 + 0.5 * beat[2])


    print(f'4/4 adjusted correctly: {set(range(num_measures)).issubset(set(timestamps))}')
    
    return song






def save_tracks(filepath='_generation.gp5'):
    global track_data, track_indice, tracks
    
    songWriter = SongWriter(initialTempo=track_data.loc[track_indices[0]]['tempo'])

    for idx in range(len(tracks)):
        new_track = adjust_to_4_4(tracks[idx])

        # Get the tempo and tuning (lowest string note) of the song:
        #print(          track_data.loc[track_indices[idx]])
        tempo         = track_data.loc[track_indices[idx]]['tempo']
        instrument    = track_data.loc[track_indices[idx]]['instrument'] 
        name          = track_data.loc[track_indices[idx]]['song']
        lowest_string = track_data.loc[track_indices[idx]]['tuning']

        if not as_fingerings:
            # Get all the unique pitch values from the new track
            pitchnames = set.union(*[set([beat[0].split('_')[0] for beat in measure]) for measure in new_track])
            pitchnames.discard('rest') # Ignore rests
            pitchnames.discard('tied') # Ignore tied notes
            pitchnames.discard('dead') # Ignore dead/ghost notes
            lowest_string = min([MIDI[pitch] for pitch in pitchnames]) # Get the lowest MIDI value / pitch
            lowest_string = min(lowest_string, MIDI['E2']) # Don't allow any tunings higher than standard.


        # Standard tuning
        tuning = {1: MIDI['E4'],
                  2: MIDI['B3'],
                  3: MIDI['G3'],
                  4: MIDI['D3'],
                  5: MIDI['A2'],
                  6: MIDI['E2']}

        if lowest_string <= MIDI['B1']:
            # 7-string guitar case
            tuning[7] = MIDI['B1']
            downtune = MIDI['B1'] - lowest_string
        else:
            # downtune the tuning by however much is necessary.
            downtune = MIDI['E2'] - lowest_string

        tuning = {k: v - downtune for k, v in tuning.items()} # Adjust to the new tuning

        # Write the track to the song writer
        songWriter.decompress_track(new_track, tuning, tempo=tempo, instrument=instrument, name=name, as_fingerings=as_fingerings)



    songWriter.write(filepath)
    print('Finished')
'''