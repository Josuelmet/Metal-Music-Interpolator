'''
Imports
'''
import guitarpro
from guitarpro import *
import numpy as np
import os
import pickle
from tqdm import tqdm

from keras.utils import np_utils

from _NoteData import NoteData


'''
Constants
'''
# PITCH[i] = the pitch associated with midi note number i.
# For example, PITCH[69] = 'A4'
PITCH = {val : str(GuitarString(number=0, value=val)) for val in range(128)}
# MIDI[string] = the midi number associated with the note described by string.
# For example, MIDI['A4'] = 69.
MIDI  = {str(GuitarString(number=0, value=val)) : val for val in range(128)}




'''
process_notes function
'''
def process_notes(beat, tuning, as_fingerings=True):
    
    noteData = NoteData()
    
    duration = (beat.duration.value, beat.duration.isDotted)
    
    # Tuplets are cool but rare.
    # If a tuplet is found, simply halve its play time (by doubling its duration value) to simplify things.
    if beat.duration.tuplet.enters != 1 or beat.duration.tuplet.times != 1:
        duration = (duration[0] * 2, duration[1]) # Tuples aren't mutable, so just re-assign the tuple.
        
    noteData.duration = duration[0]
    noteData.isDotted = duration[1]
    
    if len(beat.notes) == 0:
        # return 'rest', duration[0], duration[1], False
        noteData.value = 'rest'
        return noteData
    
    noteData.palmMute = beat.notes[0].effect.palmMute
    
    
    note_types = [note.type for note in beat.notes]

    
    if all(note_type == NoteType.rest for note_type in note_types):
        #return 'rest', duration[0], duration[1], False
        noteData.value = 'rest'
        return noteData
    
    if all(note_type == NoteType.tie for note_type in note_types):
        #return 'tied', duration[0], duration[1], False
        noteData.value = 'tied'
        return noteData
    
    if all(note_type == NoteType.dead for note_type in note_types):
        # return 'dead', duration[0], duration[1], False
        noteData.value = 'dead'
        return noteData
                        
                        
                        
    lowest_string = len(tuning)
    
    
    if as_fingerings:
        # NEW CODE: Represent each pitch as its distance (in semitones) from the tuning of the lowest string.
        pitches = np.array([note.value + tuning[note.string] - tuning[lowest_string] for note in beat.notes if note.type == NoteType.normal])
    else:
        # note_number = MIDI note number, where A4 = 440 Hz = note 69
        # OLD CODE:
        pitches = np.array([note.value + tuning[note.string] for note in beat.notes if note.type == NoteType.normal])

    # Remove any possible NaN values.
    pitches = pitches[~np.isnan(pitches)]
    
    
    # Pitches are often stored in descending order, but we want to make sure they're in ascending order.
    # Thus, we flip the pitches before sorting, so as to help the algorithm.
    pitches = np.sort(pitches[::-1]) 
    
    if len(pitches) == 0:
        #return 'rest', duration[0], duration[1]
        noteData.value = 'rest'
        return noteData
    
    if len(pitches) == 1:
        if as_fingerings:
            # NEW CODE:
            # return str(pitches[0]), duration[0], duration[1]
            noteData.value = str(pitches[0])
            return noteData
        else:
            # OLD CODE:
            # return PITCH[pitches[0]], duration[0], duration[1]
            noteData.value = PITCH[pitches[0]]
            return noteData
    
    # Look at the pitch intervals in the lowest 3 notes that are being played.
    # Usually, chords will start at the lowest 2 notes.
    # However, sometimes players will strum the open lowest string constantly throughout the song.
    # (see: 'Be Quiet and Drive', 'Kaiowas')
    # Thus, the next-highest pair of notes should be considered when labeling a chord.
    if len(pitches) == 2:
        note_pairs = [(0, 1)]
    if len(pitches) == 3:
        note_pairs = [(0, 1), (0, 2), (1, 2)]
    elif len(pitches) >= 4:
        note_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        
    for idx1, idx2 in note_pairs:

        interval = pitches[idx2] - pitches[idx1]
        
        if interval == 12 or interval == 7:
            # Return a power chord associated with pitches[idx1]
            if as_fingerings:
                # NEW CODE:
                # return str(pitches[idx1]) + '_5', duration[0], duration[1]
                noteData.value = str(pitches[idx1]) + '_5'
                return noteData
            else:
                # OLD CODE:
                # return PITCH[pitches[idx1]] + '_5', duration[0], duration[1]
                noteData.value = PITCH[pitches[idx1]] + '_5'
                return noteData
                
        if interval == 6:
            # Return a tritone chord associated with pitches[idx1]
            if as_fingerings:
                # NEW CODE:
                # return str(pitches[idx1]) + '_dim5', duration[0], duration[1]
                noteData.value = str(pitches[idx1]) + '_dim5'
                return noteData
            else:
                # OLD CODE:
                # return PITCH[pitches[idx1]] + '_dim5', duration[0], duration[1]
                noteData.value = PITCH[pitches[idx1]] + 'dim_5'
                return noteData
        
        if interval == 5:
            # Return a P4 chord associated with pitches[idx1]
            if as_fingerings:
                # return str(pitches[idx1]) + '_4', duration[0], duration[1]
                noteData.value = str(pitches[idx1]) + '_4'
                return noteData
            else:
                # return PITCH[pitches[idx1]] + '_4', duration[0], duration[1]
                noteData.value = PITCH[pitches[idx1]] + '_4'
                return noteData
        
        
    
    if as_fingerings:
        # NEW CODE:
        #return str(pitches[0]), duration[0], duration[1]
        noteData.value = str(pitches[0])
        return noteData
    else:
        # OLD CODE:
        # return PITCH[pitches[0]], duration[0], duration[1]
        noteData.value = PITCH[pitches[0]]
        return noteData
    
    


'''
compress_track function
'''
def compress_track(track, as_fingerings=True):
    # 'song' contains the compressed representation of track.
    song = np.empty(len(track.measures), dtype=object)
    
    # Get the tuning and lowest string of the instrument in this track.
    tuning = {string.number : string.value for string in track.strings}
    lowest_string = len(tuning) # Bass have 4-6 strings, while metal guitars have 6 - 8 strings.

    #print(f'Tuning = {[PITCH[x] for x in tuning.values()]}')

    for m_i, measure in enumerate(track.measures):
        '''
        Upon inspection of some of the most popular Songsterr .gp5 tabs,
        it turns out that each measure always has two Voices.
        The first Voice (index 0) always contains music, while
        the second Voice (index 1) always just contains an empty Beat with no notes.

        Therefore, only the first Voice (index 0) actually matters.
        '''
        song[m_i] = []

        #print(m_i+1)
        for b_i, beat in enumerate(measure.voices[0].beats):
            song[m_i].append(process_notes(beat, tuning, as_fingerings).as_tuple())
            #print('\t', song[m_i][b_i], '\t', beat.duration)
            
    return song