'''
Imports
'''
import guitarpro
from guitarpro import *
import numpy as np
import os
import pickle
import re
from tqdm import tqdm

from keras.utils import np_utils


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
get_strings function
'''
def get_strings(chord, tuning, as_fingerings=True):
    
    lowest_string = len(tuning) # Bass has 4 strings, while metal guitars can have 6-8 strings.
    
    if as_fingerings:
        # NEW CODE:
        # Represent the tuning as the number of semitones between the open strings and the lowest string.
        # i.e., relative_tuning[lowest_string] = 0
        relative_tuning = {k : v - tuning[lowest_string] for k, v in tuning.items()}
    
    chord_parts = chord.split('_')
    
    
    if as_fingerings:
        # NEW CODE:
        root_value = int(chord_parts[0])
    else:
        # NEW CODE:
        root_value = MIDI[chord_parts[0]]
    
    
    if as_fingerings:
        # NEW CODE:
        if root_value < 0:
            print(f'!!!!! error !!!!!\t {root_value} < 0')
    else:
        # OLD CODE:
        if root_value < tuning[lowest_string]:
            print(f'!!!!! error !!!!!\t {root_value} < {tuning[lowest_string]}')
    
    # Using the tuning, get a list of all possible fret positions for the root note.
    if as_fingerings:
        # NEW CODE:
        tuning_values = np.array(list(relative_tuning.values()))
    else:
        # OLD CODE:
        tuning_values = np.array(list(tuning.values()))
        
    fingerings = root_value - tuning_values
    
    
    # + 1 because tuning[] is 1-indexed.
    string = np.where(fingerings >= 0, fingerings, np.inf).argmin() + 1
    fret = fingerings[string-1]
    
    # If we are just playing a single note, then the function can just return what it has now.
    if len(chord_parts) == 1:
        return [(fret, string)]
    
    # If the chord requires a very high pitch, lower its fingering to the second-highest string,
    # so as to save the highest string for the other part of the chord.
    if string == 1:
        string = 2
        fret = fingerings[string-1]
    
    if chord_parts[1] == '5':
        upper_value = root_value + 7 # perfect fifth
    elif chord_parts[1] == 'dim5':
        upper_value = root_value + 6 # tritone
    elif chord_parts[1] == '4':
        upper_value = root_value + 5
    else:
        upper_value = root_value + 5 # in case of an error, assume that the upper value is a perfect 4th above the root.
    

    if as_fingerings:
        # NEW CODE:
        upper_fret = upper_value - relative_tuning[string-1]
    else:
        # OLD CODE:
        upper_fret = upper_value - tuning[string-1]
        
        
    if upper_fret < 0:
        # There are some rare cases where the chord cannot be played given a tuning.
        # For example, a tritone or a perfect 4th with root C2 in a drop-C guitar.
        # In that case, just return the root note.
        return [(fret, string)]
    
    return [(fret, string), (upper_fret, string-1)]




class SongWriter:
    
    def __init__(self, initialTempo):
        self.song = guitarpro.models.Song(tempo=initialTempo)
        self.song.tracks = [] # Initialize song.tracks to an empty list.
        
        self.currentTempo = initialTempo

    '''
    decompress_track function
    '''
    def decompress_track(self, track, tuning, tempo=None, instrument=None, name=None, as_fingerings=True):
        
        # Instantiate the Midi Channel / Instrument for this track,
        # making sure to avoid channel 9 (the drum channel)
        channel = MidiChannel(channel = len(self.song.tracks) + (len(self.song.tracks) >= 9),
                              effectChannel = len(self.song.tracks) + (len(self.song.tracks) >= 9))
        if instrument is not None:
            channel.instrument = instrument
        
        # Instantiate the name for this track
        if name is None:
            name = f'Track {len(self.song.tracks) + 1}'
            
        # Pre-format the name to avoid any characters that the tab file format doesn't like.
        # Keep only spaces and alphanumeric characters.
        name = name.split('(')[0]
        name = re.sub(r' \W+', '', name)
        
        # Instantiate the actual Track itself
        self.song.tracks.append(Track(song=self.song, name=name, channel=channel))

        # Calculate the dict key corresponding to the lowest-tuned string.
        lowest_string = len(tuning)
        
        # Set the guitar tuning for the instrument.
        self.song.tracks[-1].strings = [GuitarString(number=num, value=val) for num, val in tuning.items()]

        
         

        # The first measureHeader and measure are already added by default. Let's remove them to make things more standard.
        if len(self.song.tracks) == 1:
            self.song.measureHeaders = []
            self.song.tracks[0].measures = []
            
        startingTrackIdx = len(self.song.measureHeaders)
        
        for i in range(startingTrackIdx, startingTrackIdx+len(track)):
            start = guitarpro.Duration.quarterTime * (1 + i*6)

            self.song.addMeasureHeader(MeasureHeader(number=i+1, start=start))
            
            # Add new measure to every existing track.
            for existing_track in self.song.tracks:
                existing_track.measures.append( Measure(existing_track, self.song.measureHeaders[i]) )





        for m_i, measure in enumerate(self.song.tracks[-1].measures):

            if m_i < startingTrackIdx:
                continue # Skip tracks that have already been written to.
            
            # "beats" starts off as an empy array [].
            voice = measure.voices[0]
            beats = voice.beats

            #print(m_i - startingTrackIdx)
            # For the m_i-th measure, get the indices b_i and the beats track_beat of the compressed song.
            for b_i, track_beat in enumerate(track[m_i - startingTrackIdx]):
                
                #print(f'\t{b_i}')
                
                # If a tempo change is needed:
                if tempo is not None and tempo != self.currentTempo:
                    # Implement the tempo change, then update the current tempo.
                    effect = BeatEffect(mixTableChange=MixTableChange(tempo=MixTableItem(value=tempo), hideTempo=False))
                    self.currentTempo = tempo
                else:
                    effect = BeatEffect()
                

                chord = track_beat[0]
                duration = Duration(value=int(track_beat[1]), isDotted=bool(track_beat[2]))

                # since "beats" is empty, we can append Beat objects to it.
                beats.append(Beat(voice, duration=duration, effect=effect))
                if chord == 'rest':
                    beats[b_i].status = guitarpro.BeatStatus.rest

                elif chord == 'tied':
                    # If this tied note is the first beat in its measure:
                    if b_i == 0:
                        # If this tied note is the first beat in the first measure:
                        if m_i == 0:
                            # Designate this beat as a rest, then move on to the next beat.
                            beats[b_i].status = guitarpro.BeatStatus.rest
                            continue
                        else:
                            # Get the last Beat object from the previous Measure.
                            previous_beats = self.song.tracks[-1].measures[m_i-1].voices[0].beats
                            if len(previous_beats) == 0:
                                beats[b_i].status = guitarpro.BeatStatus.rest
                                continue
                            previous_beat = previous_beats[-1]
                    else:
                        # Get the previous Beat object from the current Measure.
                        previous_beat = beats[b_i-1]

                    for note in previous_beat.notes:
                        beats[b_i].notes.append(Note(beat=beats[b_i], value=note.value, string=note.string, type=NoteType.tie))




                elif chord == 'dead':
                    beats[b_i].notes.append(Note(beat=beats[b_i], value=0, string=lowest_string, type=NoteType.dead))
                    beats[b_i].notes.append(Note(beat=beats[b_i], value=0, string=lowest_string-1, type=NoteType.dead))
                    beats[b_i].notes.append(Note(beat=beats[b_i], value=0, string=lowest_string-2, type=NoteType.dead))

                else:
                    for fret, string in get_strings(chord, tuning, as_fingerings):
                        noteEffect = NoteEffect(palmMute=track_beat[3])
                        beats[b_i].notes.append(Note(beat=beats[b_i], value=fret, string=string,
                                                     type=NoteType.normal, effect=noteEffect))


                #print('\t\t', chord, '\t', duration)


        # Lastly, return the song so that it can be saved to a .gp5 file.
        # return new_song
        
    
    def write(self, filename):
        guitarpro.write(self.song, filename)