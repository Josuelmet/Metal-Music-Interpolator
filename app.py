from datetime import datetime
import gradio as gr
from gradio.components import *
import mgzip
import numpy as np
from os.path import join
import pickle
from zipfile import ZipFile

from _Generation import Generator

import tensorflow as tf
from tensorflow import keras



gen = Generator()

def main(artist):
    
    if artist == 'Any':
        artist = None
    gen.generate_track_batch(artist)
    filename = f'generation_{datetime.now().strftime("%Y_%m_%d %H_%M_%S")}.gp5'
    gen.save_tracks(filename)
    
    # create a ZipFile object
    zipObj = ZipFile(filename.replace('.gp5', '.zip'), "w")
    zipObj.write(filename)
    zipObj.close()

    return filename.replace('.gp5', '.zip')

with mgzip.open(join('data', 'track_data.pickle.gz'), 'rb') as file:
    track_data = pickle.load(file)

    
inputs = Radio(['Any'] + list(track_data.artist.unique()), label='Choose an Artist for Song Initialization:')


i = gr.Interface(fn = main, inputs = inputs,
                 outputs = File(label='Generated Guitar Tabs. Download and Unzip to View:'))
i.launch()