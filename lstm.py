""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN

uniqueNotes =[]
notesCount= []

def train_network():
    """ Train a Neural Network to generate music """
    notes = parser()
    #notes.append(str(note.Rest().fullName))
    # get amount of pitch names
    print(set(notes))    
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)



def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 1

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)
    print("output:")
    print(network_output)
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    filepath_acc="CSV-DATEI.csv"
    accuracy = CSVLogger(filepath_acc, separator=',', append=False)

    callbacks_list = [checkpoint,accuracy, TerminateOnNaN()]

    history = model.fit(network_input, network_output, epochs=10000, batch_size=64, callbacks=callbacks_list)
    print(history.history.keys)
"""checks wether the note has already been seen, adds it into seen ones if its new, also keeps track of number of appearances"""
def appNotes(musicNote):
    isNew = True
    string = str(musicNote)
    for a, sign in enumerate(uniqueNotes):
        if sign == string:
            isNew = False
            notesCount[a]+=1
            break
    if(isNew):
        uniqueNotes.append(string)
        notesCount.append(1)

"""goes through every midi file in folder and fills a list of unique notes and a list of all the notes played in the file
also counts how many times a certain note is played"""
def parser():
    notes_to_parse = parse_files()
    for element in notes_to_parse: #takes every note and asks if its been seen already
            if isinstance(element, note.Note):
                #notes.append(str(element.pitch))
                #print(str(element.pitch))
                appNotes(element.pitch)
            #elif isinstance(element, chord.Chord):
               # notes.append('.'.join(str(n) for n in element.normalOrder))
    #print("length of notes:", len(notes))
    print("length of uniqueNotes:", len(uniqueNotes))
    counter = 0
    #printout because we can
    for c, obj in enumerate(uniqueNotes):
        print(uniqueNotes[c], "amount: ", notesCount[c])
        counter += notesCount[c]
    print("Sum:       ", counter)
    with open('notes', 'wb') as filepath:
        pickle.dump(uniqueNotes, filepath)
    print(counter)
    return uniqueNotes


"""parses the files in source folder"""
def parse_files():
    notes_to_parse = None
    for file in glob.glob("source/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None
        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
            print(len(s2.parts))
            print("partitioned by instrument")
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
            print("flat notes")

    return notes_to_parse

"""old method, notes will be too long"""
def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    counter = 0
    for file in glob.glob("source/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:

            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                print(str(element.pitch))
                counter+=1
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    print(counter)
    return notes

if __name__ == '__main__':
    train_network()
