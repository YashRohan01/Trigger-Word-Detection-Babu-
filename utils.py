import os
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import lfilter
import numpy as np
import matplotlib.pyplot as plt
import IPython

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def show_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
        plt.show()
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        plt.show()
    return pxx


def load_raw_audio(path):
    babus = []
    backgrounds = []
    negatives = []
    babu_path=os.path.join(path,'babu')
    background_path=os.path.join(path,'backgrounds')
    negative_path=os.path.join(path,'negatives')
    final_path=os.path.join(path,'Final')
    for filename in os.listdir(babu_path):
        if filename.endswith(".wav"):
            babu = AudioSegment.from_wav(os.path.join(babu_path,filename))
            babu=babu.set_frame_rate(44100)
            babus.append(babu)
    for filename in os.listdir(background_path):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(os.path.join(background_path,filename))
            background=background.set_frame_rate(44100)
            backgrounds.append(background)
    for filename in os.listdir(negative_path):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav(os.path.join(negative_path,filename))
            negative=negative.set_frame_rate(44100)
            negatives.append(negative)
    return babus, negatives, backgrounds


def adjust_audio_length(audio_file, target_duration=10000):
  """Adjusts the length of an audio clip to the target duration (in milliseconds).

  Args:
      audio_file: The AudioSegment object representing the audio clip.
      target_duration: The desired length of the audio clip in milliseconds (default: 10 seconds).

  Returns:
      An AudioSegment object with the adjusted length.
  """

  # Get clip duration
  clip_duration = len(audio_file)

  if clip_duration <= target_duration:
    # Clip is already shorter or equal to target duration, return as-is
    return audio_file
  else:
    # Clip needs shortening
    # Calculate the amount to shorten (in milliseconds)
    excess_duration = clip_duration - target_duration
    
    # Slice the audio segment to keep the desired portion
    start_time = 0
    end_time = start_time + target_duration
    adjusted_audio = audio_file[start_time:end_time]

    return adjusted_audio
  
Tx=5511 # Tokens of the input data
n_freq=101 # Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375 # The number of time steps in the output of our model

def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    
    segment_start, segment_end = segment_time
    
    overlap = False
    
    for previous_start, previous_end in previous_segments:
        if segment_start<=previous_end and segment_end>=previous_start:
            overlap = True
            break
    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    Arguments:
    background -- a 10 second background audio recording.  
    audio_clip -- the audio clip to be inserted/overlaid. 
    previous_segments -- times where audio segments have already been placed
    
    Returns:
    new_background -- the updated background audio
    """
    
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    


    segment_time = get_random_time_segment(segment_ms)
    
    retry = 7 
    while is_overlapping(segment_time,previous_segments) and retry >= 0:
        segment_time = get_random_time_segment(segment_ms)
        retry = retry - 1
   
    # if last try is not overlaping, insert it to the background
    if not is_overlapping(segment_time, previous_segments):

        previous_segments.append(segment_time)

        new_background = background.overlay(audio_clip, position = segment_time[0])
    else:
        #print("Timeouted")
        new_background = background
        segment_time = (10000, 10000)
    
    return new_background, segment_time

def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 following labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    _, Ty = y.shape
    
    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    if segment_end_y < Ty:
        # Add 1 to the correct index in the background label (y)
        for i in range(segment_end_y+1, segment_end_y+51):
            if i < Ty:
                y[0, i] = 1
    
    return y

def create_training_example(background, babus, negatives, Ty):
    """
    Creates a training example with a given background, activates, and negatives.
    
    Arguments:
    background -- a 10 second background audio recording
    babus -- a list of audio segments of the word "babu"
    negatives -- a list of audio segments of random words that are not "activate"
    Ty -- The number of time steps in the output

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    # Get a random background
    #background_index=np.random.randint(0,len(backgrounds))
    #background=backgrounds[background_index]
    
    # Make background quieter
    background = background - 20

    # Initialize y (label vector) of zeros 
    y = np.zeros((1,Ty))

    # Initialize segment times as empty list 
    previous_segments = []
    
    # Select 0-2 random "babu" audio clips from the entire list of "babus" recordings
    number_of_babus = np.random.randint(0, 3)
    random_indices = np.random.randint(len(babus), size=number_of_babus)
    random_babus = [babus[i] for i in random_indices]
    
    # Loop over randomly selected "babu" clips and insert in background
    for one_random_babu in random_babus:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, one_random_babu, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y" at segment_end
        y = insert_ones(y,segment_end)
    
    # Select 0-4 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 5)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    ### START CODE HERE ### (≈ 2 lines)
    # Step 4: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    ### END CODE HERE ###
    
    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # Export new training example 
    file_handle = background.export("train" + ".wav", format="wav")
    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = show_spectrogram("train.wav")
    
    return x, y



path=r"D:\Trigger Word"
babus, negatives, backgrounds = load_raw_audio(path)
for i in range(len(backgrounds)):
    backgrounds[i]=adjust_audio_length(backgrounds[i])

np.random.seed(10)
nsamples = 2048
X = []
Y = []
for i in range(0, nsamples):
    if i%10 == 0:
        print(i)
    j=np.random.randint(0,len(backgrounds))
    x, y = create_training_example(backgrounds[j], babus, negatives, Ty)
    X.append(x.swapaxes(0,1))
    Y.append(y.swapaxes(0,1))
X = np.array(X)
Y = np.array(Y)

np.save('train_X.npy',X)
np.save('train_Y.npy',Y)