# 🎶 Bach-Style Music Generation with Deep Learning

This project uses a sequence-to-sequence deep learning model to generate new four-part chorales in the style of J.S. Bach. The model leverages a combination of **1D Convolutional layers (Conv1D)** to capture local melodic and harmonic patterns, followed by a **Long Short-Term Memory (LSTM) network** to maintain the long-range harmonic and structural coherence characteristic of Bach's music.

The notebook covers data loading, preprocessing, model architecture design, training, and music generation, complete with utilities to listen to the generated output.

## 📦 Project Contents

The notebook `Bach_Style_Music_Generation.ipynb` is structured into the following sections:

1.  **Importing Libraries**: Essential Python libraries for data manipulation (`pandas`, `numpy`), deep learning (`tensorflow.keras`), and music processing (`music21`).
2.  **Understanding the Dataset**: Loading and initial inspection of the Bach chorale CSV files, which contain MIDI note numbers for four voices (`note0` to `note3`).
3.  **Data Preprocessing**:
    * Loading all training, testing, and validation data.
    * Implementing a sliding window technique to create input (`X`) and output (`Y`) sequences for next-note prediction.
    * Scaling MIDI notes from their original range (36-81) to a smaller categorical range (1-46), with `0` reserved for rests/silence.
    * Determining the vocabulary size (`num_notes = 47`).
4.  **Model Building and Training**:
    * Defining the **Conv1D + LSTM** sequence model.
    * Using Causal 1D Convolution with increasing dilation rates (`1, 2, 4, 8, 16`) to efficiently capture short- to mid-range temporal context.
    * Applying `BatchNormalization` for stable training.
    * Utilizing an `LSTM` layer for long-range sequence modeling.
    * Training the model using the `Nadam` optimizer and `sparse_categorical_crossentropy` loss.
5.  **Generating Music with Model**:
    * Implementation of the `sample_next_note` function to randomly select the next note based on the model's output probabilities (with safe handling for zero probabilities).
    * The `generate_chorale` function uses a trained model and a seed sequence to iteratively predict and append new notes.
    * The generated sequence is then converted back to MIDI and played using `music21`.
6.  **Insights and Future Improvements**: Summary of the model's strengths and potential avenues for enhancement.

---

## 🧠 Model Architecture Summary

The neural network is built using the Keras Sequential API:

| Layer (type) | Output Shape | Param # | Purpose |
| :--- | :--- | :--- | :--- |
| `Embedding` | `(None, None, 5)` | 235 | Learns a vector representation for each note integer. |
| `Conv1D` (Dilation 1) | `(None, None, 32)` | 352 | Captures immediate temporal patterns (t and t-1). |
| `BatchNormalization` | `(None, None, 32)` | 128 | Stabilizes activations. |
| `Conv1D` (Dilation 2) | `(None, None, 48)` | 3,120 | Expands receptive field to look back further (t and t-2). |
| `Conv1D` (Dilation 4, 8, 16) | ... | 43,496 | Exponentially grows the receptive field for mid-range context. |
| `Dropout` | `(None, None, 128)` | 0 | Regularization. |
| `LSTM` | `(None, None, 256)` | 394,240 | Captures long-term musical structure and dependencies. |
| `Dense` (`softmax`) | `(None, None, 47)` | 12,079 | Outputs probabilities for the next note in the vocabulary. |
| **Total Params** | | **454,794** | |

The architecture is designed so that the **Conv1D layers** handle the short- to medium-term dependencies (e.g., chord voicings, local voice leading), while the **LSTM** handles the complex, long-range harmonic movement and formal structure of the chorale.

## 🚀 Future Improvements

* **Advanced Architecture**: Replace the Conv1D+LSTM structure with a more modern **Transformer-based architecture** for better global context modeling.
* **Creative Sampling**: Introduce **temperature sampling** or **top-p (nucleus) sampling** during generation to control the creativity and randomness of the output.
* **Richer Data**: Incorporate additional music metadata such as **voice separation**, **note durations**, and **key signatures** into the input features.
* **Automated Export**: Implement functionality to automatically export generated outputs as standard **MIDI files**.
* **Interactive UI**: Develop a web interface (e.g., with Gradio or Streamlit) for interactive, real-time music generation.
