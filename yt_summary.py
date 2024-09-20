import streamlit as st
import yt_dlp
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import time
import os
import logging
from pydub import AudioSegment

st.set_page_config(layout="wide")

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Function to convert audio to wav format
def convert_to_wav(input_path):
    try:
        # Extract file extension
        file_ext = os.path.splitext(input_path)[-1]

        if file_ext != '.wav':
            output_path = input_path.replace(file_ext, '.wav')
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format='wav')
            logging.info(f"Converted audio to WAV: {output_path}")
            return output_path
        else:
            return input_path
    except Exception as e:
        logging.error(f"Error converting file to WAV: {e}")
        st.error(f"Error converting file to WAV: {e}")
        return None

# Function to download the video using yt-dlp
def download_video(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',  # Download the best audio format available
            'outtmpl': 'downloaded_audio.%(ext)s',  # Output filename template
            'quiet': True  # Suppress verbose output for cleaner logs
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            file_name = ydl.prepare_filename(info_dict)  # Extract the downloaded filename
            
            # Verify if the file exists
            if not os.path.exists(file_name):
                st.error("Downloaded file not found.")
                return None
            
            # Convert to wav if needed
            file_name = convert_to_wav(file_name)
            if not file_name:
                return None

            return file_name
    except yt_dlp.utils.DownloadError as e:
        st.error(f"Download error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Initialize the Llama model for summarization
def initialize_model(full_path):
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )

# Initialize the prompt node for summarization
def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

# Transcribe audio using Whisper and summarize with the prompt node
def transcribe_audio(file_path, prompt_node):
    try:
        # Ensure the file exists
        if not os.path.exists(file_path):
            st.error("Audio file not found. Please check the file path.")
            return None
        
        logging.debug(f"Transcribing audio file: {file_path}")
        
        whisper = WhisperTranscriber()
        pipeline = Pipeline()
        pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
        pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])

        output = pipeline.run(file_paths=[file_path])
        logging.debug(f"Transcription output: {output}")
        return output
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        logging.error(f"Transcription error: {e}")
        return None

# Main function for Streamlit app
def main():
    st.title("YouTube Video Summarizer 🎥")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built with the Llama 2 🦙, Haystack, Streamlit')

    # Expander for app details
    with st.expander("About the App"):
        st.write("This app allows you to summarize while watching a YouTube video.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start.")

    # Input box for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL")

    # Submit button
    if st.button("Submit") and youtube_url:
        start_time = time.time()

        # Download the video using yt-dlp
        file_path = download_video(youtube_url)
        if file_path is None:
            st.error("Video download failed. Please try with a different video.")
            return

        # Initialize the Llama model and prompt node
        full_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
        model = initialize_model(full_path)
        prompt_node = initialize_prompt_node(model)

        # Transcribe the audio and get the summarization output
        output = transcribe_audio(file_path, prompt_node)
        if output is None:
            st.error("Transcription failed. Please try again.")
            return

        end_time = time.time()
        elapsed_time = end_time - start_time

        col1, col2 = st.columns([1, 1])

        with col1:
            st.video(youtube_url)

        with col2:
            st.header("Summarization of YouTube Video")
            st.write(output)
            if "results" in output:
                st.success(output["results"][0].split("\n\n[INST]")[0])
            st.write(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
