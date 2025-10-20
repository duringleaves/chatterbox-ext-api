# UI.md

# Overview: 
A user interface designed to access the Chatterbox TTS API we've created.  Targeted for internal end users generating AI content for radio broadcast voiceover workflows.

I've provided a previous implementation in the /Ursula directory to analyze and use for guidance on new features or functionality.  Do not depend on this content as it will be deleted later.  You may need to implement some of this functionality into this new API.

# DATA:
- Station Formats: Radio stations conform to formats that inform the music they play, the types of shows they syndicate, and other branding elements.  Formats include [ac, chr, classichits, classicrock, country, hotac, rock, urban, urbanac].  Supported formats are named as json files in ./data/script_templates/
- Station Forms: Radio station producers will fill in or submit a form with the values for various keywords for their station, including [stationCalls, slogan, morningShow,mobileApp, etc]. While the values will change depending on the station and format, there will be plenty of overlap between the keys for each format.  Samples (for viewing and for testing use in the UI) are located in ./data/sample_stations/
- VO Kit Scripts: Each station format uses a templated script for VO talent.  This takes the Station Form input and inserts those values into different phrases for a VO talent to read.  Sample scripts for analysis and testing are located in ./data/sample_scripts/

# VOICES:
Chatterbox AI supports several methods of generating speech audio:
- TTS: Given a text phrase, an audio reference file, and a small set of parameters, it generates an audio file of the phrase in the sound of the reference voice.
- Voice Cloning: Give two audio reference files (a recording to clone in another voice, and a reference file of the targeted voice), it generates a new file speaking the original recording in the new voice.
- REFERENCES:
    - ./data/reference_voices/<VO Artist>/<read_style>/<reference_audio>.wav
    - Reference audio collections contain multiple recordings from a single VO Artist. While we will offer multiple read_styles in the future, right now we'll limit to "VO Kit" for simplicity.
    - The reference files are named to be similar to the templated script lines. As you can see in Ursula, we select an audio reference file that most closely matches the line of text we're generating.  We need this behavior in this API as well.
    - Each read_style directory has a series of JSON files that correspond to chatterbox TTS parameters.  As you can see in Ursula, some script lines may contain tags like [fast], [slow], or [intense] that correspond to a json file. There is also a "chatterbox_settings.json" that is used for untagged lines and provides the default settings to use for this reference voice and read_style.
- CLONES:
    - ./data/clone_voices/<CLONED NAME>/<clone_reference_audio>.wav
    - Cloned audio collections will contain reference recordings of various people. If there are multiple audio files in this directory, one can be selected randomly for each clone request.

# USER EXPERIENCE:
The web app you design should feel clean and modern, with a bold color scheme, intuitive icons and labels, and displaying only the most important options by default.  (Advanced options can be hidden or collapsed until needed.)

At the top, a user should enter a "password" which will be used as an API-KEY for the API.  We'll need to expand the API to support authorization for the endpoints, and the api key should be sent with requests.

The next module in the UI should allow the user to configure the input source.  There should be two tabs: VO KIT and VOICE CLONE
- VO KIT
    - There will be a few ways to generate the script for the VO Kit generation. These can be tabs at the top of this module:
        - Import txt/docx file.  This will be a ready-made script.
        - Select Sample script: The user can also choose to use one of the sample_scripts/ from the directory
        - Select Station Format: The user can select a Station Format, then either fill in the form variables for that format as found in script_templates/ or choose one of the sample_stations/ to fill in the form with test data.
    - Once the script has been loaded, there should be a button to analyze it using ChatGPT to edit and conform better for our text to speech needs. (see Ursula).  There will also be options for the VO REFERENCE (the reference_voice/ used to generate the initial TTS) and VO CLONE (the clone_voices/ voice used by the Cloning step.)
    - Once analyzed (or parsed wihtout analysis), each line of the script should be shown in a clean spreadsheet-like grid. Each row will have the following columns, though this can be on multiple lines per row to keep things groups and organized nicely:
        - Section (eg. Station Calls, Legal Calls, etc from the script headers)
        - Text (the text string that will be sent to the TTS AI. Let the user edit as needed)
        - Read Style
        - Tag (fast, slow, etc. The user shold be able to change this to select ay supported tag)
        - Reference audio (the selected reference file for this line.  The user should be able to change this to select ANY of the reference files, and preview the selected file.)
        - Generate, preview, and download each single line. (Supports testing and iterating.)
    - The user should also be able to generate ALL the lines (that have not yet been individually generated) with a single button.  As per Ursula, as each line is generated, the UI should update so that the user can preview and/or re-generate the lines if they are not good.  The user should also be able to cancel the generate request, clearing the queue and stopping the generation.
    - Once happy, the user should be able to download the entire collection as a zip (per Ursula.  see that the zip contains a concatenated version of the renders, individual files for each line, and the raw TTS audio before cloning.)
- VOICE CLONE
    - The user should be able to upload a recording, select one of the clone_voices, and quickly render a voice clone from the uploaded file.  The only parameter needed would be Pitch.
    - The user can preview the rendered audio and download.
