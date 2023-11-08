<img src="./public/logo.gif"  />

# MIA (Marvelously Incompetent Assistant)

- MIA is a smart, AI-powered robot that can help you with your day-to-day tasks. With its powerful computer vision and natural language processing capabilities, MIA can understand your commands and provide you with personalized assistance.
- Uses OpenAI's Whisper and GPT API to transcribe and respond to conversations
- Uses Google's Sheets API to log data in workbooks

## Features

- Voice control: Use your voice to control MIA and ask it to perform tasks for you.
- Natural language processing: MIA can understand your commands and respond to them appropriately.
- Personalized recommendations: MIA can learn your preferences over time and provide you with personalized recommendations.

## Getting Started

To get started with MIA, follow these steps:

- Clone this repository to your local machine.
- Install the required dependencies using `pip install -r requirements.txt`.
- Run the `test1_mIA.py` and `test1_ears.py` script
- Enjoy using MIA!

## To-Do

### MIA

- [ ] Add deep sleep functions
    - 1. If > N tokens then summarize conversation until now in advance to then directly load on hotword detection
    - 2. Remind users of upcoming events based on conversations saved or documents retrived
    - 3. Randomnly ping the user based on what you think they're doing currently
- [ ] Add logic to upsert relevant data from user 
- [ ] Update logic for past conversation summary (sumamrize + window)?
- [ ] Add ability to access the internet
- [ ] Add user voice recognition
- [ ] Add voice emotion detection
- [x] Add ability to put MIA to sleep when not needed
- [x] Save previous conversations in a vector database to serve as memory
    - No a normal conversation will remain in the JSON. Only factual information about user is to be upserted to PineCone.
- [x] Add voice using 11Labs API
- [x] Add instant listening on first play
- [x] Hotword detection

### EARS

- [ ] Add a function which adds actual date and day information if detetced in transcript
- [ ] Create knowledge graph from upserted data?
  - Needs further research
- [ ] Add metadata to documents being upserted
- [ ] Add a way to pause without stopping script
- [ ] Figure out how to crop out silent portions during TIMEOUT_LENGTH
- [ ] Check for error of audio file < 0.1 s
    - Need to test further
- [x] Improve the fact retrieval logic from transcripts
- [x] Instead of upserting the combined transcript, pull factual information from it then upsert
- [x] Have GPT account for the fact that the transcription may include multiple speakers
    - Needs further testing to confirm
- [x] Find out how to send recorded audio directly to Whisper without saving
  - Needs further research
  - Can't seem to do it with io.Bytes or converting to numpy array
- [x] Ensure any existing docs are combined and upserted before deleting full folder
- [x] Combine multiple small transcript txts into single large then upsert
    - If length of current transcript txt < N characters block upsert and if sum of lengths of previous N transcript txts > N characters then allow upsert
    - Had to change this to time between trasncriptions instead of length
- [x] Treat single word transcriptions as errors and stop saving
  - Problem is if you just said "hi"
    - Allowing single word since time gap shouldn't merge related transcripts together
- [x] Add Whisper prompt to not fill silence with hallucinations
- [x] Split recorded audio such that no file size is >25 MB
- [x] Continuous listening/recording with separate thread for transcribing, file saving and upserting

## Other

- [ ] Batch export Google Recorder transcripts into the vector database

## Contributing

If you'd like to contribute to MIA, please fork this repository and submit a pull request with your changes. We welcome contributions of all kinds, including bug fixes, new features, and improvements to the documentation.

## Support

If you have any questions or issues with MIA, please email us at [Nikhil Nair](mailto:niknair31898@gmail.com?subject=[MIA-Help]) or visit my [website](https://nikhil-nair.web.app/).

## License

This project is not licensed. Go away.
