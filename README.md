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

- [ ] Restructure MIA'S BRAIN
    - [ ] Split into 1. User Facing 2. Internal Thoughts 3. Knowledge Base
    - [ ] 
- [ ] Add ability to put MIA to sleep when not needed
- [ ] Add `summarize` function that takes in full conversation with window size for actual conversations
- [ ] Save previous conversations in a vector database to serve as memory
- [ ] Add ability to access the internet
- [x] Add voice using 11Labs API
- [x] Add instant listening on first play
- [x] Hotword detection

### EARS

- [ ] Create knowledge graph from upserted data?
  - Needs further research
- [ ] Find out how to send recorded audio directly to Whisper without saving
  - Needs further research
- [ ] Figure out how to crop out silent portions during TIMEOUT_LENGTH
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
