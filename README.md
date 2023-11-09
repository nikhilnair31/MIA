<img src="./other/logo.gif"  />

# MIA (Marvelously Incompetent Assistant)

- MIA is an AI companion
- Uses OpenAI's Whisper and GPT, 11Labs, Porcupine and PineCone's APIs

## Getting Started

To get started with MIA, follow these steps:

- Clone this repository to your local machine.
- Install the required dependencies using `pip install -r requirements.txt`.
- Add a `.env` containing the following
    - THRESHOLD = 20
    - TIMEOUT_LENGTH = 4
    - DEEPSLEEPIN = 10
    - CHANNELS = 1
    - SWIDTH = 2
    - RATE = 44000
    - CHUNK = 1024
    - MAX_REC_TIME = 11
    - INDEX_NAME = "<insert here>"
    - CHUNK_SIZE = 256
    - CHUNK_OVERLAP = 16
    - SPEECH_GAP_DELAY = 60
    - OPENAI_API_KEY = "<insert here>"
    - PORCUPINE_API_KEY = "<insert here>"
    - ELEVENLABS_API_KEY = "<insert here>"
    - PINECONE_API_KEY = "<insert here>"
    - PINECONE_ENV_KEY = "gcp-starter"
- Run the `EARS.py` script
    - This records all conversations, transcribes them and eventually upserts to PineCone  
- Run the `MIA.py` 
    - This can be interacted with using voice
- Enjoy!

## Contributing

If you'd like to contribute to MIA, please fork this repository and submit a pull request with your changes. I welcome contributions of all kinds, including bug fixes, new features, and improvements to the documentation.

## Support

If you have any questions or issues with MIA, please email me at [Nikhil Nair](mailto:niknair31898@gmail.com?subject=[MIA-Help]) or visit my [website](https://nikhil-nair.web.app/).

## License

This project is not licensed. Go away.