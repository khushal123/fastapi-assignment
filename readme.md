# Problem

Here, at Oxus.AI we provide a service that looks for certain phrases in audio files. We constantly improve our ML models and want to provide most accurate predictions for our customers. For that we need to constantly deploy new  ML model versions with zero-to-minmal service downtime.

# Assignment

Implement a service that stores information about audio files, provides access to this information and periodically updates this data with updates from an included deep learning model.

## Data

- Audio files for testing when generating predictions are in Google Cloud Platform bucket `gs://example-recordings`; (access should have been added to your email user)
- Module that provides model inference results for an audio (`inference.py`);

## Requirements

### Model predictions

Implement a service that can use `inference.py` to generate confidence scores for audio files. The module runs an API which takes in a path to an audio file and runs a  model on the contents of that file. The API doc can be found at `/docs` when the API is running. The API returns a list of model confidences for each second in the given file and an utterance (these could be probabilities that a specific word exists in the given second).

Our models constantly improve because we want to ensure the best accuracy and provide the best results to our customers.
That is why results should be re-generated at certain time intervals, imitating using new model versions (for homework case it could be every 2 minutes).

### Database

Implement a database architecture to store the model predictions. Infer database schema from other requirements.

### REST API

Your implemented solution must expose a REST API that provides:

- List of audio files (names);
- The length of an audio file;
- The latest confidence scores for an audio file (generated by `inference.py`);

E.g.:
```json
{
	"file": "file_name_1.wav",
	"duration": 3000,
	"confidences": [
		{
			"utterance": "utterance_1",
			"time": 5,
			"confidence": 0.94322
		},
		{
			"utterance": "utterance_2",
			"time": 27,
			"confidence": 0.64322
		}
	]
}
```

# Suggestions

- Your solution should use Python software development best practices.
- There should be an easy way to run your solution.
- README is a perfect place to provide instructions for how to run your solution.
- The solution should be local, i.e., it should not make network requests when making the inference (predicting).
- A short documentation of design decisions and assumptions can be provided in the code itself or in the README.
- Time is not constrained, feel free to solve it whenever you're able to. Just don't forget to communicate with us if you can't find a free evening within a couple of weeks :)

# Evaluation Criteria

Your solution will be evaluated based on how well all requirements are implemented and how well you follow software engineering best practices.
Keep in mind, that you do not have to implement them all to submit the homework.

# Bonus Points

We understand that you have skills that you couldn't show us in this homework, so we provide some ideas for what you can do to show off your skills. These bonus points are not mandatory - do them only if they capture your strengths or if you have extra time to spare on this task and want to try something new. These bonus tasks are mutually independent and not in order. Here are the bonus tasks:

- Now you worked with fixed audio dataset. How would you solve adding additional audio files on demand?
- Deploy the solution on a cloud service.
- Currently we have only one model per utterance. But in the future we would like to run additional models side by side and get confidence score from each of them. Implement A/B testing solution with multiple running models. Confidence scores should be tracked along with the model that made predictions.
- Model performance monitoring. From time to time we will retrain the model and we would like to monitor if nothing broke. You have a controll dataset (original audio dataset privided) and implement model monitoring.