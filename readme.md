# To run the code

We have two options, docker and without docker.
To run on local, we need to create locals.env file in bin folder
copy the content from locals.env.example and replace the values.
Postgresql server should be running.

- python3.9 -m pip install -r requirements.txt
- To run the migrations
  alembic upgrade heads
- to run the server
  bin/devserver

## Data

For now, audio file's relative path to the working directory should be passed in detect api call.  
For example [localhost:8000/api/detect/?utterance=recorded&audio_path=audio/dialog.mp3](localhost:8000/api/detect/?utterance=recorded&audio_path=audio/dialog.mp3)

The passed audio file path will be treated as unique name, to store the predictions, so if a same filename passed again, the code will get the id and store the new predictions. 
We can get the list of predictions by [localhost:8000/api/confidence/0/10](localhost:8000/api/confidence/0/10)
The second paramter is limit parameter, its just a reference, it can be implemented. 
