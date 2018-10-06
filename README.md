# meats_detector

## annotation
```
pyenv virtualenv 3.6.5 meat_detector
pyenv local meat_detector
pip install -r requirements.txt
```

move videos into datasources/movies/

run
```
python data/split_videos.py \\
--source_path datasource/movies/<video_name>
--target_dir datasource/processed/<video_name> \\
--num_fps 1
```

