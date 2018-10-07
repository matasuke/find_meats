# meats_detector

## annotation

### image preparation
Use python3 for annotation.
```
pip install 'pip==18.0'
pip install pipenv
pipenv install
```

move all videos to be edit into data/movies/

run
```
pipenv run python meats_detector/data/split_videos.py
```

all splitted frames are saved in data/processed/

### annotation tool

Clone labelImg.
```
git clone https://github.com/tzutalin/labelImg.git
```

run command below
```
docker run -it \
--user $(id -u) \
-e DISPLAY=unix$DISPLAY \
--workdir=$(pwd) \
--volume="/home/$USER:/home/$USER" \
--volume="/etc/group:/etc/group:ro" \
--volume="/etc/passwd:/etc/passwd:ro" \
--volume="/etc/shadow:/etc/shadow:ro" \
--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
-v /tmp/.X11-unix:/tmp/.X11-unix \
tzutalin/py2qt4
```

Open labelImg by

```
make qt4py2;./labelImg.py
```
