LOCAL_DIR=/home/user/dev/aipipeline/cv-usecase/pipeline-manager/data

docker run -d --rm \
  --shm-size=2g \
  --ipc=shareable \
  --name=pipeline-manager \
  -v $LOCAL_DIR:/data \
	cv-pipeline-manager:v1 \
	python server.py --servers detector tracker classifier

docker run -d --rm \
	--gpus all \
  --ipc=container:pipeline-manager \
	detector-component:v1 \
	python server.py --service detector --next_service tracker

docker run -d --rm \
	--gpus all \
  --ipc=container:pipeline-manager \
	tracker-component:v1 \
	python server.py --service tracker --next_service classifier

docker run -it --rm \
	--gpus all \
  --ipc=container:pipeline-manager \
	classifier-component:v1 \
	python server.py --service classifier