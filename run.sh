set -ex

docker build -t total-perspective-vortex .

docker run --rm -it -v $PWD:/app total-perspective-vortex python3 $@